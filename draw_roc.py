from fista_asdi.datacube import DataCubeASDI
from fista_asdi.algorithms import PCA, FISTA_ASDI
from fista_asdi.roc import EvalRoc
import vip_hci as vip
import os

import numpy as np
import argparse

parser = argparse.ArgumentParser(description="draw_roc")
parser.add_argument("output_folder", type=str)
parser.add_argument("n_injections", help='n_injections', type=int)
parser.add_argument("algo_name", help="name of the algorithm", type=str)
parser.add_argument("algo_params", type=str)
parser.add_argument("adi_sampling_frame_nb", type=str)
parser.add_argument("sdi_sampling_frame_nb", type=str)
parser.add_argument("crop_frames_nb", type=int)
parser.add_argument("dist", type=str)
parser.add_argument("flux", type=str)
parser.add_argument("seed", type=int)
parser.add_argument("mask_r", type=str)
parser.add_argument("cube_folder", type=str)
parser.add_argument("cube_id", type=str)
parser.add_argument("save_roc_folder", type=str)
args = parser.parse_args()

colors=["#0000ff", "#00ff00", "#ff0000", "#000000", "#ff00ff", "#ffff00", "#00ffff"]*3
symbols=[".", "^", "o", ">"]*5

def import_data(algo_params=None,
         cube_name='sphere_ifs', cube_id='2', cube_folder="../../empty_cubes/", 
         adi_sampling_frame_nb_=(None,None,1),sdi_sampling_frame_nb_=(None,None,1),
         mask_rin=5, mask_rout=20, crop_frames_nb=90):
    
    sdi_sampling_frame_nb = sdi_sampling_frame_nb_.split(',')
    adi_sampling_frame_nb = adi_sampling_frame_nb_.split(',')
    algo_params = algo_params.split(",")
    
    for i in range(len(sdi_sampling_frame_nb)):
        if sdi_sampling_frame_nb[i] == 'None': sdi_sampling_frame_nb[i] = None
        else: sdi_sampling_frame_nb[i] = int(sdi_sampling_frame_nb[i])
    for i in range(len(adi_sampling_frame_nb)):
        if adi_sampling_frame_nb[i] == 'None': adi_sampling_frame_nb[i] = None
        else: adi_sampling_frame_nb[i] = int(adi_sampling_frame_nb[i])

    datacube = DataCubeASDI(cube="%s_cube_%s" % (cube_name, cube_id), 
                            wavelengths="%s_wls_%s" % (cube_name, cube_id), angles="%s_pa_%s" % (cube_name, cube_id),
                            psf="%s_psf_%s" % (cube_name, cube_id), pxscale="%s_pxscale_%s" % (cube_name, cube_id), 
                            folder=cube_folder,        # 7.57 for gpi
                            verbose=False, imlib='opencv', remove_bad_pixel=False,
                            adi_sampling_frame_nb=adi_sampling_frame_nb,
                            sdi_sampling_frame_nb=sdi_sampling_frame_nb,
                            mask_rin=mask_rin, mask_rout=mask_rout, 
                            crop_frames_number=crop_frames_nb)                
    
    datacube.find_scale_vector_preproc()
    return datacube

def create_roc_object(datacube, n_injections, algo_name="pca",
                      algo_params=None, output_folder='output/',dist=4.0, flux=8.0, eps=1e-3, max_iter=100):
    
    dist_flux = ("uniform", *flux)
    inrad, outrad = dist[0]*datacube.mean_fwhm, dist[1]*datacube.mean_fwhm
    roc = EvalRoc(datacube, plsc=datacube.px_scale, 
            n_injections=n_injections, 
            inrad=inrad, outrad=outrad, folder=output_folder,
            dist_flux=dist_flux, file_mode='fits')
    
    if algo_name == 'fista': 
        algo = FISTA_ASDI(datacube, params=(int(algo_params[0]), float(algo_params[1])), stepsize='lipschitz', device='cpu', verbose=4,
                    tol=eps, MAX_ITER=max_iter, fast_gradient=True, output_folder=output_folder) 
         
        roc.add_algo(name='fista', algo=algo, color="#ff0000", symbol="o", thresholds=np.array(np.arange(0,0.5,0.005)), 
                        folder=output_folder, label="S map", detection_map="detection_map") # [:-1]+'_v2' ### +algo_params[0]+algo_params[1]+"/" [:-1]+'_v2/' [:-1]+"_v2/"
            
    elif algo_name == 'pca':
        algo = PCA(datacube, params=int(algo_params[0]), stepsize=1.0, verbose=4, device='cpu', output_folder=output_folder) 
        roc.add_algo(name='pca', algo=algo, color="#000000", symbol="x", thresholds=np.array(np.arange(0,8.0,0.1)), 
                    folder=output_folder, label="PCA S/N", detection_map="detection_map") # +algo_params[0]+algo_params[1]+"/" output_folder[:-9]+'pca_16_v2/'
   
    roc.read_data(flux[0], estimated_flux=False)
    return roc

flux = args.flux.split(",")
flux = (float(flux[0]), float(flux[0])) if len(flux) == 1 else (float(flux[0]), float(flux[1]))
dist = args.dist.split(",")
dist = (float(dist[0]), float(dist[0])) if len(dist) == 1 else (float(dist[0]), float(dist[1]))

radius = args.mask_r.split(",")
mask_rin, mask_rout = float(radius[0]), float(radius[1])

dataset = import_data(cube_id=args.cube_id, 
                      algo_params=args.algo_params, 
                      adi_sampling_frame_nb_=args.adi_sampling_frame_nb,
                      sdi_sampling_frame_nb_=args.sdi_sampling_frame_nb, 
                      mask_rin=mask_rin, mask_rout=mask_rout,
                      cube_folder=args.cube_folder)


fluxes = (0.0, 0.25, 0.5)
roc = [None]*len(fluxes)
for i, f in enumerate(fluxes): # 0, 0.25, 0.5,  1,) 0, 0.25, 
     roc[i] = create_roc_object(dataset, args.n_injections, args.algo_name, 
                                algo_params=args.algo_params, 
                                output_folder=args.output_folder,
                                dist=dist,flux=(f,f))
     
for i in range(len(fluxes)):
    roc[i].compute_tpr_fpr_sqrt(debug=False, rin=3, rout=7)

if not os.path.exists(args.save_roc_folder): os.makedirs(args.save_roc_folder)

for i in range(len(fluxes)):
    for j, m in enumerate(roc[i].methods):
        vip.fits.write_fits(args.save_roc_folder+"tpr_sqrt_roc_%d_methods_%d.fits" % (i,j), m.sqrt_tpr, precision=np.float64, verbose=False)
        vip.fits.write_fits(args.save_roc_folder+"fpr_sqrt_roc_%d_methods_%d.fits" % (i,j), m.sqrt_fpr, precision=np.float64, verbose=False)
        vip.fits.write_fits(args.save_roc_folder+"thresholds_%d_methods_%d.fits" % (i,j), m.thresholds, precision=np.float64, verbose=False)

for i in range(len(fluxes)):
    roc[i].plot_roc_curves_sqrt(show_data_labels=True, save_plot=args.save_roc_folder+"roc_%d.pdf" % (i))
