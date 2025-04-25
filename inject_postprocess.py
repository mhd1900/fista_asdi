"""
Injection of companion + post-processing + save data
"""

from fista_asdi.datacube import DataCubeASDI
from fista_asdi.algorithms import PCA, FISTA_ASDI
from fista_asdi.roc import EvalRoc

import numpy as np
import argparse

parser = argparse.ArgumentParser(description="TFE-main")

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
args = parser.parse_args()


def main(n_injections, algo_name="pca", algo_params=None, output_folder='output/',
         cube_name='sphere_ifs', cube_id='2', cube_folder="../../empty_cubes/", 
         adi_sampling_frame_nb=(None,None,1),sdi_sampling_frame_nb=(None,None,1), verbose=0,
         dist=4.0, flux=8.0, eps=1e-3, max_iter=200, seed=1, mask_rin=5, mask_rout=20, crop_frames_nb=90):
    
    algo_params = algo_params.split(",")
    sdi_sampling_frame_nb = sdi_sampling_frame_nb.split(',')
    adi_sampling_frame_nb = adi_sampling_frame_nb.split(',')
    for i in range(len(sdi_sampling_frame_nb)):
        if sdi_sampling_frame_nb[i] == 'None': sdi_sampling_frame_nb[i] = None
        else: sdi_sampling_frame_nb[i] = int(sdi_sampling_frame_nb[i])
    for i in range(len(adi_sampling_frame_nb)):
        if adi_sampling_frame_nb[i] == 'None': adi_sampling_frame_nb[i] = None
        else: adi_sampling_frame_nb[i] = int(adi_sampling_frame_nb[i])

    datacube = DataCubeASDI(cube="%s_cube_%s" % (cube_name, cube_id), 
                            wavelengths="%s_wls_%s" % (cube_name, cube_id), angles="%s_pa_%s" % (cube_name, cube_id),
                            psf="%s_psf_%s" % (cube_name, cube_id), pxscale="%s_pxscale_%s" % (cube_name, cube_id),
                            folder=cube_folder,
                            verbose=False, imlib='opencv', remove_bad_pixel=False,
                            adi_sampling_frame_nb=adi_sampling_frame_nb,
                            sdi_sampling_frame_nb=sdi_sampling_frame_nb,
                            mask_rin=mask_rin, mask_rout=mask_rout, 
                            crop_frames_number=crop_frames_nb)                
    
    datacube.find_scale_vector_preproc()

    dist_flux = ("uniform", *flux)
    inrad, outrad = dist[0]*datacube.mean_fwhm, dist[1]*datacube.mean_fwhm

    roc = EvalRoc(datacube, plsc=datacube.px_scale, 
                n_injections=n_injections, 
                inrad=inrad, outrad=outrad, folder=output_folder,
                dist_flux=dist_flux, file_mode='fits')
    
    if algo_name == 'fista': 
        algo = FISTA_ASDI(datacube, params=(int(algo_params[0]), float(algo_params[1])), stepsize='lipschitz', device='cpu', verbose=4,
                    tol=eps, MAX_ITER=max_iter, fast_gradient=True, output_folder=output_folder) 
    elif algo_name == 'pca':
        algo = PCA(datacube, params=int(algo_params[0]), stepsize=1.0, verbose=verbose, device='cpu', output_folder=output_folder) 
    
    ##################
    ## AREA TO EDIT ##
    ##############################################################################################################

    roc.add_algo(name=algo_name, algo=algo, color="#d62728", symbol="^", thresholds=np.array(np.arange(0,8,0.02)), folder=output_folder)
    roc.inject_and_postprocess_grid(i=seed, scale_factor_algo='linear')

    ##############################################################################################################

flux = args.flux.split(",")
flux = (float(flux[0]), float(flux[0])) if len(flux) == 1 else (float(flux[0]), float(flux[1]))
dist = args.dist.split(",")
dist = (float(dist[0]), float(dist[0])) if len(dist) == 1 else (float(dist[0]), float(dist[1]))

radius = args.mask_r.split(",")
mask_rin, mask_rout = float(radius[0]), float(radius[1])

main(args.n_injections, args.algo_name, cube_id=args.cube_id,
     algo_params=args.algo_params, output_folder=args.output_folder,
     adi_sampling_frame_nb=args.adi_sampling_frame_nb,
     sdi_sampling_frame_nb=args.sdi_sampling_frame_nb, 
     dist=dist,flux=flux, mask_rin=mask_rin, mask_rout=mask_rout, 
     seed=args.seed, cube_folder=args.cube_folder, crop_frames_nb=args.crop_frames_nb)
