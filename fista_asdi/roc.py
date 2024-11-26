"""
Modified version of EvalRoc from VIP and roc_updated_vip_roc from https://github.com/hazandaglayan/likelihoodratiomap
"""
#__author__ = "Modified by Nicolas Mil-Homens Cavaco"
#__all__ = 'EvalRoc'

"""
ROC curves generation.
"""

import numpy as np
import matplotlib.pyplot as plt
from hciplot import plot_frames
from scipy import stats
from photutils.segmentation import detect_sources
from munch import Munch

import pandas as pd

from vip_hci.fm import cube_inject_companions
from vip_hci.var import frame_center
from vip_hci.config import time_ini, timing, Progressbar

import vip_hci as vip
from vip_hci.var import frame_center, get_annulus_segments, get_circle
import os

from vip_hci.fits import open_fits, write_fits
import time

class EvalRoc(object):
    """
    Class for the generation of receiver operating characteristic (ROC) curves.
    """
    def __init__(self, asdi_dataset, plsc=0.0272, n_injections=100, inrad=8, file_mode='fits',
                 outrad=12, dist_flux=("uniform", 2, 500), mask=None, folder='output/'):
        """
        [...]
        dist_flux : tuple ('method', *args)
            'method' can be a string, e.g:
                ("skewnormal", skew, mean, var)
                ("uniform", low, high)
                ("normal", loc, scale)
            or a function.
        [...]
        """
        self.dataset = asdi_dataset
        self.plsc = plsc
        self.n_injections = n_injections
        self.inrad = inrad
        self.outrad = outrad
        self.dist_flux = dist_flux
        self.mask = mask
        self.methods = []

        self.file_mode = file_mode
        self.folder = folder

    def add_algo(self, name, algo, color, symbol, thresholds, folder, **kwargs):
        """
        Parameters
        ----------
        algo : HciPostProcAlgo
        thresholds : list of lists

        """
        label = kwargs.get("label", None)
        detection_map = kwargs.get("detection_map", "detection_map")
        self.methods.append(Munch(algo=algo, name=name, color=color, label=label, detection_map=detection_map,
                                  symbol=symbol, thresholds=thresholds, folder=folder))


    def save_file(self, data, filename):
        if self.file_mode == 'csv':
            pd.DataFrame(data).to_csv(filename+'.csv', header=None, index=None)
        elif self.file_mode == 'fits':
            write_fits(filename+'.fits', data, precision=np.float64, verbose=False)

    def read_file(self, filename):
        if self.file_mode == 'csv':
            return pd.read_csv(filename+'.csv').to_numpy(dtype=np.float64)
        elif self.file_mode == 'fits':
            return open_fits(filename, verbose=False)
    
    @staticmethod
    def draw_grid(dists, fwhm, cx, cy):
        angles = {}
        for sep_i in np.arange(*dists):
            source_rad = sep_i*fwhm
            sourcex, sourcey = vip.var.pol_to_cart(source_rad, 0, cx=cx, cy=cy)
            sep = vip.var.dist(cy, cx, float(sourcey), float(sourcex))
            angle = np.arcsin(fwhm/sep)*1
            number_apertures = int(np.floor(2*np.pi/angle))
        
            yy = np.zeros((number_apertures))
            xx = np.zeros((number_apertures))
            cosangle = np.cos(angle)
            sinangle = np.sin(angle)
            xx[0] = sourcex - cx
            yy[0] = sourcey - cy
            for ap in range(number_apertures-1):
                xx[ap+1] = cosangle*xx[ap] + sinangle*yy[ap]
                yy[ap+1] = cosangle*yy[ap] - sinangle*xx[ap]                 
            xx += cx
            yy += cy
            angles[sep_i] = np.array([vip.var.cart_to_pol(xx[ap], yy[ap], cx=cx, cy=cy)[1] for ap in range(number_apertures)])
        return angles
    
    def inject_and_postprocess_grid(self, i, scale_factor_algo='linear'):

        fwhm = self.dataset.mean_fwhm
        cy,cx = self.dataset.init_center
        inrad, outrad = int(self.inrad/fwhm), int(self.outrad/fwhm)
        angles = self.draw_grid((inrad, outrad), fwhm, cx, cy)
        flux = self.dist_flux[1]

        aperture_id_start = 0
        for k in np.arange(inrad, i):
            aperture_id_start += len(angles[k])
        
        rad = i*fwhm
        eps = np.array([fwhm])
        mask = self.dataset.mask_operator(self.dataset.cube, rad-eps, rad+eps)[0]
        sigma = np.std(self.dataset.cube[:, :, mask], axis=(1,2))

        for j in range(len(angles[i])):
            myPlanet = {'name': "myPlanet", 
                        'flux': flux*sigma, 
                        'dist': rad, 
                        'angle': angles[i][j]}
            
            self.dataset.cancel_preproc()
            self.dataset.inject_companions(myPlanet, n_branches=1)
            coy, cox = self.dataset.injections['myPlanet']['yx']
            cox = int(np.round(cox))
            coy = int(np.round(coy))

            self.dataset.find_scale_vector(scale_factor_algo=scale_factor_algo)
            self.dataset.preproc()   

            for m in self.methods:
                t1 = time.perf_counter()
                m.algo.run()
                t2 = time.perf_counter()
                elapsed_time = t2-t1
                folder = self.folder+"%s_flux%.2g_planet%d/" % (m.name,flux,aperture_id_start+j) 
                
                if not os.path.exists(folder): os.makedirs(folder)
                m.algo.save_outputs(folder)
                self.save_file(np.array(self.dataset.injections['myPlanet']['yx']), folder+"planet_pos")
                self.save_file(np.array(myPlanet['flux']), folder+"planet_flux")
                self.save_file(np.array([myPlanet['angle']]), folder+"planet_angle")
                self.save_file(np.array([myPlanet['dist']]), folder+"planet_dist")
                self.save_file(np.array([elapsed_time]), folder+'time')
                if isinstance(m.algo.params, tuple):
                    self.save_file(np.array([m.algo.params[0]]), folder+"rank")
                    self.save_file(np.array(m.algo.params[1:]), folder+"alpha")
                else:
                    self.save_file(np.array([m.algo.params]), folder+"rank")

            self.dataset.cancel_preproc()
            self.dataset.planet_free("myPlanet")



    def inject_and_postprocess(self, patch_size=0, cevr=0.9, scale_factor_algo='linear',
                               expvar_mode='annular', nproc=1, seed=1):
        # """
        # Notes
        # -----
        # # TODO `methods` are not returned inside `results` and are *not* saved!
        # # TODO order of parameters for `skewnormal` `dist_flux` changed! (was [3], [1], [2])
        # # TODO `save` not implemented

        # """
        starttime = time_ini()

        self.dataset.find_scale_vector(scale_factor_algo=scale_factor_algo)

        # Empty cube (testing FP and TN)
        if self.n_injections == 0:
            for m in self.methods:
                t1 = time.perf_counter()
                m.algo.run()
                t2 = time.perf_counter()
                elapsed_time = t2-t1
                folder = self.folder+"empty_%s_%d/" % (m.name,0) 
                if not os.path.exists(folder): os.makedirs(folder)
                self.save_file(m.algo.frame_final, folder+"frame_final")
                self.save_file(m.algo.detection_map, folder+"detection_map")
                self.save_file(m.algo.estimated_flux, folder+"estimated_flux")
                self.save_file(np.array([elapsed_time]), folder+'time')

        if self.n_injections > 0:
            # Injections (testing TP and FN)
            # Defining Fluxes according to chosen distribution
            self.dists = []
            self.thetas = []        

            dist_fkt = dict(skewnormal=stats.skewnorm.rvs,
                            normal=np.random.normal,
                            uniform=np.random.uniform).get(self.dist_flux[0],
                                                           self.dist_flux[0])

            self.fluxes = dist_fkt(*self.dist_flux[1:], size=self.n_injections)
            self.fluxes.sort()

            np.random.seed(seed)
            for rad in np.linspace(self.inrad, self.outrad, self.n_injections):
                theta = np.random.randint(0,360)
                self.dists.append(rad)
                self.thetas.append(theta)
            
            for n in Progressbar(range(self.n_injections), desc="injecting"):
                
                eps = np.array([1.5*self.dataset.mean_fwhm])
                mask = self.dataset.mask_operator(self.dataset.cube, self.dists[n]-eps, self.dists[n]+eps)[0]
                sigma = np.std(self.dataset.cube[:, :, mask], axis=(1,2))

                myPlanet = {'name': "myPlanet", 
                            'flux': self.fluxes[n]*sigma, 
                            'dist': self.dists[n], 
                            'angle': self.thetas[n]}
                
                print(myPlanet)
                
                self.dataset.cancel_preproc()
                self.dataset.inject_companions(myPlanet, n_branches=1)
                coy, cox = self.dataset.injections['myPlanet']['yx']
                cox = int(np.round(cox))
                coy = int(np.round(coy))

                self.dataset.find_scale_vector(scale_factor_algo=scale_factor_algo)
                self.dataset.preproc()   

                for m in self.methods: 
                    dist = myPlanet['dist'] #/self.dataset.mean_fwhm
                    t1 = time.perf_counter()
                    m.algo.run()
                    t2 = time.perf_counter()
                    elapsed_time = t2-t1
                    folder = self.folder+"%s_flux%.2g_planet%d/" % (m.name,self.fluxes[n],(seed-1)*self.n_injections+n) 
                    
                    if not os.path.exists(folder): os.makedirs(folder)
                    m.algo.save_outputs(folder)
                    self.save_file(np.array(self.dataset.injections['myPlanet']['yx']), folder+"planet_pos")
                    self.save_file(np.array(myPlanet['flux']), folder+"planet_flux")
                    self.save_file(np.array([myPlanet['angle']]), folder+"planet_angle")
                    self.save_file(np.array([dist]), folder+"planet_dist")
                    self.save_file(np.array([elapsed_time]), folder+'time')
                    if isinstance(m.algo.params, tuple):
                        self.save_file(np.array([m.algo.params[0]]), folder+"rank")
                        self.save_file(np.array(m.algo.params[1:]), folder+"alpha")
                    else:
                        self.save_file(np.array([m.algo.params]), folder+"rank")

                self.dataset.cancel_preproc()
                self.dataset.planet_free("myPlanet")

        timing(starttime)

    def save_algo_data(self):
        if not os.path.exists(self.folder): os.makedirs(self.folder)
        for m in self.methods:
            m.algo.save_outputs(self.folder)

    
    def read_data(self, flux=2, estimated_flux=False, seed=1):
        for m in self.methods:
            m.frames = []
            m.probmaps = []
            m.fluxes = []
            m.planet_angle = []
            m.planet_dist = []
            m.planet_flux = []
            m.planet_yx = []
            m.rank = []
            m.alpha = []
            m.time = []
            for n in range(self.n_injections):
                folder = m.folder+"%s_flux%.2g_planet%d/" % (m.name,flux,(seed-1)*self.n_injections+n) 
                m.frames.append(self.read_file(folder+"frame_final"))
                m.probmaps.append(self.read_file(folder+m.detection_map))
                m.planet_dist.append(self.read_file(folder+"planet_dist"))
                m.planet_angle.append(self.read_file(folder+"planet_angle"))
                m.planet_flux.append(self.read_file(folder+"planet_flux"))
                m.planet_yx.append(self.read_file(folder+"planet_pos"))
                m.time.append(self.read_file(folder+"time"))
                m.rank.append(self.read_file(folder+"rank"))

                if not estimated_flux:
                    m.fluxes.append(self.read_file(folder+"estimated_flux"))
                
                else:
                    m.alpha.append(self.read_file(folder+"alpha"))
                    m.fluxes.append(np.zeros((self.dataset.W, self.dataset.N, self.dataset.N)))
                    m.fluxes[n][:, :] = np.array(self.read_file(folder+"estimated_flux")).T.reshape((self.dataset.W, self.dataset.N, self.dataset.N))

            m.fluxes = np.array(m.fluxes)  
            m.frames = np.array(m.frames)
            m.probmaps = np.array(m.probmaps)     


    def compute_tpr_fpr_sqrt(self, debug=False, rin=3, rout=14, **kwargs):
        """
        Calculate number of dets/fps for every injection/method/threshold.

        Take the probability maps and the desired thresholds for every method,
        and calculates the binary map, number of detections and FPS using
        ``compute_binary_map``. Sets each methods ``detections``, ``fps`` and
        ``bmaps`` attributes.

        Parameters
        ----------
        **kwargs : keyword arguments
            Passed to ``compute_binary_map``

        """       
                
        def __remove_blob(frame_loc, blob_xy, fwhm):
            frame_mask = get_circle(np.ones_like(frame_loc), radius=fwhm/2,
                                        cy=blob_xy[1], cx=blob_xy[0],
                                        mode="mask")
            if ~np.isnan(frame_loc[frame_mask==1]).any():
                frame_loc[frame_mask==1] = np.nan
                return True
            else:
                return False
        def __remove_blob_enforce(frame_loc, blob_xy, fwhm):
            frame_mask = get_circle(np.ones_like(frame_loc), radius=fwhm/2,
                                        cy=blob_xy[1], cx=blob_xy[0],
                                        mode="mask")

            frame_loc[frame_mask==1] = np.nan
            return True

        # This function is taken from the inner funcion of vip_hci.metrics.roc.compute_binary_map
        def __overlap_injection_blob(injection, fwhm, blob_mask):

            injection_mask = get_circle(np.ones_like(blob_mask), radius=fwhm/2,
                                        cy=injection[1], cx=injection[0],
                                        mode="mask")

            intersection = injection_mask & blob_mask
            smallest_area = min(blob_mask.sum(), injection_mask.sum())
            return intersection.sum() / smallest_area

        cy,cx = frame_center(self.dataset.cube_origin)

        print('Evaluating injections:')
        fwhm = self.dataset.mean_fwhm
        for m in self.methods:
            injections = [[m.planet_yx[i][::-1]] for i in range(self.n_injections)]
            total_det = np.zeros((1,len(m.thresholds)))
            total_fps = np.zeros((1,len(m.thresholds)))
            total_false = np.zeros((1,len(m.thresholds)))
            
            starttime = time_ini()
            for i in Progressbar(range(self.n_injections)):       
                frame_loc = np.array(m.probmaps[i])
                cy, cx = frame_center(frame_loc)
                #_, n = frame_loc.shape
                mask = get_annulus_segments(np.ones_like(frame_loc), rin*fwhm, rout*fwhm, mode="mask")
                frame_loc[mask[0]==0] = np.nan
                for j in range(len(injections[0])):
                    __remove_blob_enforce(frame_loc, (injections[i][j][0], injections[i][j][1]), fwhm)
                            
                falses = []
                for sep_i in np.arange(rin, rout): #, int((n/2)/fwhm)):#range(10,11):
                    source_rad = sep_i*fwhm
                    sourcex, sourcey = vip.var.pol_to_cart(source_rad, 0.0, cx=cx, cy=cy)
                    sep = vip.var.dist(cy, cx, float(sourcey), float(sourcex))
                    angle = np.arcsin(fwhm/sep)*1
                    number_apertures = int(np.floor(2*np.pi/angle))
                
                    yy = np.zeros((number_apertures))
                    xx = np.zeros((number_apertures))
                    cosangle = np.cos(angle)
                    sinangle = np.sin(angle)
                    xx[0] = sourcex - cx
                    yy[0] = sourcey - cy
                    for ap in range(number_apertures-1):
                        xx[ap+1] = cosangle*xx[ap] + sinangle*yy[ap]
                        yy[ap+1] = cosangle*yy[ap] - sinangle*xx[ap]                 
                    xx += cx
                    yy += cy
                    for r_i in range(0,number_apertures):
                        if __remove_blob(frame_loc, (xx[r_i],yy[r_i]), fwhm):
                            falses.append((xx[r_i],yy[r_i]))

                list_detections = []
                list_fps = []
                list_false = []

                #color_circ = ("grey",)*len(falses)
                #plot_frames(m.probmaps[i], 
                #            circle=tuple(falses)+tuple(injections[i]), circle_alpha=1, circle_radius=fwhm/2, 
                #            circle_color=(*color_circ, "red"), cmap='viridis', colorbar=False) # save="ROC_resol_elem.pdf", 
                 
                for ithr, threshold in enumerate(m.thresholds):
                    if debug:
                        print("\nprocessing threshold #{}: {}".format(ithr + 1, threshold))
                
                    segments = detect_sources(np.array(m.probmaps[i]), threshold-0.0001, 1, connectivity=4)
                    detections = 0
                    fps = 0
                    if segments is not None:
                        binmap = (segments.data != 0)
                        for injection in injections[i]:
                            overlap = __overlap_injection_blob(injection, fwhm, binmap)
                            if overlap > 0.0:
                                if debug:
                                    print("\toverlap of {}! (+1 detection)".format(overlap))
                                detections += 1

                        for false in falses:
                            overlap = __overlap_injection_blob(false, fwhm, binmap)
                            if overlap > 0.0:
                                fps += 1 
                        #print(falses)
                        if debug:
                            print(fps, falses)
                            color_circ = ("grey",)*len(falses)
                            plot_frames(binmap, 
                                        circle=tuple(falses)+tuple(injections[i]), circle_alpha=1, circle_radius=fwhm/2, 
                                        circle_color=(*color_circ, "red"), cmap='hot', colorbar=False) # save="ROC_resol_elem.pdf", 
                            return

                    list_detections.append(detections)
                    list_fps.append(fps)
                    list_false.append(len(falses))
                
                total_det   += np.array(list_detections)
                total_fps   += np.array(list_fps)
                total_false += np.array(list_false)

            m.sqrt_tpr = np.sqrt(total_det/self.n_injections)
            m.sqrt_fpr = np.sqrt(total_fps/total_false)
            timing(starttime)
            # return total_false, total_det, total_fps
            

    def plot_roc_curves_sqrt(self, dpi=100, figsize=(5, 5), xmin=0, xmax=1,
                        ymin=-0.05, ymax=1.02, xlog=False, label_skip_one=False,
                        legend_loc='lower right', legend_size=12,
                        show_data_labels=False, hide_overlap_label=True,
                        label_gap=(0, -0.028), save_plot=False, label_params={},
                        line_params={}, marker_params={}, verbose=True):
        # """
        # Parameters
        # ----------


        # Returns
        # -------
        # None, but modifies `methods`: adds .tpr and .mean_fps attributes

        # Notes
        # -----
        # # TODO: load `roc_injections` and `roc_tprfps` from file (`load_res`)
        # # TODO: print flux distro information (is it actually stored in inj?
        # What to do with functions, do they pickle?)
        # # TODO: hardcoded `methodconf`?

        # """
        labelskw = dict(alpha=1, fontsize=5.5, weight="bold", rotation=0,
                        annotation_clip=True)
        linekw = dict(alpha=0.2)
        markerkw = dict(alpha=0.5, ms=3)
        labelskw.update(label_params)
        linekw.update(line_params)
        markerkw.update(marker_params)
        n_thresholds = len(self.methods[0].thresholds)

        if verbose:
            print('{} injections'.format(self.n_injections))
            # print('Flux distro : {} [{}:{}]'.format(roc_injections.flux_distribution,
            # roc_injections.fluxp1, roc_injections.fluxp2))
            print('Annulus from {} to {} pixels'.format(self.inrad,
                                                        self.outrad))

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)

        if not isinstance(label_skip_one, (list, tuple)):
            label_skip_one = [label_skip_one]*len(self.methods)
        labels = []

        # methodconf = {"CADI": dict(color="#d62728", symbol="^"),
        #              "PCA": dict(color="#ff7f0e", symbol="X"),
        #              "LLSG": dict(color="#2ca02c", symbol="P"),
        #              "SODIRF": dict(color="#9467bd", symbol="s"),
        #              "SODINN": dict(color="#1f77b4", symbol="p"),
        #              "SODINN-pw": dict(color="#1f77b4", symbol="p")
        #             }  # maps m.name to plot style

        for i, m in enumerate(self.methods):

            if not hasattr(m, "sqrt_tpr") or not hasattr(m, "sqrt_fpr"):
                raise AttributeError("method #{} has no sqrt_tpr/sqrt_fpr. Run"
                                     "`compute_tpr_fpr` first.".format(i))
            
            # print(m.sqrt_fpr, m.sqrt_tpr)

            plt.plot(m.sqrt_fpr[0], m.sqrt_tpr[0], '--', color=m.color, **linekw)
            plt.plot(m.sqrt_fpr[0], m.sqrt_tpr[0], m.symbol, label=m.label, color=m.color,
                     **markerkw)

            if show_data_labels:
                if label_skip_one[i]:
                    lab_x = m.sqrt_fpr[0][1::2]
                    lab_y = m.sqrt_tpr[0][1::2]
                    thr = m.thresholds[1::2]
                else:
                    lab_x = m.sqrt_fpr[0]
                    lab_y = m.sqrt_tpr[0]
                    thr = m.thresholds

                for i, xy in enumerate(zip(lab_x + label_gap[0],
                                           lab_y + label_gap[1])):
                    labels.append(ax.annotate('{:.2f}'.format(thr[i]),
                                  xy=xy, xycoords='data', color=m.color,
                                              **labelskw))
                    # TODO: reverse order of `self.methods` for better annot.
                    # z-index?

        plt.legend(loc=legend_loc, prop={'size': legend_size})
        if xlog:
            ax.set_xscale("symlog")
        plt.ylim(ymin=ymin, ymax=ymax)
        plt.xlim(xmin=xmin, xmax=xmax)
        plt.ylabel(r'$\sqrt{\rm TPR}$', fontsize=16)
        plt.xlabel(r'$\sqrt{\rm FPR}$', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(alpha=0.4)

        if show_data_labels:
            mask = np.zeros(fig.canvas.get_width_height(), bool)

            fig.canvas.draw()

            for label in labels:
                bbox = label.get_window_extent()
                negpad = -2
                x0 = int(bbox.x0) + negpad
                x1 = int(np.ceil(bbox.x1)) + negpad
                y0 = int(bbox.y0) + negpad
                y1 = int(np.ceil(bbox.y1)) + negpad

                s = np.s_[x0:x1, y0:y1]
                if np.any(mask[s]):
                    if hide_overlap_label:
                        label.set_visible(False)
                else:
                    mask[s] = True

        if save_plot:
            if isinstance(save_plot, str):
                plt.savefig(save_plot, dpi=dpi, bbox_inches='tight')
            else:
                plt.savefig('roc_curve_sqrt.pdf', dpi=dpi, bbox_inches='tight')

        
    def plot_detmaps(self, i=None, id='flux', thr=9, dpi=100,
                     axis=True, grid=False, vmin=-10, vmax='max',
                     plot_type="horiz"):
        """
        Plot the detection maps for one injection.

        Parameters
        ----------
        i : int or None, optional
            Index of the injection, between 0 and self.n_injections. If None,
            takes the 30st injection, or if there are less injections, the
            middle one.
        thr : int, optional
            Index of the threshold.
        dpi, axis, grid, vmin, vmax
            Passed to ``pp_subplots``
        plot_type : {"horiz" or "vert"}, optional
            Plot type.

            ``horiz``
                One row per algorithm (frame, probmap, binmap)
            ``vert``
                1 row for final frames, 1 row for probmaps and 1 row for binmaps

        """
        # input parameters
        if i is None:
            if len(self.list_xy) > 30:
                i = 30
            else:
                i = len(self.list_xy) // 2

        if vmax == 'max':
            # TODO: document this feature.
            vmax = np.concatenate([m.frames[i] for m in self.methods if
                                   hasattr(m, "frames") and
                                   len(m.frames) >= i]).max()/2

        # print information
        print('X,Y: {}'.format(self.list_xy[i]))
        print('dist: {:.3f}, flux: {:.3f}'.format(self.dists[i],
                                                  self.fluxes[i]))
        print()

        if plot_type in [1, "horiz"]:
            for m in self.methods:
                print('detection state: {} | false postives: {}'.format(
                    m.detections[i][thr], m.fps[i][thr]))
                labels = ('{} frame'.format(m.name), '{} S/Nmap'.format(m.name),
                          'Thresholded at {:.1f}'.format(m.thresholds[thr]))
                plot_frames((m.frames[i] if len(m.frames) >= i else
                            np.zeros((2, 2)), m.probmaps[i], m.bmaps[i][thr]),
                            label=labels, dpi=dpi, horsp=0.2, axis=axis,
                            grid=grid, cmap=['viridis', 'viridis', 'gray'])

        elif plot_type in [2, "vert"]:
            labels = tuple('{} frame'.format(m.name) for m in self.methods if
                           hasattr(m, "frames") and len(m.frames) >= i)
            plot_frames(tuple(m.frames[i] for m in self.methods if
                        hasattr(m, "frames") and len(m.frames) >= i),
                        dpi=dpi, label=labels, vmax=vmax, vmin=vmin, axis=axis,
                        grid=grid)

            plot_frames(tuple(m.probmaps[i] for m in self.methods), dpi=dpi,
                        label=tuple(['{} S/Nmap'.format(m.name) for m in
                                     self.methods]), axis=axis, grid=grid)

            for m in self.methods:
                msg = '{} detection: {}, FPs: {}'
                print(msg.format(m.name, m.detections[i][thr], m.fps[i][thr]))

            labels = tuple('Thresholded at {:.1f}'.format(m.thresholds[thr])
                           for m in self.methods)
            plot_frames(tuple(m.bmaps[i][thr] for m in self.methods),
                        dpi=dpi, label=labels, axis=axis, grid=grid,
                        colorbar=False, cmap='bone')
        else:
            raise ValueError("`plot_type` unknown")


def compute_binary_map(frame, thresholds, injections, fwhm, npix=1,
                       overlap_threshold=0.7, max_blob_fact=2, plot=False,
                       debug=False):
    """
    Take a list of ``thresholds``, create binary maps and counts detections/fps.
    A blob which is "too big" is split into apertures, and every aperture adds
    one 'false positive'.

    Parameters
    ----------
    frame : numpy ndarray
        Detection map.
    thresholds : list or numpy ndarray
        List of thresholds (detection criteria).
    injections : tuple, list of tuples
        Coordinates (x,y) of the injected companions. Also accepts 1d/2d
        ndarrays.
    fwhm : float
        FWHM, used for obtaining the size of the circular aperture centered at
        the injection position (and measuring the overlapping with found blobs).
        The circular aperture has 2 * FWHM in diameter.
    npix : int, optional
        The number of connected pixels, each greater than the given threshold,
        that an object must have to be detected. ``npix`` must be a positive
        integer. Passed to ``detect_sources`` function from ``photutils``.
    overlap_threshold : float
        Percentage of overlap a blob has to have with the aperture around an
        injection.
    max_blob_fact : float
        Maximum size of a blob (in multiples of the resolution element) before
        it is considered as "too big" (= non-detection).
    plot : bool, optional
        If True, a final resulting plot summarizing the results will be shown.
    debug : bool, optional
        For showing optional information.

    Returns
    -------
    list_detections : list of int
        List of detection count for each threshold.
    list_fps : list of int
        List of false positives count for each threshold.
    list_binmaps : list of 2d ndarray
        List of binary maps: detection maps thresholded for each threshold
        value.

    """
    def _overlap_injection_blob(injection, fwhm, blob_mask):
        """
        Parameters
        ----------
        injection: tuple (y,x)
        fwhm : float
        blob_mask : 2d bool ndarray

        Returns
        -------
        overlap_fact : float between 0 and 1
            Percentage of the area overlap. If the blob is smaller than the
            resolution element, this is ``intersection_area / blob_area``,
            otherwise ``intersection_area / resolution_element``.

        """
        if len(injections[0]) > 0:
            injection_mask = get_circle(np.ones_like(blob_mask), radius=fwhm,
                                        cy=injection[1], cx=injection[0],
                                        mode="mask")
        else:
            injection_mask = np.zeros_like(blob_mask)
        intersection = injection_mask & blob_mask
        smallest_area = min(blob_mask.sum(), injection_mask.sum())
        return intersection.sum() / smallest_area

    # --------------------------------------------------------------------------
    list_detections = []
    list_fps = []
    list_false = []
    list_binmaps = []
    sizey, sizex = frame.shape
    cy, cx = frame_center(frame)
    reselem_mask = get_circle(frame, radius=fwhm, cy=cy, cx=cx, mode="val")
    npix_circ_aperture = reselem_mask.shape[0]

    # normalize injections: accepts combinations of 1d/2d and tuple/list/array.
    injections = np.asarray(injections)
    if injections.ndim == 1:
        injections = np.array([injections])

    for ithr, threshold in enumerate(thresholds):
        if debug:
            print("\nprocessing threshold #{}: {}".format(ithr + 1, threshold))

        segments = detect_sources(frame, threshold, npix, connectivity=4)
        detections = 0
        fps = 0
        false = 0
        if segments is not None:
            binmap = (segments.data != 0)

            if debug:
                plot_frames((segments.data,), cmap=('tab20b',),
                            circle=tuple(tuple(xy) for xy in injections),
                            circle_radius=fwhm/2, circle_alpha=1, circle_color="white",
                            label=("segmentation map",), label_size=16, save="segment_map.pdf")
                return 

            if debug:
                plot_frames((segments.data, binmap), cmap=('tab20b', 'binary'),
                            circle=tuple(tuple(xy) for xy in injections),
                            circle_radius=fwhm, circle_alpha=0.6,
                            label=("segmentation map", "binary map"))

            for segment in segments.segments:
                label = segment.label
                blob_mask = (segments.data == label)
                blob_area = segment.area

                if debug:
                    lab = "blob #{}, area={}px**2".format(label, blob_area)
                    plot_frames(blob_mask, circle_radius=fwhm/2, circle_alpha=0.6,
                                circle=tuple(tuple(xy) for xy in injections),
                                cmap='binary', label_size=8, label=lab,
                                size_factor=3)
                    return 

                for iinj, injection in enumerate(injections):
                    if len(injections[0]) > 0:  # checking injections is not empty
                        if injection[0] > sizex or injection[1] > sizey:
                            raise ValueError("Wrong coordinates in `injections`")

                        if debug:
                            print("\ttesting injection #{} at {}".format(iinj + 1,
                                                                        injection))

                    if blob_area > max_blob_fact * npix_circ_aperture:
                        number_of_apertures_in_blob = blob_area / npix_circ_aperture
                        fps += number_of_apertures_in_blob  # float, rounded at end
                        false += number_of_apertures_in_blob
                        if debug:
                            print("\tblob is too big (+{:.0f} fps)"
                                "".format(number_of_apertures_in_blob))
                            print("\tskipping all other injections")
                        # continue with next blob, do not check other injections
                        break

                    overlap = _overlap_injection_blob(injection, fwhm, blob_mask)
                    if overlap > overlap_threshold:
                        if debug:
                            print("\toverlap of {}! (+1 detection)"
                                "".format(overlap))

                        detections += 1
                        # continue with next blob, do not check other injections
                        break

                    if debug:
                        print("\toverlap of {} -> do nothing".format(overlap))

                else:
                    if debug:
                        print("\tdid not find a matching injection for this "
                            "blob (+1 fps)")
                    fps += 1
                    false += 1

            if debug:
                print("done with threshold #{}".format(ithr))
                print("result: {} detections, {} fps".format(detections, fps))
            
            false += len(injections) - detections

        fps = np.round(fps).astype(int).item()  # -> python `int`
        false = np.round(false).astype(int).item()

        list_detections.append(detections)
        list_binmaps.append(binmap)
        list_fps.append(fps)
        list_false.append(false)

    if plot:
        labs = tuple(str(det) + ' detections' + '\n' + str(fps) +
                     ' false positives' for det, fps in zip(list_detections,
                                                            list_fps))
        if len(injections[0]) > 0:
            circles = tuple(tuple(xy) for xy in injections)
        else:
            circles = None
        plot_frames(tuple(list_binmaps), title='Final binary maps', label=labs,
                    label_size=8, cmap='binary', circle_alpha=0.8,
                    circle=circles, circle_radius=fwhm,
                    circle_color='deepskyblue', axis=False)

    return list_detections, list_fps, list_false, list_binmaps



def _create_synt_cube(cube, psf, ang, plsc, dist, flux, theta=None,
                      verbose=False):
    """
    """
    centy_fr, centx_fr = frame_center(cube[0])
    if theta is None:
        np.random.seed()
        theta = np.random.randint(0, 360)

    posy = dist * np.sin(np.deg2rad(theta)) + centy_fr
    posx = dist * np.cos(np.deg2rad(theta)) + centx_fr
    if verbose:
        print('Theta:', theta)
        print('Flux_inj:', flux)
    cubefc = cube_inject_companions(cube, psf, ang, flevel=flux, plsc=plsc,
                                    rad_dists=[dist], n_branches=1, theta=theta,
                                    verbose=verbose)
    return cubefc, posx, posy

    #fig, ax = fig.subplots()

    
    #if sqrt:
    #    plt.plot(np.sqrt((total_fprl2l1/25).T),np.sqrt(total_tprl2l1.T), '-', label=label, markersize=5)#, marker=next(marker)), color=cmap(vers-1))
    #    plt.ylabel(r'$\sqrt{\rm TPR}$')
    #    plt.xlabel(r'$\sqrt{\rm FPR}$')
    #else:
    #    plt.plot((total_fprl2l1/25).T,np.sqrt(total_tprl2l1.T), '-o',label=label, markersize=10)#, marker=next(marker))#, color=cmap(vers-1))
    #    plt.xlabel('FPR')
    #    plt.xlabel('TPR')
        
        
    #plt.legend()
    #return total_tpr, total_fpr
