"""
Modified version of EvalRoc from VIP and roc_updated_vip_roc 
from https://github.com/hazandaglayan/likelihoodratiomap
"""
__author__ = "Modified by Nicolas Mil-Homens Cavaco"
__all__ = ['EvalRoc',]
"""
ROC curves generation.
"""

import copy
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
