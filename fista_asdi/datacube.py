__author__ = "Nicolas Mil-Homens Cavaco"
__all__ = ["DataCubeASDI"]

import vip_hci as vip
from vip_hci.fits import open_fits
from vip_hci.var import frame_center

import scipy.signal as sg

import numpy as np
import time
import gc


class DataCubeASDI:
    def __init__(self, nproc=1, folder=None, verbose=False, imlib='vip-fft', interpol="lanczos4",
                  adi_sampling_frame_nb=(None,None,1),
                  sdi_sampling_frame_nb=(None,None,1), 
                  mask_rin=1, mask_rout=14,
                  **kwargs):
        """
            <string> name: 
                Name of the telescope
            <string> id: 
                ID of the dataset corresponding to the selected telescope in 'name'
            <string> folder: 
                Folder containing the datasets
            <int> crop_frames_number: 
                If not None, crop the PSF frame by the value given by this parameter
            <boolean> verbose: 
                Print things for debug
            <string> imlib: 
                Gives the library used to perform interpolations on images 
            <boolean> inplace: 
                Specify if we want to override the original datacube by the preprocessed one.
                Should not be used if we want to massively inject companion
            <int> adi_sampling_frame_nb:
                Useful if we want to decrease the memory load of ADI frames. 
                Sample ADI frames 1 up to 'adi_sampling_nb'.
            <float> mask_rin, mask_rout: 
                inner and outer radius of the mask, allowing to select relevant entries of the datacube.
        """
        
        # Loading datacube
        temp_output = DataCubeASDI.load_ASDI_datacube(folder, **kwargs)
        dataset, self.fwhm, self.psfn, self.flux_st, self.px_scale = temp_output
        self.verbose = verbose

        # Select ADI frames
        begin, end, step = adi_sampling_frame_nb
        if begin is None: begin = 0
        if end is None: end = dataset.cube.shape[1]
        self.cube = dataset.cube[:, begin:end:step]      # ASDI cube
        self.angles = dataset.angles[begin:end:step]     # ADI data
        
        # Select SDI frames
        begin, end, step = sdi_sampling_frame_nb
        if begin is None: begin = 0
        if end is None: end = dataset.cube.shape[0]
        self.cube = self.cube[begin:end:step]
        self.wavelengths = dataset.wavelengths[begin:end:step]

        self.psfn = self.psfn[begin:end:step]
        self.fwhm = self.fwhm[begin:end:step]
        self.flux_st = self.flux_st[begin:end:step]

        self.mean_fwhm = np.mean(self.fwhm)

        self.W, self.T, self.N, _ = self.cube.shape
        self.init_center = vip.var.frame_center(self.cube)
        
        self.cube_origin = self.cube             # Store the orginal datacube in a hidden variable if not inplace
        
        self.imlib = imlib                       # Interpolation library
        self.interpol = interpol
        self.nproc = nproc                       # Number of processors for interpolation operators

        self.injections = {}                     # Injected companions characteristics
        self.is_preproc = False                  # Indicates if data are preprocessed (i.e. scaled)
        self.is_found_scale_vec = False          # Indicates if scale factor is computed

        self.mask_rin, self.mask_rout = mask_rin, mask_rout
    
    def find_scale_vector_preproc(self):
        self.find_scale_vector()
        self.preproc()

    
    def find_scale_vector(self):
        if self.is_preproc:
            if self.verbose: print("Cannot compute scale factor on a preprocessed ASDI datacube")
            return
        
        if self.is_found_scale_vec and self.verbose:
            print("Scale factor already computed")
            print("Recompute the scale factor...")
        elif self.verbose: 
            print("Computing scale factor...")
        
        self.scale_vec = self.wavelengths[-1]/self.wavelengths
        self.sc_flux = None
        if self.verbose: print(f"scale factor: {self.scale_vec}")
        self.is_found_scale_vec = True


    def preproc(self):
        """
        @params:
            The 'mask' allows to select relevant entries of the datacube.
            <float> rin:  inner radius of the mask 
            <float> rout: outer radius of the mask 
        """
        
        if self.is_preproc:
            if self.verbose: print("Cannot preprocess data twice")
            return 
    
        if not self.is_found_scale_vec:
            if self.verbose:
                print("Should compute the scale factor before preprocessing the data")
                print("use 'asdi.find_scale_vector()'")
            return
            
        if self.verbose: print("Preprocessing...")
        t1 = time.perf_counter()
        self.cube = self.apply_downscaling(self.cube)
        t2 = time.perf_counter()
        if self.verbose: print("scale time:", t2-t1)

        fwhm_ones_W = self.mean_fwhm*np.ones(self.W)
        self.truncated_coronagraph_corr_zone = DataCubeASDI.mask_operator(self.cube, Rin=self.mask_rin*fwhm_ones_W, Rout=self.mask_rout*fwhm_ones_W)
        self.truncated_coronagraph_corr_zone[np.abs(self.cube[:,0]) <= 1e-8] = 0
        self.is_preproc = True
 
    
    def cancel_preproc(self, forced_free_memory=False):
        """
        Kill the pointer pointing on the preprocessed data
        Memory is freed automaticaly via the gc of Python

        @pre: forced_free_memory: 
            If the memory of the preprocessed cube is not freed, you can try to set this parameter to True.
        """
        if forced_free_memory:
            del self.cube 
            gc.collect()

        self.cube = self.cube_origin  
        self.is_preproc = False
        

    @staticmethod
    def load_ASDI_datacube(folder, **kwargs):    
        """
        Load ASDI data, that is, the datacube, the psf, angles, wavelengths, ...
        Use **kwargs to provide the paths.

        @params:
            Int crop_frames_number: allow to crop the frame (subsample from the spatial dimension).
        """
        ds = vip.objects.Dataset(
            cube = folder+kwargs.get("cube"), 
            angles = folder+kwargs.get("angles"),
            psf = folder+kwargs.get("psf"),
            wavelengths= folder+kwargs.get("wavelengths")
        )
        crop_frames_number = kwargs.get("crop_frames_number", None)
        if crop_frames_number is not None: ds.crop_frames(crop_frames_number)              
        
        if kwargs.get("normalize", True):
            psfn, flux_st, fwhm = vip.fm.normalize_psf(ds.psf, fwhm='fit', full_output=True, debug=False, size=19, verbose=kwargs.get("verbose"))
        else:
            psfn = ds.psf
            flux_st = open_fits(folder+kwargs.get('flux'))
            fwhm = open_fits(folder+kwargs.get('fwhm'))
        
        px_scale = kwargs.get("pxscale")
    
        if kwargs.get("verbose"):
            print("\n\n", px_scale, "arcsec/px\n")
            print("Wavelength set :\n\t", ds.wavelengths)
    
        return ds, fwhm, psfn, flux_st, px_scale

    
    def inject_companions(self, planet, n_branches=1):
        """
        Allow to inject planets one by one
        Each time we want to inject a planet in the datacube, this function should be called.
        """
        planet_id = planet['name']
        flux = planet['flux']
        rad = planet['dist']
        theta = planet['angle']

        if self.is_preproc:
            if self.verbose: print("Cannot inject planet after preprocessing")
            return

        if self.verbose: print("\n\nInjecting companion...")
        cube, injections_yx = vip.fm.cube_inject_companions(self.cube, psf_template=self.psfn, angle_list=self.angles, flevel=flux, plsc=self.px_scale,
                                                            rad_dists=[rad], theta=theta, n_branches=n_branches, imlib=self.imlib, full_output=True)
        
        self.cube_origin = self.cube = cube
        self.injections[planet_id] = {"yx": injections_yx[0], "params": (rad, theta, flux)}
        if self.verbose: print(f"Companion injected at position {injections_yx}")

    
    def planet_free(self, planet_id):
        """
        Remove injected companions from the dataset
        """
        if self.is_preproc: 
            if self.verbose: print("Cannot remove planets once the cube is preprocessed.")
            return
        
        r, theta, flux = self.injections[planet_id]["params"]
        ones_W = np.ones(self.W)
        r = r*ones_W
        theta = theta*ones_W
        if isinstance(flux, (int, float, np.float64, np.int64, np.float32, np.int32)): 
            flux = flux*ones_W
            
        cube = vip.fm.utils_negfc.cube_planet_free((r, theta, flux), self.cube, self.angles, self.psfn, imlib=self.imlib,
                     interpolation=self.interpol, transmission=None)

        self.cube_origin = self.cube = cube
        del self.injections[planet_id]

    @staticmethod
    def mask_operator(cube, Rin, Rout):
        """
        Find the mask operator M_{(Rin, Rout)} of the cube

        @pre:
            4D array cube: an ASDI datacube on which the mask is applied
            1D array Rin: the inner-radius of the mask (for each wavelength)
            1D array Rout: the outer-radius of the mask (for each wavelength)

        @returns:
            4D array: cube whose values outside the mask was set to zero
            
        """
        dim = cube.shape[-1]
        X,Y = np.ogrid[:dim, :dim]
        cx, cy = frame_center(cube)
        d2 = (X-cx)**2 + (Y-cy)**2
        return np.array([(ro**2 >= d2)*(d2 >= ri**2) for ri, ro in zip(Rin, Rout)])
    
    def apply_rotation(self, cube, inplace=False, inverse=False):
        angles = self.angles if inverse else -self.angles
        derotated_cube = cube if inplace else np.zeros_like(cube)
        for lamb in range(self.W):
            derotated_cube[lamb] = vip.preproc.cube_derotate(cube[lamb], angles, imlib=self.imlib, nproc=self.nproc, interpolation=self.interpol) #, edge_blend='interp') blinear
        
        return derotated_cube

    def apply_downscaling(self, cube, t=None, inverse=False, imlib=None):
        """
        Rescale all the frames of a ASDI datacube according to the scale factor given in the 1D array sc_factor
        @pre:
            4D array datacube: datacube to rescale
            1D array sc_factor: scaling factor to apply for each wavelength
            bool inverse: if True, apply the inverse transform
            tuple yx_init: initial dimension of the frame before rescaling (needed to compute the inverse)

        @returns:
            4D array cube_pp: 
        """
        if imlib is None: imlib = self.imlib

        if t is not None:              
            cube_sc = DataCubeASDI._scale_operator(cube[:,t], t, inverse=inverse, interpol=self.interpol,
                                                    scale_vec=self.scale_vec, sc_flux=self.sc_flux, 
                                                    N=self.N, imlib=imlib)[0]
            return cube_sc[:, None, :, :]

        if self.is_preproc:
            cube_sc = np.zeros(self.cube.shape) if not inverse else np.zeros((self.W, self.T, self.N, self.N)) 
            
            for t in range(self.T):
                cube_sc[:,t,:,:] = DataCubeASDI._scale_operator(cube[:,t], t, inverse=inverse, interpol=self.interpol,
                                            scale_vec=self.scale_vec, sc_flux=self.sc_flux, 
                                            N=self.N, imlib=imlib)[0]
        
        else:
            cube_p = [None]*self.T
            for t in range(self.T):
                cube_p[t] = DataCubeASDI._scale_operator(cube[:,t], t, inverse=inverse, interpol=self.interpol,
                                            scale_vec=self.scale_vec, sc_flux=self.sc_flux, 
                                            N=self.N, imlib=imlib)[0]
            n = cube_p[0].shape[-1]
            cube_sc = np.zeros((self.W, self.T, n, n))

            for t in range(self.T):
                cube_sc[:,t,:,:] = cube_p[t]

        return cube_sc
        

    @staticmethod
    def _scale_operator(cube, t, inverse=False, scale_vec=None, sc_flux=None, N=None, imlib='vip-fft', interpol='lanczos4'):
        if inverse:
            cube_p = vip.preproc.cube_rescaling_wavelengths(cube, scale_vec, imlib=imlib, 
                                                interpolation=interpol, full_output=True, 
                                                inverse=True, y_in=N, x_in=N, pad_mode='constant')[0] # bilinear
            if sc_flux is not None: 
                cube_p = cube_p / sc_flux[:, None, None]
        else:
            cube_p = vip.preproc.cube_rescaling_wavelengths(cube, scale_vec, imlib=imlib, 
                                                interpolation=interpol, full_output=True, pad_mode='constant')[0] # blinear
            if sc_flux is not None: 
                cube_p = cube_p * sc_flux[:, None, None]

        return cube_p, t
    
    def apply_psf_convolution(self, signal, adjoint=False): 
        psf = self.psfn[:,::-1,::-1] if adjoint else self.psfn
        convolved_signal = sg.fftconvolve(signal, psf, mode='same', axes=(1,2))
        return convolved_signal
    

    def apply_trajectorlet(self, signal, convolve=True):
        signal = signal.reshape((self.W, self.N, self.N))
        measured_signal = self.apply_psf_convolution(signal) if convolve else signal
        measured_signal = self.apply_downscaling(measured_signal[:,None], t=0)
        measured_signal = self.apply_rotation(measured_signal.repeat(self.T, axis=1))
        n = measured_signal.shape[-1]
        return measured_signal.reshape(self.W*self.T, n*n)