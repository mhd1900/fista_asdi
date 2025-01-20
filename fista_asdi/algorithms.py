__author__ = "Nicolas Mil-Homens Cavaco"
__all__ = ["AlgoASDI", 
           "PCA", 
           "FISTA_ASDI"]

import numpy as np

import torch
torch.set_default_dtype(torch.float64)

import numpy as np
import vip_hci as vip
from vip_hci.var import frame_center
from fista_asdi.optimizer import *
from fista_asdi.operators import *
import os

import time

np.random.seed(42)

class AlgoASDI:

    def __init__(self, asdi, params, output_folder='output/', verbose=False, **kwargs):

        self.verbose = verbose
        self.asdi = asdi
        self.output_folder = output_folder
        self.MAX_ITER = kwargs.get('MAX_ITER', 100)
        self.tol = kwargs.get('tol', 1e-3)
        self.stepsize = kwargs.get('stepsize', 'lipschitz')
        self.device = kwargs.get('device', 'cpu')
        self.nproc = kwargs.get('nproc', 4)
        self.params = params
        self.speckle_mask = self.asdi.truncated_coronagraph_corr_zone[0]

        self.oracle = None


    def run(self, save_folder=None):
        
        if self.verbose: print("starting post-processing")
        # Concatenate time and spectral dimensions
        shape = self.asdi.cube.shape
        cube = self.asdi.cube.reshape(shape[0]*shape[1], *shape[2:]) 

        t1 = time.perf_counter()
        residuals, flux, _, output_list = self.apply_algorithm(cube=cube, verbose=self.verbose,
                                                               device=self.device)
        
        
        residuals = residuals.reshape(self.asdi.W, self.asdi.T, self.asdi.N, self.asdi.N)

        t2 = time.perf_counter()
        if self.verbose: print("Total running time for post-processing:", t2-t1)

        # Apply derotation on each spectral frame
        self.compute_res_map_and_detection_map(residuals, flux)
        if save_folder is not None:
            self.save_outputs(save_folder)

        
    def apply_algorithm(self, **kwargs):
        cube = kwargs.get('cube')
        verbose = kwargs.get('verbose',False)
        device = kwargs.get('device', 'cpu')
        stepsize = self.stepsize

        L, S = self.oracle.init_weights(cube)

        lipschitz_constant = self.oracle.lipschitz_const(L.shape, S.shape)
        stepsize = 1./lipschitz_constant
        if verbose: print(stepsize)

        L,S,fval,i,output_list = optimize(self.oracle, L, S, stepsize, 
                                        MAX_ITER=self.MAX_ITER,tol=self.tol,
                                        verbose=verbose,output_list=None)
        # Compute final residual
        res = torch.zeros(((cube.shape[0], cube.shape[1]*cube.shape[2]))).to(device).double()
        self.remove_background(res, L)
        flux = S.detach().cpu().numpy()

        if cube.shape[0] == self.asdi.W*self.asdi.T:
            res = res.reshape(self.asdi.W, self.asdi.T, *cube.shape[1:]).detach().cpu().numpy()
            if verbose: print("Convergence after %d iterations" % (i))
            res_final = self.asdi.apply_downscaling(res, inverse=True) 
            return res_final, flux, i, output_list
        
        res_final = res.reshape((cube.shape[0],1,*cube.shape[1:]))

        if verbose: print("Convergence after %d iterations" % (i))
        return self.asdi.apply_downscaling(res_final.detach().cpu().numpy(), inverse=True), flux, i, output_list

    def remove_background(self, datacube, background):
        """
        Substract the background from a datacube. 
        /!\ Every child class inheriting this super class has to implement this function.
        """
        pass

    def compute_res_map_and_detection_map(self, residuals, flux):
        residuals = self.asdi.apply_rotation(residuals, inverse=True)
        self.frame_final = np.mean(residuals, axis=(0,1))
        self.estimated_flux = flux
        self.detection_map = self.compute_snr_map(self.frame_final)
        self.background_model = residuals

    def compute_snr_map(self, residuals):
        return vip.metrics.snrmap(residuals, fwhm=self.asdi.mean_fwhm, exclude_negative_lobes=False, nproc=self.nproc)
    
    def save_outputs(self, folder):
        if not os.path.exists(folder): os.makedirs(folder)
        vip.fits.write_fits(folder+"detection_map", self.detection_map, precision=np.float64, verbose=False)
        vip.fits.write_fits(folder+"frame_final", self.frame_final, precision=np.float64, verbose=False)
        vip.fits.write_fits(folder+"estimated_flux", self.estimated_flux, precision=np.float64, verbose=False)
         

class PCA(AlgoASDI):

    def __init__(self, asdi, params=None, output_folder="output/", verbose=False, **kwargs):

        super().__init__(asdi, params, output_folder=output_folder, 
                         verbose=verbose, MAX_ITER=0, **kwargs)
        
        device = kwargs.get('device', 'cpu')
        rank = params
        speckle_mask = self.speckle_mask

        class myOracle:                

            def init_weights(self, cube):
    
                # Init data matrix
                shape = cube.shape
                svd_mask = np.where(speckle_mask.flatten())[0]
                Z = cube.reshape((shape[0], shape[1]*shape[2]))[:,svd_mask]

                # Apply speckle mask on data and compute SVD
                u,s,vh = np.linalg.svd(Z, full_matrices=False)
                self.Z = torch.from_numpy(Z).to(device)
                self.Mask = Mask(torch.from_numpy(svd_mask).to(device))

                # Init variables
                L = np.dot(u[:,:rank], s[:rank,None]*vh[:rank,:])
                return torch.from_numpy(L).to(device), torch.zeros(1).to(device)
            
            def grad(self, L, S):
                return L, S
            
            def lipschitz_const(self, L_shape, S_shape):
                return 1.0
        
            def eval_obj(self, L, S):
                return 0.0

        self.oracle = myOracle()
    
    def remove_background(self, datacube, L):
        self.oracle.Mask.set(datacube, self.oracle.Z - L)


class FISTA_ASDI(AlgoASDI): 

    def __init__(self, asdi, params=(None,None),
                 output_folder="output/", 
                 verbose=False, **kwargs):
        
        super().__init__(asdi, params, output_folder=output_folder, verbose=verbose, **kwargs)
        shape = asdi.W,asdi.T,asdi.N,asdi.N
        device = kwargs.get('device', 'cpu')
        
        Psi = Trajectorlet(asdi.angles, asdi.scale_vec, asdi.psfn, frame_center(asdi.cube), shape, device=device)

        rank, alpha = params
        speckle_mask = self.speckle_mask
    
        class myOracle:                

            def init_weights(self, cube):
                
                # Init data matrix
                shape = cube.shape
                svd_mask = np.where(speckle_mask.flatten())[0]
                Z = cube.reshape((shape[0], shape[1]*shape[2]))[:,svd_mask]

                # Apply speckle mask on data and compute SVD
                u,s,vh = np.linalg.svd(Z, full_matrices=False)

                self.U = torch.from_numpy(u[:,:rank]).to(device)
                self.Z = torch.from_numpy(Z).to(device)
                self.Mask = Mask(torch.from_numpy(svd_mask).to(device))

                # Init variables
                X, S = torch.from_numpy((s[:rank,None]*vh[:rank,:]).T).to(device), torch.zeros((asdi.W, asdi.N*asdi.N)).to(device)
                return X, S

            def loss_func(self, X, S, no_cube=False):
                if no_cube: # Used in the power method to avoid evaluating the adjoint explicitely.
                    return 0.5*torch.sum((self.forward_operator(X, S))**2)
               
                return 0.5*torch.sum((self.Z - self.forward_operator(X, S))**2)
        
            def grad(self, X, S, no_cube=False):
                X.requires_grad_(); S.requires_grad_()
                X.grad, S.grad = None, None
                self.loss_func(X, S, no_cube).backward()
                X_grad, S_grad = X.grad, S.grad
                X.requires_grad_(False); S.requires_grad_(False)
                return X_grad, S_grad
            
            def regularizer(self, S):
                return alpha*torch.sum(torch.sqrt(torch.sum(S*S, axis=0)))  
            
            def forward_operator(self, X, S):
                return self.U @ X.T + self.Mask.apply(Psi.apply(S))
            
            def lipschitz_const(self, shape_X, shape_S):
                return power_method(shape_X, shape_S, self.grad)

            def eval_obj(self, X, S):
                return self.loss_func(X, S) + self.regularizer(S)
            
            def update(self, X, S, grad_X, grad_S, mu=None):
                kwargs = {'alpha': mu*alpha} 
                return X - mu*grad_X, prox_L21(S - mu*grad_S, **kwargs)
            
            def stop_crit(self, S, S_prev):
                return torch.norm(S-S_prev) / torch.norm(S)
            
        self.oracle = myOracle()
    
    def remove_background(self, datacube, X):
        self.oracle.Mask.set(datacube, self.oracle.Z - self.oracle.U @ X.T)

    def compute_res_map_and_detection_map(self, residuals, flux):
        residuals = self.asdi.apply_rotation(residuals, inverse=True)
        self.frame_final = np.mean(residuals, axis=(0,1))
        self.estimated_flux = flux
        self.detection_map = np.mean(flux, axis=0).reshape(self.asdi.N, self.asdi.N)
    
    def save_outputs(self, folder):
        if not os.path.exists(folder): os.makedirs(folder)
        vip.fits.write_fits(folder+"detection_map", np.mean(self.estimated_flux, axis=0).reshape(self.asdi.N, self.asdi.N), precision=np.float64, verbose=False)
        vip.fits.write_fits(folder+"frame_final", self.frame_final, precision=np.float64, verbose=False)
        vip.fits.write_fits(folder+"estimated_flux", self.estimated_flux, precision=np.float64, verbose=False)