__author__ = "Nicolas Mil-Homens Cavaco"
__all__ = ["prox_L21", 
           "Mask", 
           "Trajectorlet", 
           "Rotation",
           "Scaling"]

import torch 
import torch.nn.functional as F
import kornia.geometry.transform as kornia
import numpy as np
from vip_hci.var import frame_center

torch.set_default_dtype(torch.float64)

def prox_L21(x, alpha=0.0, eps=1e-16):
    # From prox repository: https://proximity-operator.net/
    norm_x = torch.sqrt(torch.sum(x*x, axis=0, keepdims=True))
    return x * torch.where(norm_x <= alpha, 0.0, 1-alpha/(norm_x+eps))

# Forward model operators
class Mask:
    """
    Mask operator : m x n -> m x k, such that |index| = k.
    """
    def __init__(self, index=None):
        self.index = index
    
    def apply(self, signal):
        return signal[:, self.index]

    def adjoint(self, signal, new_signal):
        new_signal[:] = 0.0 
        new_signal[:, self.index] = signal
        return new_signal

    def set(self, signal, new_signal):
        signal[:, self.index] = new_signal


class Trajectorlet:
    """
    Trajectorlet operator : W x N² -> WT x n²
    """
    def __init__(self, angles, scale_vec, psf, cube_center, cube_shape, device='cuda'):
        self.convolution = Convolution(psf, device=device).apply
        self.rotation = Rotation(angles, cube_center, device=device).apply
        self.scaling = Scaling(scale_vec, cube_center, cube_shape, device=device).apply
        self.scale_vec = torch.from_numpy(scale_vec)
        self.W, self.T, self.N, _ = cube_shape
        self.device = device

    def apply(self, signal, convolve=True):
        signal = signal.reshape((self.W, self.N, self.N))
        measured_signal = self.convolution(signal) if convolve else signal[:,None,:,:]
        measured_signal = self.scaling(measured_signal, t=0)
        measured_signal = self.rotation(measured_signal.repeat(1, self.T, 1, 1))
        n = measured_signal.shape[-1]
        return measured_signal.reshape(self.W*self.T, n*n)


class Convolution:
    """
    Convolve a torch signal of shape (W, N, N) with a torch PSF of shape (W, K, K).
    """
    def __init__(self, psf, device='cuda'):
        self.psf = torch.from_numpy(psf).to(device)

    def apply(self, signal, adjoint=False):
        W, N, _ = self.psf.shape
        convolved_results = []

        psf = self.psf.flip(dims=(1,2)) if not adjoint else self.psf

        # Loop through each image and its corresponding kernel
        for i in range(W):
            # Convolve signal[i] with psf[i]
            convolved = F.conv2d(signal[i:i+1, None], psf[i:i+1, None], padding='same')
            convolved_results.append(convolved)

        return torch.cat(convolved_results, dim=0)

        
class Rotation:
    """
    Rotate a torch datacube of shape (W, T, n, n) w.r.t a set of 2x2 rotation matrices
    """
    def __init__(self, angles, cube_center, device='cuda'):
        T = angles.shape[0]
        center_T = torch.tensor(cube_center)[None,:].repeat(T, 1).double()
        angles = torch.from_numpy(angles).double()
        ones_T = torch.ones((T, 2))

        self.Q = kornia.get_rotation_matrix2d(center_T, angles, ones_T).double().to(device)
        self.Qinv = kornia.get_rotation_matrix2d(center_T, -angles, ones_T).double().to(device)

    def apply(self, cube, inverse=False, mode='bilinear'):
        x, y = cube.shape[2:]
        rot_matrix = self.Q if not inverse else self.Qinv
        derotated_cube = kornia.warp_affine(torch.transpose(cube, 0, 1).double(), rot_matrix, (x, y), mode=mode)
        derotated_cube = torch.clamp(derotated_cube, min=float(cube.min()), max=float(cube.max()))
        return torch.transpose(derotated_cube, 0, 1)
    
    
class Scaling:
    """
    Scale a torch datacube of shape (W, T, N, N) w.r.t a set of 2x2 scaling matrices
    """
    def __init__(self, scale_vec, cube_center, cube_shape, device='cuda'):
        self.scale_vec = scale_vec

        W = scale_vec.shape[0]
        center_W = torch.tensor(cube_center)[None,:].repeat(W, 1).double()
        scale_vec = torch.from_numpy(scale_vec)[:,None].repeat(1, 2).double()
        zeros_W = torch.zeros(W)

        self.D = kornia.get_rotation_matrix2d(center_W, zeros_W, scale_vec).double().to(device)
        self.Dinv = kornia.get_rotation_matrix2d(center_W, zeros_W, 1/scale_vec).double().to(device)
        self.W, self.T, self.N, _ = cube_shape


    def apply(self, cube, t=0, inverse=False, mode='bilinear'):
        #                                                                                           
        # Modified version of cube_rescaling_wavelengths from VIP:
        # https://vip.readthedocs.io/en/latest/_modules/vip_hci/preproc/rescaling.html 
        #                                                                                                                

        ########################
        ## Apply Zero-padding ##
        ########################
        max_sc = np.amax(self.scale_vec)
        if not inverse and max_sc > 1:
            n, _, y, x = cube.shape
            new_y = int(np.ceil(max_sc * y))
            new_x = int(np.ceil(max_sc * x))
            if (new_y - y) % 2 != 0:
                new_y += 1
            if (new_x - x) % 2 != 0:
                new_x += 1
            pad_len_y = (new_y - y) // 2
            pad_len_x = (new_x - x) // 2
            pad_width = (pad_len_y, pad_len_y, pad_len_x, pad_len_x)
            big_cube  = F.pad(cube[:,t:t+1], pad=pad_width, mode='constant').double()
        else:
            big_cube  = cube[:,t:t+1].clone()

        #########################
        ## Apply downsc_matrix ##
        #########################
        y,x = big_cube.shape[2:] if not inverse else cube.shape[2:]
        downsc_matrix = self.D if not inverse else self.Dinv
        downscaled_cube = kornia.warp_affine(big_cube, downsc_matrix, (x, y), mode=mode)
        downscaled_cube = torch.clamp(downscaled_cube, min=float(cube.min()), max=float(cube.max()))

        cy, cx  = frame_center(big_cube) if not inverse else frame_center(cube)

        ###############################
        ## Focus on the small square ##
        ###############################
        if inverse and np.amax(self.scale_vec) > 1:
            if cube.shape[-1] > self.N:
                array_old = downscaled_cube.clone()
                downscaled_cube = torch.zeros([self.W, 1, self.N, self.N])
                for zz in range(self.W):
                    downscaled_cube[zz,0] = Scaling.get_square(array_old[zz,0], self.N, cy, cx)

        return downscaled_cube

    @staticmethod
    def get_square(array, size, y, x, position=False, force=False, verbose=True):
        #                                                                                           
        # Modified version of get_square from VIP:
        # https://vip.readthedocs.io/en/latest/_modules/vip_hci/var/shapes.html
        #         
        """
        Return an square subframe from a 2d array or image.

        Parameters
        ----------
        array : 2d numpy ndarray
            Input frame.
        size : int
            Size of the subframe.
        y : int or float
            Y coordinate of the center of the subframe (obtained with the function
            ``frame_center``).
        x : int or float
            X coordinate of the center of the subframe (obtained with the function
            ``frame_center``).
        position : bool, optional
            If set to True return also the coordinates of the bottom-left vertex.
        force : bool, optional
            Size and the size of the 2d array must be both even or odd. With
            ``force`` set to False, the requested size is flexible (i.e. +1 can be
            applied to requested crop size for its parity to match the input size).
            If ``force`` set to True, the requested crop size is enforced, even if
            parities do not match (warnings are raised!).
        verbose : bool optional
            If True, warning messages might be shown.

        Returns
        -------
        array_out : numpy ndarray
            Sub array.
        y0, x0 : int
            [position=True] Coordinates of the bottom-left vertex.

        """
        size_init_y = array.shape[0]
        size_init_x = array.shape[1]
        size_init = array.shape[0]  # "force" cases assume square input frame

        if not force:
            # Even input size
            if size_init % 2 == 0:
                # Odd size
                if size % 2 != 0:
                    size += 1
                    if verbose:
                        print("`Size` is odd (while input frame size is even). "
                            "Setting `size` to {} pixels".format(size))
            # Odd input size
            else:
                # Even size
                if size % 2 == 0:
                    size += 1
                    if verbose:
                        print("`Size` is even (while input frame size is odd). "
                            "Setting `size` to {} pixels".format(size))
        else:
            # Even input size
            if size_init % 2 == 0:
                # Odd size
                if size % 2 != 0 and verbose:
                    print("WARNING: `size` is odd while input frame size is even. "
                        "Make sure the center coordinates are set properly")
            # Odd input size
            else:
                # Even size
                if size % 2 == 0 and verbose:
                    print("WARNING: `size` is even while input frame size is odd. "
                        "Make sure the center coordinates are set properly")

        # wing is added to the sides of the subframe center
        wing = (size - 1) / 2

        y0 = int(y - wing)
        y1 = int(y + wing + 1)  # +1 cause endpoint is excluded when slicing
        x0 = int(x - wing)
        x1 = int(x + wing + 1)

        if y0 < 0 or x0 < 0 or y1 > size_init_y or x1 > size_init_x:
            # assuming square frames
            raise RuntimeError('square cannot be obtained with size={}, y={}, x={}'
                            ''.format(size, y, x))

        array_out = array[y0:y1, x0:x1].clone()

        if position:
            return array_out, y0, x0
        else:
            return array_out