'''k-t space to x-f space.'''

import numpy as np

def kt2xf(kt, shift=False, time_axis=-1):
    '''k-t space to x-f space.

    Parameters
    ----------
    kt : array_like
        k-t space data.
    shift: bool, optional
        Perform fftshift when Fourier transforming.
    time_axis : int, optional
        Dimension that holds time data.

    Returns
    -------
    xf : array_like
        Corresponding x-f space data.
    '''

    # Do the transformin' (also move time axis to and fro)
    if not shift:
        return np.moveaxis(np.fft.fft(np.fft.ifft2(
            np.moveaxis(kt, time_axis, -1),
            axes=(0, 1)), axis=-1), -1, time_axis)


    return np.moveaxis(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        np.fft.ifftshift(np.fft.fft(
            np.moveaxis(kt, time_axis, -1),
            axis=-1), axes=-1),
        axes=(0, 1)), axes=(0, 1)), axes=(0, 1)), -1, time_axis)
