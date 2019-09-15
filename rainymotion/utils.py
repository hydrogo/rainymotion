"""
The rainymotion library provides different utils to help to prepare
raw radar data (dBZ) for nowcasting.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def depth2intensity(depth, interval=300):
    """
    Function for convertion rainfall depth (in mm) to
    rainfall intensity (mm/h)

    Args:
        depth: float
        float or array of float
        rainfall depth (mm)

        interval : number
        time interval (in sec) which is correspondend to depth values

    Returns:
        intensity: float
        float or array of float
        rainfall intensity (mm/h)
    """
    return depth * 3600 / interval


def intensity2depth(intensity, interval=300):
    """
    Function for convertion rainfall intensity (mm/h) to
    rainfall depth (in mm)

    Args:
        intensity: float
        float or array of float
        rainfall intensity (mm/h)

        interval : number
        time interval (in sec) which is correspondend to depth values

    Returns:
        depth: float
        float or array of float
        rainfall depth (mm)
    """
    return intensity * interval / 3600


def RYScaler(X_mm):
    '''
    Scale RY data from mm (in float64) to brightness (in uint8).

    Args:
        X (numpy.ndarray): RY radar image

    Returns:
        numpy.ndarray(uint8): brightness integer values from 0 to 255
                              for corresponding input rainfall intensity
        float: c1, scaling coefficient
        float: c2, scaling coefficient

    '''
    def mmh2rfl(r, a=256., b=1.42):
        '''
        .. based on wradlib.zr.r2z function

        .. r --> z
        '''
        return a * r ** b

    def rfl2dbz(z):
        '''
        .. based on wradlib.trafo.decibel function

        .. z --> d
        '''
        return 10. * np.log10(z)

    # mm to mm/h
    X_mmh = depth2intensity(X_mm)
    # mm/h to reflectivity
    X_rfl = mmh2rfl(X_mmh)
    # remove zero reflectivity
    # then log10(0.1) = -1 not inf (numpy warning arised)
    X_rfl[X_rfl == 0] = 0.1
    # reflectivity to dBz
    X_dbz = rfl2dbz(X_rfl)
    # remove all -inf
    X_dbz[X_dbz < 0] = 0

    # MinMaxScaling
    c1 = X_dbz.min()
    c2 = X_dbz.max()

    return ((X_dbz - c1) / (c2 - c1) * 255).astype(np.uint8), c1, c2


def inv_RYScaler(X_scl, c1, c2):
    '''
    Transfer brightness (in uint8) to RY data (in mm).
    Function which is inverse to Scaler() function.

    Args:
        X_scl (numpy.ndarray): array of brightness integers obtained
                               from Scaler() function.
        c1: first scaling coefficient obtained from Scaler() function.
        c2: second scaling coefficient obtained from Scaler() function.

    Returns:
        numpy.ndarray(float): RY radar image

    '''
    def dbz2rfl(d):
        '''
        .. based on wradlib.trafo.idecibel function

        .. d --> z
        '''
        return 10. ** (d / 10.)

    def rfl2mmh(z, a=256., b=1.42):
        '''
        .. based on wradlib.zr.z2r function

        .. z --> r
        '''
        return (z / a) ** (1. / b)

    # decibels to reflectivity
    X_rfl = dbz2rfl((X_scl / 255)*(c2 - c1) + c1)
    # 0 dBz are 0 reflectivity, not 1
    X_rfl[X_rfl == 1] = 0
    # reflectivity to rainfall in mm/h
    X_mmh = rfl2mmh(X_rfl)
    # intensity in mm/h to depth in mm
    X_mm = intensity2depth(X_mmh)

    return X_mm
