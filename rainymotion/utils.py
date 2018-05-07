"""
``rainymotion.utils``: scaling data for optical flow based nowcasting models
============================================================================

.. autosummary::
   :nosignatures:
   :toctree: generated/

    Scaler
    invScaler
    
  
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def Scaler(X_mmh):
    '''
    Transfer X from mm/h (raw data)
    to dBz values (the most suitable for optical tracking),
    and then convert them to [0, 255] interval
    for image tracking capability
    
    Args:
        X (numpy.ndarray): radar image of the rainfall intensity

    Returns:
        numpy.ndarray(uint8): brightness integer values from 0 to 255 for corresponding input rainfall intensity
        float: minimum value of rainfall intensities in dBz
        float: maximum value of rainfall intensities in dBz
    
    .. X_mmh - mm/h
       X_rfl - reflectivity
       X_dbz - decibels
       X_scl - decibels scaled to [0, 255]
    
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
    
    # mmh to reflectivity
    X_rfl = mmh2rfl(X_mmh)
    # remove zero reflectivity
    # then log10(0.1) = -1 not inf (numpy warning arised)
    X_rfl[X_rfl == 0] = 0.1
    # reflectivity to dBz
    X_dbz = rfl2dbz(X_rfl)
    #X_dbz = rfl2dbz(mmh2rfl(X_mmh)) # first version with warning arised
    # remove all -inf
    X_dbz[X_dbz < 0] = 0

    # MinMaxScaling
    dbz_min = X_dbz.min()
    dbz_max = X_dbz.max()

    #X_scl = ( (X_dbz - dbz_min) / (dbz_max - dbz_min) * 255 ).astype(np.uint8)

    return ( (X_dbz - dbz_min) / (dbz_max - dbz_min) * 255 ).astype(np.uint8), dbz_min, dbz_max

def inv_Scaler(X_scl, dbz_min, dbz_max):
    '''
    Transfer brightness integer value in uint8 [0, 255] to rainfall intensities in mm/h.
    Function which is inverse to Scaler() function. 

    Args:
        X_scl (numpy.ndarray): array of brightness integers obtained from Scaler() function.
        dbz_min: minimum value of rainfall intensities in dBz obtained from Scaler() function.
        dbz_max: maximum value of rainfall intensities in dBz obtained from Scaler() function.
    
    Returns:
        numpy.ndarray(float): rainfall intensities in mm/h

    .. X_mmh - mm/h
       X_rfl - reflectivity
       X_dbz - decibels
       X_scl - decibels scaled to [0, 255]

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
    
    # [0, 255] to decibels
    #X_dbz = (X_scl / 255)*(dbz_max - dbz_min) + dbz_min
    # decibels to reflectivity
    X_rfl = dbz2rfl((X_scl / 255)*(dbz_max - dbz_min) + dbz_min)
    # the truth is that 0 dBz are 0 reflectivity, not 1
    X_rfl[X_rfl == 1] = 0
    # reflectivity to rainfall in mm/h
    X_mmh = rfl2mmh(X_rfl)
    
    return X_mmh