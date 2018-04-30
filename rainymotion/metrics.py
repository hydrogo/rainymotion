"""
Efficiency (goodness of fit) metrics for precipitation nowcasting
=================================================================

.. autosummary::
   :nosignatures:
   :toctree: generated/
   
   R
   R2
   RMSE
   MAE
   FAR
   POD
   CSI
   ETS
   HSS
   BSS
   FSC
   MCC

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

####################################################################################
### Regression metrics ###

def R(obs, sim):
        '''
        coefficient of correlation
        '''
        obs = obs.flatten()
        sim = sim.flatten()
        
        return np.corrcoef(obs, sim)[0, 1]
        
    
def R2(obs, sim):
    '''
    coefficient of determination
    '''
    obs = obs.flatten()
    sim = sim.flatten()

    numerator = ((obs - sim) ** 2).sum()

    denominator = ((obs - np.mean(obs)) ** 2).sum()

    return 1 - numerator/denominator
    
    
def RMSE(obs, sim):
    '''
    Root mean square error
    '''
    obs = obs.flatten()
    sim = sim.flatten()
    
    return np.sqrt(np.mean((obs - sim) ** 2))
    
    
def MAE(obs, sim):
    '''
    Mean absolute error
    '''
    obs = obs.flatten()
    sim = sim.flatten()
    
    return np.mean(np.abs(sim - obs))


def gof_metrics(obs, sim):
    '''
    observation - real image (2D numpy array)
    simulation - predicted image (2D numpy array)
    '''
    obs = obs.flatten()
    sim = sim.flatten()
    
    return np.array( ( R(obs, sim), R2(obs, sim), RMSE(obs, sim), MAE(obs, sim) ) )

####################################################################################
### Radar-specific classification metrics ###

def prep_clf(obs, sim, threshold=0.1):
    
    obs = np.where(obs >= threshold, 1, 0)
    sim = np.where(sim  >= threshold, 1, 0)
    
    # True positive (TP)
    hits = np.sum((obs == 1) & (sim == 1))
    
    # False negative (FN)
    misses = np.sum((obs == 1) & (sim == 0))
    
    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (sim == 1))
    
    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (sim == 0))
    
    return hits, misses, falsealarms, correctnegatives

def CSI(obs, sim, threshold=0.1):
    '''
    input:
    
    observation - real image (2D numpy array)
    simulation - predicted image (2D numpy array)
    threshold - the threshold over that we consider rain falls
    
    output:
    
    CSI - critical success index
    
    details in the paper: 
    Woo, W., & Wong, W. (2017). 
    Operational Application of Optical Flow Techniques to Radar-Based Rainfall Nowcasting. 
    Atmosphere, 8(3), 48. https://doi.org/10.3390/atmos8030048
    
    '''
    
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim=sim, threshold=threshold)
    
    return hits / (hits + misses + falsealarms)

def FAR(obs, sim, threshold=0.1):
    '''
    input:
    
    observation - real image (2D numpy array)
    simulation - predicted image (2D numpy array)
    threshold - the threshold over that we consider rain falls
    
    output:

    FAR - false alarm ratio
   
    details in the paper: 
    Woo, W., & Wong, W. (2017). 
    Operational Application of Optical Flow Techniques to Radar-Based Rainfall Nowcasting. 
    Atmosphere, 8(3), 48. https://doi.org/10.3390/atmos8030048
    
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim=sim, threshold=threshold)
    
    return falsealarms / (hits + falsealarms)

def POD(obs, sim, threshold=0.1):
    '''
    input:
    
    observation - real image (2D numpy array)
    simulation - predicted image (2D numpy array)
    threshold - the threshold over that we consider rain falls
    
    output:

    POD - probability of detection
   
    details in the paper: 
    Woo, W., & Wong, W. (2017). 
    Operational Application of Optical Flow Techniques to Radar-Based Rainfall Nowcasting. 
    Atmosphere, 8(3), 48. https://doi.org/10.3390/atmos8030048
    
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim=sim, threshold=threshold)
    
    return hits / (hits + misses)

def HSS(obs, sim, threshold=0.1):
    '''
    input:
    
    observation - real image (2D numpy array)
    simulation - predicted image (2D numpy array)
    threshold - the threshold over that we consider rain falls
    
    output:

    HSS - Heidke skill score
    
    details in the paper: 
    Woo, W., & Wong, W. (2017). 
    Operational Application of Optical Flow Techniques to Radar-Based Rainfall Nowcasting. 
    Atmosphere, 8(3), 48. https://doi.org/10.3390/atmos8030048
    
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim=sim, threshold=threshold)
    
    HSS_num = 2 * (hits * correctnegatives - misses * falsealarms)
    HSS_den = misses**2 + falsealarms**2 + 2*hits*correctnegatives + (misses + falsealarms)*(hits + correctnegatives)
    
    return HSS_num / HSS_den

def ETS(obs, sim, threshold=0.1):
    '''
    input:
    
    observation - real image (2D numpy array)
    simulation - predicted image (2D numpy array)
    threshold - the threshold over that we consider rain falls
    
    output:

    ETS - Equitable Threat Score
    details in the paper:
    Winterrath, T., & Rosenow, W. (2007). A new module for the tracking of radar-derived 
    precipitation with model-derived winds. Advances in Geosciences, 10, 77–83. https://doi.org/10.5194/adgeo-10-77-2007
    
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim=sim, threshold=threshold)
    
    Dr = ((hits + falsealarms) * (hits + misses)) / (hits + misses + falsealarms + correctnegatives)

    ETS = (hits - Dr) / (hits + misses + falsealarms - Dr)
    
    return ETS


def BSS(obs, sim, threshold=0.1):
    '''
    input:
    
    observation - real image (2D numpy array)
    simulation - predicted image (2D numpy array)
    threshold - the threshold over that we consider rain falls
    
    output:
    
    BSS - Brier skill score
    
    details:
    https://en.wikipedia.org/wiki/Brier_score
    '''
    obs = np.where(obs >= threshold, 1, 0)
    sim = np.where(sim  >= threshold, 1, 0)
    
    obs = obs.flatten()
    sim = sim.flatten()
    
    return np.sqrt(np.mean((obs - sim) ** 2))
    
    
def rad_metrics(obs, sim, threshold=0.1):
    '''
    input:
    
    observation - real image (2D numpy array)
    simulation - predicted image (2D numpy array)
    threshold - the threshold over that we consider rain falls
    
    output:
    
    CSI - critical success index
    FAR - false alarm ratio
    POD - probability of detection
    HSS - Heidke skill score
    
    details in the paper: 
    Woo, W., & Wong, W. (2017). 
    Operational Application of Optical Flow Techniques to Radar-Based Rainfall Nowcasting. 
    Atmosphere, 8(3), 48. https://doi.org/10.3390/atmos8030048
    
    '''

    
    return np.array(    (CSI(obs, sim, threshold=threshold), 
                         FAR(obs, sim, threshold=threshold), 
                         POD(obs, sim, threshold=threshold), 
                         HSS(obs, sim, threshold=threshold)) )

####################################################################################
### ML-specific classification metrics ###

def ACC(obs, sim, threshold=0.1):
    '''
    input:
    
    observation - real image (2D numpy array)
    simulation - predicted image (2D numpy array)
    threshold - the threshold over that we consider rain falls
    
    output:
    
    ACC - accuracy score
    
    details:
    https://en.wikipedia.org/wiki/Accuracy_and_precision
    
    '''
    
    TP, FN, FP, TN = prep_clf(obs=obs, sim=sim, threshold=threshold)
    
    return (TP + TN) / (TP + TN + FP + FN)

def precision(obs, sim, threshold=0.1):
    '''
    input:
    
    observation - real image (2D numpy array)
    simulation - predicted image (2D numpy array)
    threshold - the threshold over that we consider rain falls
    
    output:
    
    precision - precision score
    
    details:
    https://en.wikipedia.org/wiki/Information_retrieval#Precision
    
    '''
    
    TP, FN, FP, TN = prep_clf(obs=obs, sim=sim, threshold=threshold)
    
    return TP / (TP + FP)

def recall(obs, sim, threshold=0.1):
    '''
    input:
    
    observation - real image (2D numpy array)
    simulation - predicted image (2D numpy array)
    threshold - the threshold over that we consider rain falls
    
    output:
    
    recall - recall score
    
    details:
    https://en.wikipedia.org/wiki/Information_retrieval#Recall
    
    '''
    
    TP, FN, FP, TN = prep_clf(obs=obs, sim=sim, threshold=threshold)
    
    return TP / (TP + FN)

def FSC(obs, sim, threshold=0.1):
    '''
    input:
    
    observation - real image (2D numpy array)
    simulation - predicted image (2D numpy array)
    threshold - the threshold over that we consider rain falls
    
    output:
    
    FSC - F-score
    
    details:
    https://en.wikipedia.org/wiki/F1_score
    
    '''
    
    pre = precision(obs, sim, threshold=threshold)
    rec = recall(obs, sim, threshold=threshold)
    
    return 2 * ( (pre * rec) / (pre + rec) )

def MCC(obs, sim, threshold=0.1):
    '''
    input:
    
    observation - real image (2D numpy array)
    simulation - predicted image (2D numpy array)
    threshold - the threshold over that we consider rain falls
    
    output:
    
    MCC - Matthews correlation coefficient
    
    details:
    https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    
    '''
    
    TP, FN, FP, TN = prep_clf(obs=obs, sim=sim, threshold=threshold)
    
    MCC_num = TP * TN - FP * FN
    MCC_den = np.sqrt( (TP + FP)*(TP + FN)*(TN + FP)*(TN + FN) )
    
    return MCC_num / MCC_den

###############################################################################
### Curves for plotting ###

def ROC_curve(obs, sim, thresholds):
    '''
    input:
    
    observation - real image (2D numpy array)
    simulation - predicted image (2D numpy array)
    thresholds - number of thresholds over which we consider rain falls
    
    output:
    
    tpr - true positive rate according to selected thresholds (y axis on ROC)
    fpr - false positive rate according to selected thresholds (x axis on ROC)
    
    details:
    https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    
    
    '''
    
    tpr = []
    fpr = []
    
    for threshold in thresholds:
        
        TP, FN, FP, TN = prep_clf(obs=obs, sim=sim, threshold=threshold)
        
        tpr.append(TP / (TP + FN))
        
        fpr.append(FP / (FP + TN))
    
    return np.array(tpr), np.array(fpr)

def PR_curve(obs, sim, thresholds):
    '''
    input:
    
    observation - real image (2D numpy array)
    simulation - predicted image (2D numpy array)
    thresholds - number of thresholds over which we consider rain falls
    
    output:
    
    pre - precision rate according to selected thresholds (y axis on PR)
    rec - recall rate according to selected thresholds (x axis on PR)
    
    details:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    
    
    '''
    
    pre = []
    rec = []
    
    for threshold in thresholds:
        
        pre.append(precision(obs=obs, sim=sim, threshold=threshold))
        rec.append(recall(obs=obs, sim=sim, threshold=threshold))
    
    return np.array(pre), np.array(rec)

def AUC(x, y):
    '''
    input:
    
    x - array of one metric rate (1D)
    y - array of another metric rate (1D)
    
    output:
    
    auc - area under curve wich had been computed by standard trapezial method (np.trapz)
    
    '''
    
    return np.trapz(y, x)