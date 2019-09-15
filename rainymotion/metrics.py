"""
The rainymotion library provides different goodness of fit metrics for
nowcasting models' performance evaluation.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# -- Regression metrics -- #


def R(obs, sim):
    """
    Correlation coefficient

    Reference:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations

    Returns:
        float: correlation coefficient between observed and simulated values

    """
    obs = obs.flatten()
    sim = sim.flatten()

    return np.corrcoef(obs, sim)[0, 1]


def R2(obs, sim):
    """
    Coefficient of determination

    Reference:
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations

    Returns:
        float: coefficient of determination between observed and
               simulated values

    """

    obs = obs.flatten()
    sim = sim.flatten()

    numerator = ((obs - sim) ** 2).sum()

    denominator = ((obs - np.mean(obs)) ** 2).sum()

    return 1 - numerator/denominator


def RMSE(obs, sim):
    """
    Root mean squared error

    Reference: https://en.wikipedia.org/wiki/Root-mean-square_deviation

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations

    Returns:
        float: root mean squared error between observed and simulated values

    """
    obs = obs.flatten()
    sim = sim.flatten()

    return np.sqrt(np.mean((obs - sim) ** 2))


def MAE(obs, sim):
    """
    Mean absolute error

    Reference: https://en.wikipedia.org/wiki/Mean_absolute_error

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations

    Returns:
        float: mean absolute error between observed and simulated values

    """
    obs = obs.flatten()
    sim = sim.flatten()

    return np.mean(np.abs(sim - obs))

# -- Radar-specific classification metrics -- #


def prep_clf(obs, sim, threshold=0.1):

    obs = np.where(obs >= threshold, 1, 0)
    sim = np.where(sim >= threshold, 1, 0)

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
    """
    CSI - critical success index

    details in the paper:
    Woo, W., & Wong, W. (2017).
    Operational Application of Optical Flow Techniques to Radar-Based
    Rainfall Nowcasting.
    Atmosphere, 8(3), 48. https://doi.org/10.3390/atmos8030048

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: CSI value

    """

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim=sim,
                                                           threshold=threshold)

    return hits / (hits + misses + falsealarms)


def FAR(obs, sim, threshold=0.1):
    '''
    FAR - false alarm rate

    details in the paper:
    Woo, W., & Wong, W. (2017).
    Operational Application of Optical Flow Techniques to Radar-Based
    Rainfall Nowcasting.
    Atmosphere, 8(3), 48. https://doi.org/10.3390/atmos8030048

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: FAR value

    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim=sim,
                                                           threshold=threshold)

    return falsealarms / (hits + falsealarms)


def POD(obs, sim, threshold=0.1):
    '''
    POD - probability of detection

    details in the paper:
    Woo, W., & Wong, W. (2017).
    Operational Application of Optical Flow Techniques to Radar-Based
    Rainfall Nowcasting.
    Atmosphere, 8(3), 48. https://doi.org/10.3390/atmos8030048

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: POD value

    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim=sim,
                                                           threshold=threshold)

    return hits / (hits + misses)


def HSS(obs, sim, threshold=0.1):
    '''
    HSS - Heidke skill score

    details in the paper:
    Woo, W., & Wong, W. (2017).
    Operational Application of Optical Flow Techniques to Radar-Based
    Rainfall Nowcasting.
    Atmosphere, 8(3), 48. https://doi.org/10.3390/atmos8030048

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: HSS value

    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim=sim,
                                                           threshold=threshold)

    HSS_num = 2 * (hits * correctnegatives - misses * falsealarms)
    HSS_den = (misses**2 + falsealarms**2 + 2*hits*correctnegatives +
               (misses + falsealarms)*(hits + correctnegatives))

    return HSS_num / HSS_den


def ETS(obs, sim, threshold=0.1):
    '''
    ETS - Equitable Threat Score
    details in the paper:
    Winterrath, T., & Rosenow, W. (2007). A new module for the tracking of
    radar-derived precipitation with model-derived winds.
    Advances in Geosciences,10, 77–83. https://doi.org/10.5194/adgeo-10-77-2007

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: ETS value

    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim=sim,
                                                           threshold=threshold)
    num = (hits + falsealarms) * (hits + misses)
    den = hits + misses + falsealarms + correctnegatives
    Dr = num / den

    ETS = (hits - Dr) / (hits + misses + falsealarms - Dr)

    return ETS


def BSS(obs, sim, threshold=0.1):
    '''
    BSS - Brier skill score

    details:
    https://en.wikipedia.org/wiki/Brier_score

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: BSS value

    '''
    obs = np.where(obs >= threshold, 1, 0)
    sim = np.where(sim >= threshold, 1, 0)

    obs = obs.flatten()
    sim = sim.flatten()

    return np.sqrt(np.mean((obs - sim) ** 2))

# -- ML-specific classification metrics -- #


def ACC(obs, sim, threshold=0.1):
    '''
    ACC - accuracy score

    details:
    https://en.wikipedia.org/wiki/Accuracy_and_precision

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: accuracy value

    '''

    TP, FN, FP, TN = prep_clf(obs=obs, sim=sim, threshold=threshold)

    return (TP + TN) / (TP + TN + FP + FN)


def precision(obs, sim, threshold=0.1):
    '''
    precision - precision score

    details:
    https://en.wikipedia.org/wiki/Information_retrieval#Precision

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: precision value

    '''

    TP, FN, FP, TN = prep_clf(obs=obs, sim=sim, threshold=threshold)

    return TP / (TP + FP)


def recall(obs, sim, threshold=0.1):
    '''
    recall - recall score

    details:
    https://en.wikipedia.org/wiki/Information_retrieval#Recall

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: recall value
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, sim=sim, threshold=threshold)

    return TP / (TP + FN)


def FSC(obs, sim, threshold=0.1):
    '''
    FSC - F-score

    details:
    https://en.wikipedia.org/wiki/F1_score

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: FSC value
    '''

    pre = precision(obs, sim, threshold=threshold)
    rec = recall(obs, sim, threshold=threshold)

    return 2 * ((pre * rec) / (pre + rec))


def MCC(obs, sim, threshold=0.1):
    '''
    MCC - Matthews correlation coefficient

    details:
    https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: MCC value
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, sim=sim, threshold=threshold)

    MCC_num = TP * TN - FP * FN
    MCC_den = np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))

    return MCC_num / MCC_den

# -- Curves for plotting -- #


def ROC_curve(obs, sim, thresholds):
    '''
    ROC - Receiver operating characteristic curve coordinates

    Reference: https://en.wikipedia.org/wiki/Receiver_operating_characteristic

    Args:

        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        thresholds (list with floats): number of thresholds over which
                                       we consider rain falls

    Returns:

        tpr (numpy.ndarray): true positive rate according to selected
                             thresholds (y axis on ROC)
        fpr (numpy.ndarray): false positive rate according to selected
                             thresholds (x axis on ROC)

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
    PRC - precision-recall curve coordinates

    Reference:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

    Args:

        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        thresholds (list with floats): number of thresholds over which
                                       we consider rain falls

    Returns:

        pre (numpy.ndarray): precision rate according to selected thresholds
                             (y axis on PR)
        rec (numpy.ndarray): recall rate according to selected thresholds
                             (x axis on PR)

    '''

    pre = []
    rec = []

    for threshold in thresholds:

        pre.append(precision(obs=obs, sim=sim, threshold=threshold))
        rec.append(recall(obs=obs, sim=sim, threshold=threshold))

    return np.array(pre), np.array(rec)


def AUC(x, y):
    '''
    AUC - area under curve

    Note: area under curve wich had been computed by standard trapezial
          method (np.trapz)

    Args:

        x (numpy.ndarray): array of one metric rate (1D)
        y (numpy.ndarray): array of another metric rate (1D)

    Returns:

        float - area under curve

    '''

    return np.trapz(y, x)
