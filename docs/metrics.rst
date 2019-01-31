Metrics
=======

The ``rainymotion`` library provides the extensive list of goodness-of-fit statistical metrics to evaluate nowcasting models performance.

================ =====================================
Metric           Description
================ =====================================
**Regression**
R                Correlation coefficient
R2               Coefficient of determination
RMSE             Root mean squared error
MAE              Mean absolute error
**QPN specific**
CSI              Critical Success Index
FAR              False Alarm Rate
POD              Probability Of Detection
HSS              Heidke Skill Score
ETS              Equitable Threat Score
BSS              Brier Skill Score
**ML specific**
ACC              Accuracy
precision        Precision
recall           Recall
FSC              F1-score
MCC              Matthews Correlation Coefficient
================ =====================================

You can easily use any metric for verification of your nowcasts:

.. code-block:: python

    # import the specific metric from the rainymotion library
    from rainymotion.metrics import CSI

    # read your observations and simulations
    obs = np.load("/path/to/observations")
    sim = np.load("/path/to/simulations")

    # calculate the corresponding metric
    csi = CSI(obs, sim, threshold=1.0)


.. seealso::
    :doc:`notebooks`.
