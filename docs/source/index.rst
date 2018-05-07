.. rainymotion documentation master file, created by
   sphinx-quickstart on Mon Apr 30 10:18:58 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


``rainymotion``: Python library for radar-based precipitation nowcasting
========================================================================

:Release: |release|
:Date: |today|

``rainymotion`` is a Python library utilizes different models for radar-based precipitation nowcasting based on the optical flow techniques.

.. note:: Please cite `rainymotion` as *Ayzel, G., Heistermann M., and Winterrath T.: Optical flow models as a benchmark for radar-based precipitation nowcasting* (in preparation)

``rainymotion`` also provides a bunch of statistical metrics for nowcasting models evaluation (module ``rainymotion.metrics``) and useful utils (module ``rainymotion.utils``) for radar data preprocessing.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   sparse
   sparsesd
   dense
   denserotation
   models
   metrics
   utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
