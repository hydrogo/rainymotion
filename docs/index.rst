.. rainymotion documentation master file, created by
   sphinx-quickstart on Mon Jan 21 15:37:40 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

``rainymotion``: Python library for radar-based precipitation nowcasting
========================================================================

:Release: |release|
:Date: |today|

``rainymotion`` is an open Python library utilizes different models for radar-based precipitation nowcasting based on the optical flow techniques.

You can find the ``rainymotion`` source code in the corresponding `Github repository <https://github.com/hydrogo/rainymotion>`_.

.. note:: Please cite ``rainymotion`` as  *Ayzel, G., Heistermann, M., and Winterrath, T.: Optical flow models as an open benchmark for radar-based precipitation nowcasting (rainymotion v0.1), Geosci. Model Dev., 12, 1387-1402, https://doi.org/10.5194/gmd-12-1387-2019, 2019.* 

``rainymotion`` also provides a bunch of statistical metrics for nowcasting models evaluation (module ``rainymotion.metrics``) and useful utils (module ``rainymotion.utils``) for radar data preprocessing.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   gettingstarted
   notebooks
   reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
