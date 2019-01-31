Utils
=====

The ``rainymotion`` library provides some useful utils to help user with a data preprocessing workflow which is usually needed to perform radar-based precipitation nowcasting. At the moment, we have utils that only deal with the RY radar product by DWD. By the way, you can use such utils as an example to construct your own data preprocessing pipeline.

================ =====================================
Function           Description
================ =====================================
depth2intensity  Convert rainfall depth (in mm) to rainfall intensity (mm/h)
intensity2depth  Convert rainfall intensity (mm/h) back to rainfall depth (mm)
RYScaler         Scale RY data from mm (in float64) to brightness (in uint8)
inv_RYScaler     Scale brightness (in uint8) back to RY data (in mm).
================ =====================================

.. seealso::
    :doc:`notebooks` for how to use ``rainymotion.utils`` in the nowcasting workflow.
