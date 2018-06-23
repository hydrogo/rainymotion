"""
``rainymotion.models``: optical flow models for radar-based precipitation nowcasting
====================================================================================

Explicit model description in plain test (like in the paper)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   Sparse
   SparseSD
   Dense
   DenseRotation
   EulerianPersistence


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from scipy.interpolate import griddata
import skimage.transform as sktf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

### SPARSE GROUP ###
class Sparse:
    """
    The basic class for the Sparse model implementation (the Sparse Group).

    **It is highly recommended to try the model with default parameters for first**


    Parameters
    ----------
    params : dictionary 
        This dictionary holds parameters for identification and tracking relevant rain field features.
 
        `params` dictionary has the folded structure: 
            * 'st_pars' key stores dictionary with the Shi-Tomasi's corner detection algorithm parameters 
            * 'lk_pars' key stores dictionary with the Lukas-Kanade's tracking algorithm parameters

        Default `params` dict is:
        
        ``params = {'st_pars' : dict(maxCorners = 200, qualityLevel = 0.2, minDistance = 7, blockSize = 21 ), 'lk_pars' : dict(winSize  = (20,20), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0))}``

        where:
            * for ['st_pars']:
                + maxCorners       (default: 200; ranges: [50, 300];  type: int)  - maximum number of corners to return
                + qualityLevel     (default: 0.2; ranges: [0.1, 0.7], type: float)- minimal accepted quality of image corners
                + minDistance      (default: 7;   ranges: [3, 15],    type: int)  - minimum possible Euclidean distance between the returned corners
                + blockSize        (default: 21;  ranges: [10, 50],   type: int)  - size of an average block for computing a derivative covariation matrix over each pixel neighborhood
            * for ['lk_pars']:
                + winSize          (default: 20;  ranges: [10, 30],   type: int)  - size of the search window at each pyramid level
                + maxLevel         (default: 2;   ranges: [0, 4],     type: int)  - 0-based maximal pyramid level number

        You can find the extensive parameters description by the following links:
            * http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=goodfeaturestotrack#goodfeaturestotrack
            * http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html
    extrapolator : dictionary 
        This dictionary holds parameters used for precipitation field extrapolation.

        `extrapolator` dictionary has the following structure:
            * 'model' key stores the regression model instance from the scikit-learn library
            * 'features' key stores the key for regression model features representation.

        Default `extrapolator` dictionyry is:
        
        ``extrapolator = {"model": LinearRegression(), "features": "ordinal"}``

        where:  
            * model (default: LinearRegression()) - any regression model instance from sklearn library 
            * features (default: "ordinal"; available: "polynomial") - key for features representation in the regression model
    
    warper: string
        The parameter determines the variant of warping transformation.

        Default value: "affine"

        Available values: "euclidean", "similarity", "projective"

        For more information, please, follow the link: http://scikit-image.org/docs/dev/api/skimage.transform.html

    input_data: numpy.ndarray
        8-bit (uint8, 0-255) 3D numpy.ndarray (frames, dim_x, dim_y) of radar data for previous 2 hours (24 frames)

        Default value: None

    lead_steps: int
        The required lead steps of nowcasting

        Default value: 12
    
    Examples
    --------
    
    See :ref:`/notebooks/sparse.ipynb`

    See :ref:`/notebooks/nowcasting.ipynb`.

    """
    def __init__(self):
                
        self.params = {'st_pars' : dict(maxCorners = 200, 
                           qualityLevel = 0.2, 
                           minDistance = 7, 
                           blockSize = 21 ),
                       'lk_pars' : dict(winSize  = (20,20), 
                           maxLevel = 2, 
                           criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0))
                      }
        
        self.extrapolator = {"model": LinearRegression(), "features": "ordinal"}
        
        self.warper = "affine"      
        
        self.input_data = None
        
        self.lead_steps = 12
        
    def run(self):
        
        """
        The method for running the Sparse model with the parameters were specified in the Sparse class parameters.

        For running with the default parameters only the input data instance is required (.input_data parameter).

        Args:
            None
        
        Returns:
            numpy.ndarray: 3D numpy.ndarray of precipitation nowcasting (frames, dim_x, dim_y)
        """
        # check input data
        
        if not isinstance(self.input_data, np.ndarray) \
        or not self.input_data.dtype == "uint8" \
        or not len(self.input_data.shape) == 3:
            raise TypeError("input data must be np.ndarray(shape=(frames, nrows, ncols), dtype='uint8')")
            
        transformations = {'euclidean': sktf.EuclideanTransform(), 
                           'similarity': sktf.SimilarityTransform(), 
                           'affine': sktf.AffineTransform(),
                           'projective': sktf.ProjectiveTransform(), 
                          }
               
        # find features to track
        old_corners = cv2.goodFeaturesToTrack(self.input_data[0], mask = None, **self.params['st_pars'])

        # Set containers to collect results (time steps in rows, detected corners in columns)
        #   corner x coords
        x = np.full((self.input_data.shape[0], len(old_corners)), np.nan)
        #   corner y coords
        y = np.full((self.input_data.shape[0], len(old_corners)), np.nan)
        #   Assign persistent corner IDs
        ids = np.arange(len(old_corners))

        # fill in first values
        x[0, :] = old_corners[:, 0, 0]
        y[0, :] = old_corners[:, 0, 1]
        
        # track corners by optical flow algorithm
        for i in range(1, self.input_data.shape[0]):             

            new_corners, st, err = cv2.calcOpticalFlowPyrLK(prevImg=self.input_data[i-1], 
                                                            nextImg=self.input_data[i], 
                                                            prevPts=old_corners, 
                                                            nextPts=None, 
                                                            **self.params['lk_pars'])

            # select only good attempts for corner tracking
            success = st.ravel() == 1
            # use only sucessfull ids for filling
            ids = ids[success]
            # fill in results
            x[i, ids] = new_corners[success,0,0]
            y[i, ids] = new_corners[success,0,1]
            # new corners will be old in the next loop
            old_corners = new_corners[success]
        
        # consider only full paths
        full_paths_without_nan = [ np.sum(np.isnan(x[:, i])) == 0 for i in range(x.shape[1]) ]
        x = x[:, full_paths_without_nan].copy()
        y = y[:, full_paths_without_nan].copy()

        # containers for corners predictions
        x_new = np.full((self.lead_steps, x.shape[1]), np.nan)
        y_new = np.full((self.lead_steps, y.shape[1]), np.nan)
        
        for i in range(x.shape[1]):
        
            x_train = x[:, i]
            y_train = y[:, i]

            X = np.arange(x.shape[0] + self.lead_steps)
            
            if self.extrapolator["features"] == "polynomial": 
                polyfeatures = PolynomialFeatures(2)
                X = polyfeatures.fit_transform(X.reshape(-1, 1))
                X_train = X[:x.shape[0], :]
                X_pred  = X[x.shape[0]:, :]
            else:
                X = X.reshape(-1, 1)
                X_train = X[:x.shape[0], :]
                X_pred  = X[x.shape[0]:, :]

            x_pred = self.extrapolator["model"].fit(X_train, x_train).predict(X_pred)
            y_pred = self.extrapolator["model"].fit(X_train, y_train).predict(X_pred)
            
            x_new[:, i] = x_pred
            y_new[:, i] = y_pred
        
        # define last frame - the general source of our transforming
        last_frame = self.input_data[-1]

        # define source corners in appropriate format
        pts_source = np.hstack([x[-1, :].reshape(-1, 1), y[-1, :].reshape(-1, 1)])
        
        # define container for targets in appropriate format
        pts_target_container = [np.hstack([x_new[i, :].reshape(-1, 1), 
                                           y_new[i, :].reshape(-1, 1)]) for i in range(x_new.shape[0])]
        
        # set up transformer object
        trf = transformations[self.warper]
        
        # now we can start to find nowcasted image
        # for every candidate of projected sets of points

        # container for our nowcasts
        nowcst_frames = []

        for lead_step, pts_target in enumerate(pts_target_container):

            # estimate transformation matrix
            # based on source and traget points
            trf.estimate(pts_source, pts_target)

            # make a nowcast
            nowcst_frame = sktf.warp(last_frame/255, trf.inverse)
            # transformations dealing with strange behaviour
            nowcst_frame = (nowcst_frame*255).astype('uint8')
            # add to the container
            nowcst_frames.append(nowcst_frame)


        forecast = np.dstack(nowcst_frames)

        return np.moveaxis(forecast, -1, 0).copy()

##### SIMPLIFIED SPARSE OPTICAL FLOW #####
class SparseSD:
    """
    The basic class for the SparseSD model implementation (the Sparse Group).

    **It is highly recommended to try the model with default parameters for first**

    Parameters
    ----------
    params : dictionary 
        This dictionary holds parameters for identification and tracking relevant rain field features.
 
        `params` dictionary has the folded structure: 
            * 'st_pars' key stores dictionary with the Shi-Tomasi's corner detection algorithm parameters 
            * 'lk_pars' key stores dictionary with the Lukas-Kanade's tracking algorithm parameters

        Default `params` dict is:
        
        ``params = {'st_pars' : dict(maxCorners = 200, qualityLevel = 0.2, minDistance = 7, blockSize = 21 ), 'lk_pars' : dict(winSize  = (20,20), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0))}``

        where:
            * for ['st_pars']:
                + maxCorners       (default: 200; ranges: [50, 300];  type: int)  - maximum number of corners to return
                + qualityLevel     (default: 0.2; ranges: [0.1, 0.7], type: float)- minimal accepted quality of image corners
                + minDistance      (default: 7;   ranges: [3, 15],    type: int)  - minimum possible Euclidean distance between the returned corners
                + blockSize        (default: 21;  ranges: [10, 50],   type: int)  - size of an average block for computing a derivative covariation matrix over each pixel neighborhood
            * for ['lk_pars']:
                + winSize          (default: 20;  ranges: [10, 30],   type: int)  - size of the search window at each pyramid level
                + maxLevel         (default: 2;   ranges: [0, 4],     type: int)  - 0-based maximal pyramid level number

        You can find the extensive parameters description by the following links:
            * http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=goodfeaturestotrack#goodfeaturestotrack
            * http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html
        
    warper: string
        The parameter determines the variant of warping transformation.

        Default value: "affine"

        Available values: "euclidean", "similarity", "projective"

        For more information, please, follow the link: http://scikit-image.org/docs/dev/api/skimage.transform.html

    input_data: numpy.ndarray
        8-bit (uint8, 0-255) 3D numpy.ndarray (frames, dim_x, dim_y) of radar data for previous 2 hours (24 frames)

        Default value: None

    lead_steps: int
        The required lead steps of nowcasting

        Default value: 12
    
    Examples
    --------
    
    See :ref:`/notebooks/sparse.ipynb`

    See :ref:`/notebooks/nowcasting.ipynb`.    

    """

    def __init__(self):
        
              
        self.params = {'st_pars' : dict(maxCorners = 200, 
                           qualityLevel = 0.2, 
                           minDistance = 7, 
                           blockSize = 21 ),
                       'lk_pars' : dict(winSize  = (20,20), 
                           maxLevel = 2, 
                           criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0))
                      }
              
        self.warper = "affine"      
        
        self.input_data = None
        
        self.lead_steps = 12
        
    def run(self):
        
        """
        The method for running the SparseSD model with the parameters were specified in the SparseSD class parameters.

        For running with the default parameters only the input data instance is required (.input_data parameter).

        Args:
            None
        
        Returns:
            numpy.ndarray: 3D numpy.ndarray of precipitation nowcasting (frames, dim_x, dim_y)
        
        """
        # check input data
        
        if not isinstance(self.input_data, np.ndarray) \
        or not self.input_data.dtype == "uint8" \
        or not len(self.input_data.shape) == 3:
            raise TypeError("input data must be np.ndarray(shape=(frames, nrows, ncols), dtype='uint8')")
            
        transformations = {'euclidean': sktf.EuclideanTransform(), 
                           'similarity': sktf.SimilarityTransform(), 
                           'affine': sktf.AffineTransform(),
                           'projective': sktf.ProjectiveTransform(), 
                          }
        
        # define penult and last frames
        penult_frame = self.input_data[-2]
        last_frame = self.input_data[-1]
        
        # find features to track
        old_corners = cv2.goodFeaturesToTrack(self.input_data[0], mask = None, **self.params['st_pars'])
        
        # track corners by optical flow algorithm
        new_corners, st, err = cv2.calcOpticalFlowPyrLK(prevImg=penult_frame, 
                                                        nextImg=last_frame, 
                                                        prevPts=old_corners, 
                                                        nextPts=None, 
                                                        **self.params['lk_pars'])

        # select only good attempts for corner tracking
        success = st.ravel() == 1
        new_corners = new_corners[success].copy()
        old_corners = old_corners[success].copy()
        
        # calculate Simple Delta
        delta = new_corners.reshape(-1, 2) - old_corners.reshape(-1, 2)
        
        # simplificate furher transformations
        pts_source = new_corners.reshape(-1, 2)
        
        # propagate our corners through time
        pts_target_container = []
        
        for lead_step in range(self.lead_steps): 
            pts_target_container.append( pts_source + delta * (lead_step + 1) )
        
        # set up transformer object
        trf = transformations[self.warper]
        
        # now we can start to find nowcasted image
        # for every candidate of projected sets of points

        # container for our nowcasts
        nowcst_frames = []

        for lead_step, pts_target in enumerate(pts_target_container):

            # estimate transformation matrix
            # based on source and traget points
            trf.estimate(pts_source, pts_target)

            # make a nowcast
            nowcst_frame = sktf.warp(last_frame/255, trf.inverse)
            # transformations dealing with strange behaviour
            nowcst_frame = (nowcst_frame*255).astype('uint8')
            # add to the container
            nowcst_frames.append(nowcst_frame)

        forecast = np.dstack(nowcst_frames)

        return np.moveaxis(forecast, -1, 0).copy()


### DENSE GROUP ###
class Dense:
    """
    The basic class for the Dense model implementation (the Dense Group).
    
    **It is highly recommended to try the model with default parameters for first**
    
    Parameters
    ----------

    params: dictionary 
        This dictionary holds parameters for the Farnerbacks's optical flow algorithm.
        
        `params` dictionary has the following structure:
            * 'farneback_param' key stores the dictionary with the Farnerback's optical flow parameters.

        Default `params` dict is:
        
        ``params = {'farneback_param': dict(pyr_scale = 0.5, levels = 3, winsize = 15, iterations = 3, poly_n = 5, poly_sigma = 1.1, flags = 0)}``
        
        where:
            * for ['farneback_param']:
                + pyr_scale  (default: 0.5; ranges: [0.1, 0.9]; type: float) - image scale to build pyramids
                + levels     (default: 3;   ranges: [1, 7],     type: int)   - number of pyramid layers
                + winsize    (default: 15;  ranges: [5, 30],    type: int)   - averaging window size
                + iterations (default: 3;   ranges: [2, 10],    type: int)   - number of iterations at each pyramid level
                + poly_n     (default: 5;   ranges: [3, 10],    type: int)   - size of the pixel neighborhood
                + poly_sigma (default: 1.1; ranges: [0.9, 2],   type: float) - std of the Gaussian for smoothing derivatives

        You can find the extensive parameters description by the following link:
            * http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback

    input_data: numpy.ndarray
        8-bit (uint8, 0-255) 3D numpy.ndarray (frames, dim_x, dim_y) of radar data for previous 10 minutes (2 frames)

        Default value: None

    lead_steps: int
        The required lead steps of nowcasting

        Default value: 12
    
    interpolator: string
        Interpolation technique.
        
        Default: "linear"

        Available: "nearest", "cubic"

        For details see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

    Examples
    --------
    
    See :ref:`/notebooks/dense.ipynb`
    
    See :ref:`/notebooks/nowcasting.ipynb`.

    """

    def __init__(self):
        
               
        self.params = {'farneback_param': dict(pyr_scale = 0.5, 
                                               levels = 3, 
                                               winsize = 15, 
                                               iterations = 3, 
                                               poly_n = 5, 
                                               poly_sigma = 1.1, 
                                               flags = 0)
                      }
        
        self.input_data = None
        
        self.lead_steps = 12
        
        self.interpolator = "linear"
    
    def run(self):
        
        """
        The method for running the model with the parameters were specified in the Dense class parameters.

        For running with the default parameters only the input data instance is required (.input_data parameter).

        Args:
            None
        
        Returns:
            numpy.ndarray: 3D numpy.ndarray of precipitation nowcasting (frames, dim_x, dim_y)

        """
        # check input data
        if not isinstance(self.input_data, np.ndarray) \
        or not self.input_data.dtype == "uint8" \
        or not len(self.input_data.shape) == 3:
            raise TypeError("input data must be np.ndarray(shape=(frames, nrows, ncols), dtype='uint8')")
        
        # define penult and last frames
        penult_frame = self.input_data[-2]
        last_frame = self.input_data[-1]
        
        # compute dense flow
        delta = cv2.calcOpticalFlowFarneback(penult_frame, last_frame, None, **self.params['farneback_param'])
        
        # no need in rounding deltas -- subpixel accuracy, bitch
        delta_x = delta[::, ::, 0]
        delta_y = delta[::, ::, 1]
        
        # make a source meshgrid
        coord_source_i, coord_source_j = np.meshgrid(range(last_frame.shape[0]), range(last_frame.shape[1]))
        
        # propagate our image through time based on dense flow
        # container for our nowcasts
        nowcst_frames = []
        
        for lead_step in range(self.lead_steps):       
            
            # calculate new coordinates of radar pixels
            coord_target_i = coord_source_i + delta_x * (lead_step + 1)
            coord_target_j = coord_source_j + delta_y * (lead_step + 1)
            
            # we suppose that old values just go to the new locations (Lagrangian persistense)
            # but we need to regrid data
            nowcst_frame = griddata((coord_target_i.flatten(), coord_target_j.flatten()), 
                                    last_frame.flatten(), 
                                    (coord_source_i.flatten(), coord_source_j.flatten()), 
                                    method=self.interpolator, fill_value=0)
            
            # reshape output
            nowcst_frame = nowcst_frame.reshape(last_frame.shape)
            
            # converting to uint8
            nowcst_frame = nowcst_frame.astype('uint8')
            
            # add to the container
            nowcst_frames.append(nowcst_frame)
        
        #return delta 
        return np.moveaxis(np.dstack(nowcst_frames), -1, 0).copy()
    
class DenseRotation:
    """
    The basic class for the DenseRotation model implementation (the Dense Group).
    
    **It is highly recommended to try the model with default parameters for first**
    
    Parameters
    ----------

    params: dictionary 
        This dictionary holds parameters for the Farnerbacks's optical flow algorithm.
        
        `params` dictionary has the following structure:
            * 'farneback_param' key stores the dictionary with the Farnerback's optical flow parameters.

        Default `params` dict is:
        
        ``params = {'farneback_param': dict(pyr_scale = 0.5, levels = 3, winsize = 15, iterations = 3, poly_n = 5, poly_sigma = 1.1, flags = 0)}``
        
        where:
            * for ['farneback_param']:
                + pyr_scale  (default: 0.5; ranges: [0.1, 0.9]; type: float) - image scale to build pyramids
                + levels     (default: 3;   ranges: [1, 7],     type: int)   - number of pyramid layers
                + winsize    (default: 15;  ranges: [5, 30],    type: int)   - averaging window size
                + iterations (default: 3;   ranges: [2, 10],    type: int)   - number of iterations at each pyramid level
                + poly_n     (default: 5;   ranges: [3, 10],    type: int)   - size of the pixel neighborhood
                + poly_sigma (default: 1.1; ranges: [0.9, 2],   type: float) - std of the Gaussian for smoothing derivatives

        You can find the extensive parameters description by the following link:
            * http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback

    input_data: numpy.ndarray
        8-bit (uint8, 0-255) 3D numpy.ndarray (frames, dim_x, dim_y) of radar data for previous 10 minutes (2 frames)

        Default value: None

    lead_steps: int
        The required lead steps of nowcasting

        Default value: 12
    
    interpolator: string
        Interpolation technique.
        
        Default: "linear"

        Available: "nearest", "cubic"

        For details see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

    Examples
    --------
    
    See :ref:`/notebooks/dense.ipynb`
    
    See :ref:`/notebooks/nowcasting.ipynb`.
    """
    def __init__(self):
               
        self.params = {'farneback_param': dict(pyr_scale = 0.5, 
                                               levels = 3, 
                                               winsize = 15, 
                                               iterations = 3, 
                                               poly_n = 5, 
                                               poly_sigma = 1.1, 
                                               flags = 0)
                      }
        
        self.input_data = None
        
        self.lead_steps = 12
        
        self.interpolator = "linear"
    
    def run(self):
        
        """
        The method for running the model with the parameters were specified in the DenseRotation class parameters.

        For running with the default parameters only the input data instance is required (.input_data parameter).

        Args:
            None
        
        Returns:
            numpy.ndarray: 3D numpy.ndarray of precipitation nowcasting (frames, dim_x, dim_y)

        """
        # check input data
        if not isinstance(self.input_data, np.ndarray) \
        or not self.input_data.dtype == "uint8" \
        or not len(self.input_data.shape) == 3:
            raise TypeError("input data must be np.ndarray(shape=(frames, nrows, ncols), dtype='uint8')")
        
        # define penult and last frames
        penult_frame = self.input_data[-2]
        last_frame = self.input_data[-1]
        
        # compute dense flow
        delta = cv2.calcOpticalFlowFarneback(penult_frame, last_frame, None, **self.params['farneback_param'])
        
        # no need in rounding deltas -- subpixel accuracy, bitch
        delta_x = delta[::, ::, 0]
        delta_y = delta[::, ::, 1]
        
        #print(delta_x.shape, delta_y.shape)
        
        # rows, cols
        rows = last_frame.shape[0]
        cols = last_frame.shape[1]
        
        # make a source meshgrid
        coord_source_i, coord_source_j = np.meshgrid(range(rows), range(cols))
        coord_source = np.vstack([coord_source_i.ravel(), coord_source_j.ravel()]).T
        
        # make a source meshgrids for deltas
        #delta_x_source = coord_source.copy()
        #delta_y_source = coord_source.copy()
        
        # Create a KDTree for deltas
        kdtree = spatial.cKDTree(coord_source, leafsize=8)
        
        #print(tree_x.data)
               
        # Block for calculation displacement
        # init placeholders
        coord_targets = []
        
        for lead_step in range(self.lead_steps):       
            
            # find indexes to match source coordinates with displacements
            nearest_indexes = kdtree.query(coord_source)[1]
                       
            # based on corresponding indexes find the displacements values
            corresponding_delta_x = delta_x.ravel()[nearest_indexes].reshape(rows, cols)
            corresponding_delta_y = delta_y.ravel()[nearest_indexes].reshape(rows, cols)
            
            # calculate corresponding targets
            coord_target_i = coord_source_i + corresponding_delta_x
            coord_target_j = coord_source_j + corresponding_delta_y
            
            coord_targets.append([coord_target_i, coord_target_j])
            
            # now update source coordinates
            coord_source_i = coord_target_i
            coord_source_j = coord_target_j
            coord_source = np.vstack([coord_source_i.ravel(), coord_source_j.ravel()]).T
            
        # Block for calculation nowcasts
        # Need to create coordinate sources from scratch
        coord_source_i, coord_source_j = np.meshgrid(range(rows), range(cols))
        
        # container for our nowcasts
        nowcst_frames = []
        
        for lead_step in range(self.lead_steps):
            
            # unpack our target coordinates
            coord_target_i, coord_target_j = coord_targets[lead_step]
            
            # and with this implementation WE CONSIDER ROTATION
            nowcst_frame = griddata((coord_target_i.flatten(), coord_target_j.flatten()), 
                                    last_frame.flatten(), 
                                    (coord_source_i.flatten(), coord_source_j.flatten()), 
                                    method=self.interpolator, fill_value=0)
            
            # reshape output
            nowcst_frame = nowcst_frame.reshape(last_frame.shape)
            
            # converting to uint8
            nowcst_frame = nowcst_frame.astype('uint8')
            
            # add to the container
            nowcst_frames.append(nowcst_frame)
                    
        return np.moveaxis(np.dstack(nowcst_frames), -1, 0).copy()

##### PERSISTENCE #####
class EulerianPersistence:
    
    """
    The basic class for the Eulerian Persistence (Persistence) model implementation (weak baseline solution).

    Parameters
    ----------

    input_data: numpy.ndarray
        8-bit (uint8, 0-255) 3D numpy.ndarray (frames, dim_x, dim_y) of radar data for previous 10 minutes (2 frames)

        Default value: None

    lead_steps: int
        The required lead steps of nowcasting

        Default value: 12
    
    Examples
    --------
    
    See :ref:`/notebooks/nowcasting.ipynb`.
    """
    
    def __init__(self):
        
        self.input_data = None
        
        self.lead_steps = 12
    
    def run(self):
        '''
        The method for running the Sparse model with the parameters were specified in the EulerianPersistence class parameters.

        For running with the default parameters only the input data instance is required (.input_data parameter).

        Args:
            None
        
        Returns:
            numpy.ndarray: 3D numpy.ndarray of precipitation nowcasting (frames, dim_x, dim_y)
        '''

        last_frame = self.input_data[-1, :, :]

        forecast = np.dstack([last_frame for i in range(self.lead_steps)])

        return np.moveaxis(forecast, -1, 0).copy()