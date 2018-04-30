"""
Optical flow models for radar-based precipitation nowcasting
============================================================

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
    The basic class for Sparse models in the common Sparse Group.

    Methods
    _______

    Parameters
    __________
    params: dictionary of Shi-Tomasi's ['st_pars'] and Lukas-Kanade's ['lk_pars'] algorithm parameters

        ['st_pars']:
        maxCorners       (default: 200; ranges: [50, 300];  type: int)  - maximum number of corners to return
        qualityLevel     (default: 0.2; ranges: [0.1, 0.7], type: float)- minimal accepted quality of image corners
        minDistance      (default: 7;   ranges: [3, 15],    type: int)  - minimum possible Euclidean distance between the returned corners
        blockSize        (default: 21;  ranges: [10, 50],   type: int)  - size of an average block for computing a derivative covariation matrix over each pixel neighborhood
        
        ['lk_pars']:
        winSize          (default: 20;  ranges: [10, 30],   type: int)  - size of the search window at each pyramid level
        maxLevel         (default: 2;   ranges: [0, 4],     type: int)  - 0-based maximal pyramid level number

        link to the parameters description:
        http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=goodfeaturestotrack#goodfeaturestotrack
        http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html

    extrapolator: dictionary of parameters used for precipitation field extrapolation
        model       (default: LinearRegression()) - any regression model instance from sklearn library
        features    (default: "ordinal"; available: "polynomial") 

    warper:

    input_data:

    lead_steps:

    """
    def __init__(self):
        
        """
        self.params - dictionary of Shi-Tomasi's ['st_pars'] and Lukas-Kanade's ['lk_pars'] algorithm parameters
        ['st_pars']:
        maxCorners       (default: 200; ranges: [50, 300];  type: int)  - maximum number of corners to return
        qualityLevel     (default: 0.2; ranges: [0.1, 0.7], type: float)- minimal accepted quality of image corners
        minDistance      (default: 7;   ranges: [3, 15],    type: int)  - minimum possible Euclidean distance between the returned corners
        blockSize        (default: 21;  ranges: [10, 50],   type: int)  - size of an average block for computing a derivative covariation matrix over each pixel neighborhood
        ['lk_pars']:
        winSize          (default: 20;  ranges: [10, 30],   type: int)  - size of the search window at each pyramid level
        maxLevel         (default: 2;   ranges: [0, 4],     type: int)  - 0-based maximal pyramid level number

        link to the parameters description:
        http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=goodfeaturestotrack#goodfeaturestotrack
        http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html
        """
        
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
        input data: 3D numpy array (frames, dim_x, dim_y) of radar data for previous 2 hours*
        forecast: 3D numpy array with predictions for 1 hour ahead
        * for consistency with more sophisticated approaches
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
    
    def __init__(self):
        
        """
        self.params - dictionary of Shi-Tomasi's ['st_pars'] and Lukas-Kanade's ['lk_pars'] algorithm parameters
        ['st_pars']:
        maxCorners       (default: 200; ranges: [50, 300];  type: int)  - maximum number of corners to return
        qualityLevel     (default: 0.2; ranges: [0.1, 0.7], type: float)- minimal accepted quality of image corners
        minDistance      (default: 7;   ranges: [3, 15],    type: int)  - minimum possible Euclidean distance between the returned corners
        blockSize        (default: 21;  ranges: [10, 50],   type: int)  - size of an average block for computing a derivative covariation matrix over each pixel neighborhood
        ['lk_pars']:
        winSize          (default: 20;  ranges: [10, 30],   type: int)  - size of the search window at each pyramid level
        maxLevel         (default: 2;   ranges: [0, 4],     type: int)  - 0-based maximal pyramid level number

        link to the parameters description:
        http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=goodfeaturestotrack#goodfeaturestotrack
        http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html
        """
        
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
        input data: 3D numpy array (frames, dim_x, dim_y) of radar data for previous 2 hours*
        forecast: 3D numpy array with predictions for 1 hour ahead
        * for consistency with more sophisticated approaches
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
    
    def __init__(self):
        
        """
        self.params - dictionary of Farnerback's algorithm parameters
        pyr_scale  (default: 0.5; ranges: [0.1, 0.9]; type: float) - image scale to build pyramids
        levels     (default: 3;   ranges: [1, 7],     type: int)   - number of pyramid layers
        winsize    (default: 15;  ranges: [5, 30],    type: int)   - averaging window size
        iterations (default: 3;   ranges: [2, 10],    type: int)   - number of iterations at each pyramid level
        poly_n     (default: 5;   ranges: [3, 10],    type: int)   - size of the pixel neighborhood
        poly_sigma (default: 1.1; ranges: [0.9, 2],   type: float) - std of the Gaussian for smoothing derivatives

        link to the parameters description:
        http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        """
        
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
        input data: 3D numpy array (frames, dim_x, dim_y) of radar data for previous 2 hours*
        forecast: 3D numpy array with predictions for 1 hour ahead
        * for consistency with more sophisticated approaches
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
    
    def __init__(self):
        
        """
        self.params - dictionary of Farnerback's algorithm parameters
        pyr_scale  (default: 0.5; ranges: [0.1, 0.9]; type: float) - image scale to build pyramids
        levels     (default: 3;   ranges: [1, 7],     type: int)   - number of pyramid layers
        winsize    (default: 15;  ranges: [5, 30],    type: int)   - averaging window size
        iterations (default: 3;   ranges: [2, 10],    type: int)   - number of iterations at each pyramid level
        poly_n     (default: 5;   ranges: [3, 10],    type: int)   - size of the pixel neighborhood
        poly_sigma (default: 1.1; ranges: [0.9, 2],   type: float) - std of the Gaussian for smoothing derivatives

        link to the parameters description:
        http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        """
        
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
        input data: 3D numpy array (frames, dim_x, dim_y) of radar data for previous 2 hours*
        forecast: 3D numpy array with predictions for 1 hour ahead
        * for consistency with more sophisticated approaches
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
        
        # calculate target coordinats OF MESH only once
        coord_target_i = coord_source_i + delta_x
        coord_target_j = coord_source_j + delta_y
        
        # propagate our image through time based on dense flow
        # container for our nowcasts
        nowcst_frames = []
        
        for lead_step in range(self.lead_steps):       
            
            # we suppose that old values just go to the new locations (Lagrangian persistense)
            # but we need to regrid data
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
            
            # and now the nowcast frame became an a last frame
            # and in the another iteration it will go through
            # the same flow vectors
            last_frame = nowcst_frame.copy()
        
        return np.moveaxis(np.dstack(nowcst_frames), -1, 0).copy()

##### PERSISTENCE #####
class EulerianPersistence:
    
    """
    Eulerian Persistence (Persistence) model
    """
    
    def __init__(self):
        
        self.input_data = None
        
        self.lead_steps = 12
    
    def run(self):
        '''
        input data: 3D numpy array (frames, dim_x, dim_y) of radar data for previous 2 hours*
        forecast: 3D numpy array with predictions for 1 hour ahead
        * for consistency with more sophisticated approaches
        '''

        last_frame = self.input_data[-1, :, :]

        forecast = np.dstack([last_frame for i in range(self.lead_steps)])

        return np.moveaxis(forecast, -1, 0).copy()