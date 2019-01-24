"""
``rainymotion.models``: optical flow models for radar-based precipitation nowcasting
====================================================================================

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import wradlib.ipol as ipol
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import map_coordinates
import skimage.transform as sktf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from rainymotion.utils import RYScaler, inv_RYScaler

### SPARSE GROUP ###
### --- helpers --- ###
def _sparse_linear(data_instance,
                   of_params={'st_pars' : dict(maxCorners = 200,
                                               qualityLevel = 0.2,
                                               minDistance = 7,
                                               blockSize = 21 ),
                              'lk_pars' : dict(winSize  = (20,20),
                                               maxLevel = 2,
                                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0))},
                   extrapol_params={"model": LinearRegression(), "features": "ordinal"},
                   lead_steps=12):

    # find features to track
    old_corners = cv2.goodFeaturesToTrack(data_instance[0], mask = None, **of_params['st_pars'])

    # Set containers to collect results (time steps in rows, detected corners in columns)
    #   corner x coords
    x = np.full((data_instance.shape[0], len(old_corners)), np.nan)
    #   corner y coords
    y = np.full((data_instance.shape[0], len(old_corners)), np.nan)
    #   Assign persistent corner IDs
    ids = np.arange(len(old_corners))

    # fill in first values
    x[0, :] = old_corners[:, 0, 0]
    y[0, :] = old_corners[:, 0, 1]

    # track corners by optical flow algorithm
    for i in range(1, data_instance.shape[0]):

        new_corners, st, err = cv2.calcOpticalFlowPyrLK(prevImg=data_instance[i-1],
                                                        nextImg=data_instance[i],
                                                        prevPts=old_corners,
                                                        nextPts=None,
                                                        **of_params['lk_pars'])

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
    x_new = np.full((lead_steps, x.shape[1]), np.nan)
    y_new = np.full((lead_steps, y.shape[1]), np.nan)

    for i in range(x.shape[1]):

        x_train = x[:, i]
        y_train = y[:, i]

        X = np.arange(x.shape[0] + lead_steps)

        if extrapol_params["features"] == "polynomial":
            polyfeatures = PolynomialFeatures(2)
            X = polyfeatures.fit_transform(X.reshape(-1, 1))
            X_train = X[:x.shape[0], :]
            X_pred  = X[x.shape[0]:, :]
        else:
            X = X.reshape(-1, 1)
            X_train = X[:x.shape[0], :]
            X_pred  = X[x.shape[0]:, :]

        x_pred = extrapol_params["model"].fit(X_train, x_train).predict(X_pred)
        y_pred = extrapol_params["model"].fit(X_train, y_train).predict(X_pred)

        x_new[:, i] = x_pred
        y_new[:, i] = y_pred

    # define source corners in appropriate format
    pts_source = np.hstack([x[-1, :].reshape(-1, 1), y[-1, :].reshape(-1, 1)])

    # define container for targets in appropriate format
    pts_target_container = [np.hstack([x_new[i, :].reshape(-1, 1),
                                       y_new[i, :].reshape(-1, 1)]) for i in range(x_new.shape[0])]

    return pts_source, pts_target_container


def _sparse_sd(data_instance,
               of_params={'st_pars' : dict(maxCorners = 200,
                                           qualityLevel = 0.2,
                                           minDistance = 7,
                                           blockSize = 21 ),
                          'lk_pars' : dict(winSize  = (20,20),
                                           maxLevel = 2,
                                           criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0))},
               lead_steps=12):

    # define penult and last frames
    penult_frame = data_instance[-2]
    last_frame = data_instance[-1]

    # find features to track
    old_corners = cv2.goodFeaturesToTrack(data_instance[0], mask = None, **of_params['st_pars'])

    # track corners by optical flow algorithm
    new_corners, st, err = cv2.calcOpticalFlowPyrLK(prevImg=penult_frame,
                                                    nextImg=last_frame,
                                                    prevPts=old_corners,
                                                    nextPts=None,
                                                    **of_params['lk_pars'])

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

    for lead_step in range(lead_steps):
        pts_target_container.append( pts_source + delta * (lead_step + 1) )

    return pts_source, pts_target_container


class Sparse:

    def __init__(self):

        self.of_params =  {'st_pars' : dict(maxCorners = 200,
                           qualityLevel = 0.2,
                           minDistance = 7,
                           blockSize = 21 ),
                           'lk_pars' : dict(winSize  = (20,20),
                           maxLevel = 2,
                           criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0))}

        self.extrapolation = "linear"

        self.warper = "affine"

        self.input_data = None

        self.scaler = RYScaler

        self.inverse_scaler = inv_RYScaler

        self.lead_steps = 12

    def run(self):

        # define available transformations dictionary
        transformations = {'euclidean': sktf.EuclideanTransform(),
                           'similarity': sktf.SimilarityTransform(),
                           'affine': sktf.AffineTransform(),
                           'projective': sktf.ProjectiveTransform(),
                          }

        # scale input data to uint8 [0-255] with self.scaler
        data_scaled, c1, c2 = self.scaler(self.input_data)

        # set up transformer object
        trf = transformations[self.warper]

        # obtain source and target points
        if self.extrapolation == "linear":
            pts_source, pts_target_container = _sparse_linear(data_instance=data_scaled,
                                                              of_params=self.of_params,
                                                              lead_steps=self.lead_steps)
        elif self.extrapolation == "simple_delta":
            pts_source, pts_target_container = _sparse_sd(data_instance=data_scaled,
                                                          of_params=self.of_params,
                                                          lead_steps=self.lead_steps)

        # now we can start to find nowcasted image
        # for every candidate of projected sets of points

        # container for our nowcasts
        last_frame = data_scaled[-1]
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


        nowcst_frames = np.stack(nowcst_frames, axis=0)

        nowcst_frames = self.inverse_scaler(nowcst_frames, c1, c2)

        return nowcst_frames


class SparseSD:

    def __init__(self):

        self.of_params =  {'st_pars' : dict(maxCorners = 200,
                           qualityLevel = 0.2,
                           minDistance = 7,
                           blockSize = 21 ),
                           'lk_pars' : dict(winSize  = (20,20),
                           maxLevel = 2,
                           criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0))}

        self.extrapolation = "simple_delta"

        self.warper = "affine"

        self.input_data = None

        self.scaler = RYScaler

        self.inverse_scaler = inv_RYScaler

        self.lead_steps = 12

    def run(self):

        # define available transformations dictionary
        transformations = {'euclidean': sktf.EuclideanTransform(),
                           'similarity': sktf.SimilarityTransform(),
                           'affine': sktf.AffineTransform(),
                           'projective': sktf.ProjectiveTransform(),
                          }

        # scale input data to uint8 [0-255] with self.scaler
        data_scaled, c1, c2 = self.scaler(self.input_data)

        # set up transformer object
        trf = transformations[self.warper]

        # obtain source and target points
        if self.extrapolation == "linear":
            pts_source, pts_target_container = _sparse_linear(data_instance=data_scaled,
                                                              of_params=self.of_params,
                                                              lead_steps=self.lead_steps)
        elif self.extrapolation == "simple_delta":
            pts_source, pts_target_container = _sparse_sd(data_instance=data_scaled,
                                                          of_params=self.of_params,
                                                          lead_steps=self.lead_steps)

        # now we can start to find nowcasted image
        # for every candidate of projected sets of points

        # container for our nowcasts
        last_frame = data_scaled[-1]
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


        nowcst_frames = np.stack(nowcst_frames, axis=0)

        nowcst_frames = self.inverse_scaler(nowcst_frames, c1, c2)

        return nowcst_frames

### DENSE GROUP ###
### --- helpers --- ###

# filling holes (zeros) in velocity field
def _fill_holes(of_instance, threshold=0):

    # calculate velocity scalar
    vlcty = np.sqrt( of_instance[::, ::, 0]**2 + of_instance[::, ::, 1]**2 )

    # zero mask
    zero_holes = vlcty <= threshold

    # targets
    coord_target_i, coord_target_j = np.meshgrid(range(of_instance.shape[1]), range(of_instance.shape[0]))

    # source
    coord_source_i, coord_source_j = coord_target_i[~zero_holes], coord_target_j[~zero_holes]
    delta_x_source = of_instance[::, ::, 0][~zero_holes]
    delta_y_source = of_instance[::, ::, 1][~zero_holes]

    # reshape
    src = np.vstack( (coord_source_i.ravel(), coord_source_j.ravel()) ).T
    trg = np.vstack( (coord_target_i.ravel(), coord_target_j.ravel()) ).T

    # create an object
    interpolator = ipol.Idw(src, trg)

    #
    delta_x_target = interpolator(delta_x_source.ravel())
    delta_y_target = interpolator(delta_y_source.ravel())

    # reshape output
    delta_x_target = delta_x_target.reshape(of_instance.shape[0], of_instance.shape[1])
    delta_y_target = delta_y_target.reshape(of_instance.shape[0], of_instance.shape[1])

    return np.stack([delta_x_target, delta_y_target], axis=-1)


# calculate optical flow
def _calculate_of(data_instance,
                  method="DIS",
                  direction="forward"):

    # define frames order
    if direction == "forward":
        prev_frame = data_instance[-2]
        next_frame = data_instance[-1]
        coef = 1.0
    elif direction == "backward":
        prev_frame = data_instance[-1]
        next_frame = data_instance[-2]
        coef = -1.0

    # calculate dense flow
    if method == "Farneback":
        of_instance = cv2.optflow.createOptFlow_Farneback()
    elif method == "DIS":
        of_instance = cv2.optflow.createOptFlow_DIS()
    elif method == "DeepFlow":
        of_instance = cv2.optflow.createOptFlow_DeepFlow()
    elif method == "PCAFlow":
        of_instance = cv2.optflow.createOptFlow_PCAFlow()
    elif method == "SimpleFlow":
        of_instance = cv2.optflow.createOptFlow_SimpleFlow()
    elif method == "SparseToDense":
        of_instance = cv2.optflow.createOptFlow_SparseToDense()

    delta = of_instance.calc(prev_frame, next_frame, None) * coef

    if method in ["Farneback", "SimpleFlow"]:
        # variational refinement
        delta = cv2.optflow.createVariationalFlowRefinement().calc(prev_frame, next_frame, delta)
        delta = np.nan_to_num(delta)
        delta = _fill_holes(delta)

    return delta


# constant-vector advection
def _advection_constant_vector(of_instance, lead_steps=12):

    delta_x = of_instance[::, ::, 0]
    delta_y = of_instance[::, ::, 1]

    # make a source meshgrid
    coord_source_i, coord_source_j = np.meshgrid(range(of_instance.shape[1]), range(of_instance.shape[0]))

    # calculate new coordinates of radar pixels
    coord_targets = []
    for lead_step in range(lead_steps):
        coord_target_i = coord_source_i + delta_x * (lead_step + 1)
        coord_target_j = coord_source_j + delta_y * (lead_step + 1)
        coord_targets.append([coord_target_i, coord_target_j])

    coord_source = [coord_source_i, coord_source_j]

    return coord_source, coord_targets


# semi-Lagrangian advection
def _advection_semi_lagrangian(of_instance, lead_steps=12):

    delta_x = of_instance[::, ::, 0]
    delta_y = of_instance[::, ::, 1]

    # make a source meshgrid
    coord_source_i, coord_source_j = np.meshgrid(range(of_instance.shape[1]), range(of_instance.shape[0]))

    # create dynamic delta holders
    delta_xi = delta_x.copy()
    delta_yi = delta_y.copy()

    # Block for calculation displacement
    # init placeholders
    coord_targets = []
    for lead_step in range(lead_steps):

        # calculate corresponding targets
        coord_target_i = coord_source_i + delta_xi
        coord_target_j = coord_source_j + delta_yi

        coord_targets.append([coord_target_i, coord_target_j])

        # now update source coordinates
        coord_source_i = coord_target_i
        coord_source_j = coord_target_j
        coord_source = [coord_source_j.ravel(), coord_source_i.ravel()]

        # update deltas
        delta_xi = map_coordinates(delta_x, coord_source).reshape(of_instance.shape[1], of_instance.shape[0])
        delta_yi = map_coordinates(delta_y, coord_source).reshape(of_instance.shape[1], of_instance.shape[0])

    # reinitialization of coordinates source
    coord_source_i, coord_source_j = np.meshgrid(range(of_instance.shape[1]), range(of_instance.shape[0]))
    coord_source = [coord_source_i, coord_source_j]

    return coord_source, coord_targets


# interpolation routine
def _interpolator(points, coord_source, coord_target, method="idw"):

    coord_source_i, coord_source_j = coord_source
    coord_target_i, coord_target_j = coord_target

    # reshape
    trg = np.vstack( (coord_source_i.ravel(), coord_source_j.ravel()) ).T
    src = np.vstack( (coord_target_i.ravel(), coord_target_j.ravel()) ).T

    if method == "nearest":
        interpolator = NearestNDInterpolator(src, points.ravel(), tree_options={"balanced_tree": False})
        points_interpolated = interpolator(trg)
    elif method == "linear":
        interpolator = LinearNDInterpolator(src, points.ravel(), fill_value=0)
        points_interpolated = interpolator(trg)
    elif method == "idw":
        interpolator = ipol.Idw(src, trg)
        points_interpolated = interpolator(points.ravel())


    # reshape output
    points_interpolated = points_interpolated.reshape(points.shape)

    return points_interpolated.astype(points.dtype)

class Dense:

    def __init__(self):

        self.input_data = None

        self.scaler = RYScaler

        self.lead_steps = 12

        self.of_method = "DIS"

        self.direction = "forward"

        self.advection = "constant-vector"

        self.interpolation = "idw"

    def run(self):

        # scale input data to uint8 [0-255] with self.scaler
        scaled_data, c1, c2 = self.scaler(self.input_data)

        # calculate optical flow
        of = _calculate_of(scaled_data, method=self.of_method, direction=self.direction)

        # advect pixels accordingly
        if self.advection == "constant-vector":
            coord_source, coord_targets = _advection_constant_vector(of, lead_steps=self.lead_steps)
        elif self.advection == "semi-lagrangian":
            coord_source, coord_targets = _advection_semi_lagrangian(of, lead_steps=self.lead_steps)

        # nowcasts placeholder
        nowcasts = []

        # interpolation
        for lead_step in range(self.lead_steps):
            nowcasts.append(_interpolator(self.input_data[-1], coord_source, coord_targets[lead_step], method=self.interpolation))

        # reshaping
        nowcasts = np.moveaxis(np.dstack(nowcasts), -1, 0)

        return nowcasts


class DenseRotation:

    def __init__(self):

        self.input_data = None

        self.scaler = RYScaler

        self.lead_steps = 12

        self.of_method = "DIS"

        self.direction = "forward"

        self.advection = "semi-lagrangian"

        self.interpolation = "idw"

    def run(self):

        # scale input data to uint8 [0-255] with self.scaler
        scaled_data, c1, c2 = self.scaler(self.input_data)

        # calculate optical flow
        of = _calculate_of(scaled_data, method=self.of_method, direction=self.direction)

        # advect pixels accordingly
        if self.advection == "constant-vector":
            coord_source, coord_targets = _advection_constant_vector(of, lead_steps=self.lead_steps)
        elif self.advection == "semi-lagrangian":
            coord_source, coord_targets = _advection_semi_lagrangian(of, lead_steps=self.lead_steps)

        # nowcasts placeholder
        nowcasts = []

        # interpolation
        for lead_step in range(self.lead_steps):
            nowcasts.append(_interpolator(self.input_data[-1], coord_source, coord_targets[lead_step], method=self.interpolation))

        # reshaping
        nowcasts = np.moveaxis(np.dstack(nowcasts), -1, 0)

        return nowcasts


class Persistence:

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
