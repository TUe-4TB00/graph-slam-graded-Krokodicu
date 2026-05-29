import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))   
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1])) 
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  

def add_pose(graph, initial_estimate):
    # Odometry from X(3) to X(4): rotate +45deg, drive 2 m, rotate +45deg more.
    rot45 = gtsam.Pose2(0.0, 0.0, np.pi / 4)
    forward = gtsam.Pose2(2.0, 0.0, 0.0)
    
    # Compose the sequence of movements
    odometry = rot45.compose(forward).compose(rot45)   

    # Add the odometry factor between X(3) and X(4)
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))

    # Propagate from the nominal X(3) position to prevent early noise from corrupting the seed
    nominal_pose_3 = gtsam.Pose2(4.0, 0.0, 0.0)
    initial_estimate.insert(X(4), nominal_pose_3.compose(odometry))

    return graph, initial_estimate