'''
Created on 11/04/2015
@author: Andrew Chalmers
This code computes the Earth Mover's Distance, as explained here:
http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/RUBNER/emd.htm
This is done using numpy, scipy (minimize)
There is a simple example of two distributions computed by getExampleSignatures()
This example is chosen in order to compare the result with a C implementation
found here:
http://robotics.stanford.edu/~rubner/emd/default.htm
'''

import numpy as np
import pandas as pd
import scipy.optimize
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import os


# Constraints
def positivity(f):
    '''
    Constraint 1:
    Ensures flow moves from source to target
    '''
    return f


def fromSrc(f, wp, i, shape):
    """
    Constraint 2:
    Limits supply for source according to weight
    """
    fr = np.reshape(f, shape)
    f_sumColi = np.sum(fr[i, :])
    return wp[i] - f_sumColi


def toTgt(f, wq, j, shape):
    """
    Constraint 3:
    Limits demand for target according to weight
    """
    fr = np.reshape(f, shape)
    f_sumRowj = np.sum(fr[:, j])
    return wq[j] - f_sumRowj


def maximiseTotalFlow(f, wp, wq):
    """
    Constraint 4:
    Forces maximum supply to move from source to target
    """
    return f.sum() - np.minimum(wp.sum(), wq.sum())


# Objective function
def flow(f, D):
    """
    The objective function
    The flow represents the amount of goods to be moved
    from source to target
    """
    f = np.reshape(f, D.shape)
    return (f * D).sum()


# Distance
def groundDistance(x1, x2, norm=2):
    """
    L-norm distance
    Default norm = 2
    """
    return np.linalg.norm(x1 - x2, norm)


def getDistanceOfPoints(pi, pj):
    lat1, lon1, lat2, lon2 = list(map(radians, [float(pi[0]), float(pi[1]),
                                                float(pj[0]), float(pj[1])]))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    m = 6371000 * c
    return m


# Distance matrix
def getDistMatrix(s1, s2, norm=2):
    """
    Computes the distance matrix between the source
    and target distributions.
    The ground distance is using the L-norm (default L2 norm)
    """
    # Slow method
    # rows = s1 feature length
    # cols = s2 feature length
    numFeats1 = s1.shape[0]
    numFeats2 = s2.shape[0]
    distMatrix = np.zeros((numFeats1, numFeats2))

    for i in range(0, numFeats1):
        for j in range(0, numFeats2):
            distMatrix[i, j] = groundDistance(s1[i], s2[j], norm)
            # distMatrix[i, j] = getDistanceOfPoints(s1[i], s2[j])
    # Fast method (requires scipy.spatial)
    # import scipy.spatial
    # distMatrix = scipy.spatial.distance.cdist(s1, s2)

    return distMatrix


# Flow matrix
def getFlowMatrix(P, Q, D):
    """
    Computes the flow matrix between P and Q
    """
    numFeats1 = P[0].shape[0]
    numFeats2 = Q[0].shape[0]
    shape = (numFeats1, numFeats2)

    # Constraints
    cons1 = [{'type': 'ineq', 'fun': positivity},
             {'type': 'eq', 'fun': maximiseTotalFlow, 'args': (P[1], Q[1],)}]

    cons2 = [{'type': 'ineq', 'fun': fromSrc, 'args': (P[1], i, shape,)} for i in range(numFeats1)]
    cons3 = [{'type': 'ineq', 'fun': toTgt, 'args': (Q[1], j, shape,)} for j in range(numFeats2)]

    cons = cons1 + cons2 + cons3

    # Solve for F (solve transportation problem)
    F_guess = np.zeros(D.shape)
    F = scipy.optimize.minimize(flow, F_guess, args=(D,), constraints=cons)
    F = np.reshape(F.x, (numFeats1, numFeats2))

    return F


# Normalised EMD
def EMD(F, D):
    """
    EMD formula, normalised by the flow
    """
    return (F * D).sum() / F.sum()


# Runs EMD program
def getEMD(P, Q, norm=2):
    """
    EMD computes the Earth Mover's Distance between
    the distributions P and Q

    P and Q are of shape (2,N)

    Where the first row are the set of N features
    The second row are the corresponding set of N weights

    The norm defines the L-norm for the ground distance
    Default is the Euclidean norm (norm = 2)
    """

    D = getDistMatrix(P[0], Q[0], norm)
    F = getFlowMatrix(P, Q, D)

    return EMD(F, D)


# Examples
# def getExampleSignatures1():
#     """
#     returns signature1[features][weights], signature2[features][weights]
#     """
#     features1 = np.array([[100, 40, 22],
#                           [211, 20, 2],
#                           [32, 190, 150],
#                           [2, 100, 100]])
#     weights1 = np.array([0.4, 0.3, 0.2, 0.1])
#
#     features2 = np.array([[0, 0, 0],
#                           [50, 100, 80],
#                           [255, 255, 255]])
#     weights2 = np.array([0.5, 0.3, 0.2])
#
#     signature1 = (features1, weights1)
#     signature2 = (features2, weights2)
#
#     return signature1, signature2
#
#
# def getExampleSignatures2():
#     """
#     returns signature1[features][weights], signature2[features][weights]
#     """
#     features1 = np.array([[100, 40, 22],
#                           [211, 20, 2],
#                           [32, 190, 150],
#                           [2, 100, 100]])
#     weights1 = np.array([1.0, 1.0, 1.0, 1.0])
#
#     features2 = np.array([[0, 0, 0],
#                           [50, 100, 80],
#                           [255, 255, 255]])
#     weights2 = np.array([1.0, 1.0, 1.0])
#
#     signature1 = (features1, weights1)
#     signature2 = (features2, weights2)
#
#     return signature1, signature2


# def getExample_GaussianHistograms(N=15, showPlot=True):
#     """
#     returns signature1[features][weights], signature2[features][weights]
#     """
#     x = np.linspace(-1, 1, N)
#     y1 = np.exp(-np.power(x - 0.0, 2.0) / (2 * np.power(0.2, 2.0)))
#     y2 = np.exp(-np.power(x - 0.5, 2.0) / (2 * np.power(0.2, 2.0)))
#     y1 /= np.sum(y1)
#     y2 /= np.sum(y2)
#
#     if showPlot:
#         plt.bar(x, y1, width=0.1, alpha=0.5)
#         plt.bar(x, y2, width=0.1, alpha=0.5)

# features1 = y1.reshape((N,1))
# weights1 = (1.0/N) * np.ones((N))

# features2 = y2.reshape((N,1))
# weights2 = (1.0/N) * np.ones((N))

# signature1 = (features1, weights1)
# signature2 = (features2, weights2)

# signature1 = (x.reshape((N, 1)), y1.reshape((N, 1)))
# signature2 = (x.reshape((N, 1)), y2.reshape((N, 1)))
#
# return signature1, signature2


# def doRubnerComparisonExample():
#     # Setup
#     P, Q = getExampleSignatures1()
#
#     # Get EMD
#     emd = getEMD(P, Q)
#
#     # Output result
#     print('We got: ' + str(emd))
#     print('Rubner C example got 160.54277')


# def doGaussianHistogramExample():
#     showPlot = True
#
#     # Setup
#     P, Q = getExample_GaussianHistograms(showPlot=showPlot)
#
#     # Get EMD
#     emd = getEMD(P, Q)
#
#     # Output result
#     print('EMD: ' + str(emd))
#
#     if showPlot:
#         plt.show()

def emddistance(file_a, file_b):
    data_a = pd.read_csv(file_a)
    # print(data_a)
    data_b = pd.read_csv(file_b)
    line_a = np.array(data_a.loc[:, ['laltitude', 'longitude']])
    line_b = np.array(data_b.loc[:, ['laltitude', 'longitude']])
    weights_a = np.ones(line_a.shape[0])
    weights_b = np.ones(line_b.shape[0])
    signature_a = (line_a, weights_a)
    signature_b = (line_b, weights_b)
    emd = getEMD(signature_a, signature_b)
    # print("emd=" + str(emd))
    return emd


if __name__ == '__main__':

    # emddistance(r"C:\Users\wisdo\Documents\StayPoint\rome_project\rome_each\a2_1_basic.csv",
    #             r"C:\Users\wisdo\Documents\StayPoint\rome_project\rome_each\a2_1_density.csv")
    file_dir = r"C:\Users\wisdo\Documents\StayPoint\rome_project\rome_each"
    file_name_a = r"11_1_basic.csv"
    file_choose = os.path.join(file_dir, file_name_a)
    # r"C:\Users\wisdo\Documents\StayPoint\rome_project\rome_each\a2_1_basic.csv"
    filelist = {}
    for dirname, dirnames, filenames in os.walk(file_dir):
        filenames.remove(file_name_a)
        filenum = len(filenames)
        for i in range(filenum):
            gpsfile_a = os.path.join(dirname, filenames[i])
            filelist[filenames[i].strip("_basic.csv")] = emddistance(file_choose, gpsfile_a)
    dist_list = pd.Series(filelist).sort_values()
    # dist_list.to_csv(os.path.join(r"C:\Users\wisdo\Documents\StayPoint\rome_project\rome_each_dist",file_name_a))
    print(dist_list.iloc[0:5])
