# coding=utf-8
import numpy as np
import numpy.linalg as la
from scipy import signal, ndimage

# RANSAC HELPER FUNCTIONS
def valid_cloud(img_reshaped):
    return img_reshaped[np.where(img_reshaped[:,2]!=0)]

def sampling(img_reshaped, samples_to_fit):
    samples = img_reshaped[np.random.choice(img_reshaped.shape[0], size=samples_to_fit, replace=False), :]
    return samples

def svd(A):
    u, s, vh = la.svd(A)
    S = np.zeros(A.shape)
    S[:s.shape[0], :s.shape[0]] = np.diag(s)
    return u, S, vh

def model_estimation_svd(samples):
    #ones=np.expand_dims(np.ones(samples.shape[1]), axis=1)
    #samples=np.hstack((samples, ones))
    U, S, Vt = svd(samples)
    null_space = Vt[-1, :]
    return null_space

def find_inliers(img_reshaped, model, threshold):

    #calculate distance to points in image
    #distances = np.abs(model[0] * img_reshaped[:, 0] + model[1] * img_reshaped[:, 1]
    #                   + model[2] * img_reshaped[:, 2] - model[3])
    #distances = distances / np.sqrt(np.sum(np.power(model[:-1], 2)))

    distances = np.abs(img_reshaped @ model) / np.sqrt(model[0] ** 2 + model[1] ** 2 + model[2] ** 2)

    #point is an inlier if it's within a threshold
    inliers = distances < threshold
    num_inliers = np.count_nonzero(inliers == True)

    return num_inliers

## MASK CALCULATION
def get_plane_mask(img, model, threshold):
    img_reshaped = img.reshape((-1, 3))
    #distances = model[0] * img_reshaped[:, 0] + model[1] * img_reshaped[:, 1] + model[2] * img_reshaped[:, 2] - model[3]
    ones = np.expand_dims(np.ones(img_reshaped.shape[0]), axis=1)
    img_reshaped = np.hstack((img_reshaped, ones))

    distances = np.abs(img_reshaped @ model) / np.sqrt(model[0] ** 2 + model[1] ** 2 + model[2] ** 2)
    mask = np.where(distances < threshold, 1, 0)

    return mask.reshape((img.shape[0], img.shape[1]))

## OPENING AND CLOSING
def opening(mask, num_neighbors=None):
    if num_neighbors == None:
        return ndimage.binary_opening(mask).astype(np.int)
    else:
        return ndimage.binary_opening(mask, structure=np.ones((num_neighbors, num_neighbors))).astype(np.int)

def closing(mask, num_neighbors):
    if num_neighbors == None:
        return ndimage.binary_closing(mask).astype(np.int)
    else:
        return ndimage.binary_closing(mask, structure=np.ones((num_neighbors, num_neighbors))).astype(np.int)