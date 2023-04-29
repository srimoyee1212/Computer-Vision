import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
import util_sweep


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo (or mean albedo for RGB input images) for a pixel is less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights. All N images have the same dimension.
                  The images can either be all RGB (height x width x 3), or all grayscale (height x width x 1).

    Output:
        albedo -- float32 image. When the input 'images' are RGB, it should be of dimension height x width x 3,
                  while in the case of grayscale 'images', the dimension should be height x width x 1.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """

    I = np.array(images).reshape(np.shape(images)[0], np.prod(np.shape(images)[1:]))
    G = (np.linalg.inv((lights.T).dot(lights))).dot((lights.T).dot(I))

    kdN_shape = [G.shape[0]]
    kdN_shape.extend(np.shape(images)[1:])
    kdN_albedo = G.reshape(kdN_shape)
    albedo = np.linalg.norm(kdN_albedo, axis = 0)

    shape = list(np.shape(images)[1:])
    shape.append(3)
    normals = np.mean(G.T.reshape(shape), axis = 2)
    albedo_normal = np.linalg.norm(normals, axis = 2)
    normals /= (np.maximum(1e-7, albedo_normal[:,:,None]) * 1.0)

    return albedo, normals

    #raise NotImplementedError()

def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    projection_matrix = np.dot(K, Rt)
    height, width = points.shape[0], points.shape[1]
    projections = np.zeros((height, width, 2))
    pt = np.zeros(3)

    for i, row in enumerate(points):
        for j, column in enumerate(row):
            pt = np.array(points[i, j])
            vector_4d = np.array([pt[0], pt[1], pt[2], 1.0])
            homogenous_pt = projection_matrix.dot(vector_4d)
            projections[i, j] = np.array([homogenous_pt[0] / homogenous_pt[2], homogenous_pt[1] / homogenous_pt[2]])

    return projections
    #raise NotImplementedError()


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc_impl' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc_impl' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region; assumed to be odd
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    
    h, w, c = image.shape
    window = ncc_size // 2
    normalized = np.zeros((h, w, c * ncc_size ** 2))

    for i in range(window, h - window):
        for j in range(window, w - window):
            patch_vector = image[i - window:i + window + 1, j - window:j + window + 1, :]
            mean_vec = np.mean(patch_vector, axis=(0, 1))

            patch_vector = patch_vector - mean_vec[np.newaxis, np.newaxis, :]
            patch_vector = np.reshape(patch_vector, (ncc_size ** 2, c)).T.ravel()
            norm = np.linalg.norm(patch_vector)

            if norm >= 1e-6:
                normalized[i, j] = patch_vector / norm
            else:
                normalized[i, j] = np.zeros(c * (ncc_size ** 2))

    return normalized

   #raise NotImplementedError()

def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc_impl.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    ncc = np.sum(np.multiply(image1, image2), axis = 2)
    return ncc
    #raise NotImplementedError()
