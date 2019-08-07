import numpy as np
import numpy.linalg as la


def obj(x, obstacles, c1, c2, print_components=False):
    eps = 10**(-6)
    x = x.astype(np.float64)
    obstacle_vecs = x.reshape(-1, 1, 2) - obstacles

    obstacle_sq_dists = np.sum(obstacle_vecs**2, axis=-1)
    point_dists = np.sum(np.diff(x, axis=0)**2)

    avoidance_comp = c1 * np.sum(1/(eps + obstacle_sq_dists))
    spacing_comp = c2 * point_dists

    if print_components:
        print(avoidance_comp, spacing_comp)
    return avoidance_comp + spacing_comp


def dobj(x, obstacles, c1, c2):
    eps = 10**(-6)
    x = x.astype(np.float64)

    seg_vecs = np.diff(x, axis=0)

    npoints, _ = x.shape

    # (npoints, nobstacles, 2)
    obstacle_vecs = x.reshape(-1, 1, 2) - obstacles
    obstacle_sq_dists = np.sum(obstacle_vecs**2, axis=-1)

    result = np.zeros_like(x)
    result += 2*c1*np.sum(
            - obstacle_vecs
            /(eps+obstacle_sq_dists.reshape(npoints, -1, 1))**2, axis=1)

    result[1:] += 2*c2*seg_vecs
    result[:-1] += -2*c2*seg_vecs

    result[0] = 0
    result[-1] = 0

    return result