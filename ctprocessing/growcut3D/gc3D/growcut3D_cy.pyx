import numpy as np
import matplotlib.pyplot as plt

from skimage import color, io
from skimage import img_as_float

def g(x, y):
    return 1 - np.sqrt(np.sum((x - y) ** 2)) / np.sqrt(3)

def growcut3D(image, state, strength,max_iter = 500, window_size = 5):
    """Grow-cut segmentation.

    Parameters
    ----------
    image : (M, N, O) ndarray
        Input image.
    state : (M, N, O) ndarray
        Initial state, which stores (foreground/background, strength) for
        each pixel position or automaton.  The strength represents the
        certainty of the state (e.g., 1 is a hard seed value that remains
        constant throughout segmentation).
    strength:(M, N, O) ndarray
    max_iter : int, optional
        The maximum number of automata iterations to allow.  The segmentation
        may complete earlier if the state no longer varies.
    window_size : int
        Size of the neighborhood window.

    Returns
    -------
    mask : ndarray
        Segmented image.  A value of zero indicates background, one foreground.
    # 0 -> label
    # 1 -> strength
    """

    image = img_as_float(image)
    height, width, depth = image.shape[:3]
    ws = (window_size - 1) // 2
    #print(image.shape)
    changes = 1
    n = 0

    state_next = state.copy()
    strength_next = strength.copy()
    while changes > 0 and n < max_iter:
        changes = 0
        n += 1

        if n % 10 == 0:
            print n
        for k in range(depth):
            for j in range(width):
                for i in range(height):
                    C_p = image[i, j, k]
                    #S_p = state[i, j]
                    Strength_p= strength[i, j, k]
                    State_p = state[i, j, k]

                    for kk in xrange(max(0, k - ws), min(k + ws + 1, depth)):
                        for jj in xrange(max(0, j - ws), min(j + ws + 1, width)):
                            for ii in xrange(max(0, i - ws), min(i + ws + 1, height)):
                                # p -> current cell
                                # q -> attacker
                                C_q = image[ii, jj, kk]
                                #S_q = state[ii, jj]
                                Strength_q = strength[ii, jj, kk]
                                State_q = state[ii, jj, kk]



                                gc = g(C_q, C_p)
                                if gc * Strength_q > Strength_p:
                               # if gc * S_q[1] > S_p[1]:
                                    state_next[i, j, k] = State_q
                                    strength_next[i, j, k] = gc * Strength_q

                                    changes += 1

                                    break

        state = state_next
        strength=strength_next
    print("terminou")
    return state