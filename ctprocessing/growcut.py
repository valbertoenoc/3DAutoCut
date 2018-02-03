""" Implementation of the grow-cut algorithm """

import numpy as np
import matplotlib.pyplot as plt

from skimage import color, io
from skimage import img_as_float
from math import sqrt




def g(x, y):
    return 1 - np.sqrt(np.sum((x - y) ** 2)) / sqrt(3)


def growcut(image, state, max_iter=500, window_size=5):
    """Grow-cut segmentation.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    state : (M, N, 2) ndarray
        Initial state, which stores (foreground/background, strength) for
        each pixel position or automaton.  The strength represents the
        certainty of the state (e.g., 1 is a hard seed value that remains
        constant throughout segmentation).
    max_iter : int, optional
        The maximum number of automata iterations to allow.  The segmentation
        may complete earlier if the state no longer varies.
    window_size : int
        Size of the neighborhood window.

    Returns
    -------
    mask : ndarray
        Segmented image.  A value of zero indicates background, one foreground.

    """
    image = img_as_float(image)
    height, width = image.shape[:2]
    ws = (window_size - 1) // 2

    changes = 1
    n = 0

    state_next = state.copy()

    while changes > 0 and n < max_iter:
        changes = 0
        n += 1

        for j in range(width):
            for i in range(height):
                C_p = image[i, j]
                S_p = state[i, j]

                for jj in xrange(max(0, j - ws), min(j + ws + 1, width)):
                    for ii in xrange(max(0, i - ws), min(i + ws + 1, height)):
                        # p -> current cell
                        # q -> attacker
                        C_q = image[ii, jj]
                        S_q = state[ii, jj]

                        gc = g(C_q, C_p)

                        if gc * S_q[1] > S_p[1]:
                            state_next[i, j, 0] = S_q[0]
                            state_next[i, j, 1] = gc * S_q[1]

                            changes += 1
                            break

        state = state_next

    return state[:, :, 0]

def form_segmentation(img, imGT, spread=0.01):
    """ Forms a segmentation using the growcut method """

    img, ground_truth = img.copy(), imGT.copy()

    foreground_indices = np.nonzero(ground_truth.flatten())[0]
    background_indices = np.nonzero(~(ground_truth.flatten()))[0]

    label = np.zeros(ground_truth.shape, dtype=np.int)

    random_foreground_indices = foreground_indices[
        np.random.randint(0, len(foreground_indices), spread * len(foreground_indices))]
    random_background_indices = background_indices[
        np.random.randint(0, len(background_indices), spread * len(background_indices))]

    label.flat[random_foreground_indices] = 1
    label.flat[random_background_indices] = -1

    strength = np.zeros_like(img, dtype=np.float64)
    strength[np.nonzero(label)] = 1.0

    mask = growcut(color.gray2rgb(img), np.dstack((label, strength)), window_size=4)
    # mask = gcn.growcut(color.gray2rgb(img), np.dstack((label, strength)), window_size=4)

    return img, mask, ground_truth, label, strength


def plot_segmentation(img, mask, ground_truth, label, strength):
    """ Plot the segmentation results """
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16, 8))
    ax1.imshow(img, interpolation='nearest', cmap='gray')
    _ = ax1.axis('off')
    ax1.set_title('natural image')

    ax2.imshow(label, interpolation='nearest', cmap='gray')
    _ = ax2.axis('off')
    ax2.set_title('growcut labels')

    _ = ax3.axis('off')

    ax4.imshow(ground_truth, interpolation='nearest', cmap='gray')
    ax4.set_title('human segmentation')
    _ = ax4.axis('off')

    ax5.imshow(mask == 1, interpolation='nearest', cmap='gray')
    ax5.set_title('growcut segmentation')

    _ = ax5.axis('off')
    ax6.imshow((mask == 1).astype(int) + ground_truth.astype(int), interpolation='nearest', cmap='jet')
    _ = ax6.axis('off')
    ax6.set_title('segmentation error')

    plt.tight_layout()