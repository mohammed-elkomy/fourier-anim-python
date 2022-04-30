"""
This script is responsible for evolving animations using fourier series
1) this uses FourierApproximator class to approximate fourier coefficients using numerical integration
2) in each frame the function is reconstructed until N components from step 1)
"""
import numpy as np

from core.fourier_drawer import FourierDrawer
from core.generate_points import get_points
from examples.bezier import get_bezier_curve, get_random_points

if __name__ == "__main__":
    ######## try one of those examples ########
    ###########################################################################################
    # Example 1: heart
    # points = []
    # for arg in np.arange(0, 2 * np.pi, .01):
    #     points.append(complex(16 * np.sin(arg) ** 3, 1 - (13 * np.cos(arg) - 5 * np.cos(2 * arg) - 2 * np.cos(3 * arg) - np.cos(4 * arg))))
    # points = np.array(points) * .05
    ###########################################################################################
    # Example 2: square
    # """
    # (-1,1)
    # (-1,-1)
    # (1,-1)
    # (1,1)
    # (-1,1)
    # """
    # points = [-1 + y * 1j for y in np.linspace(1, -1, 200)][:-1]
    # points += [y + -1 * 1j for y in np.linspace(-1, 1, 200)][:-1]
    # points += [1 + y * 1j for y in np.linspace(-1, 1, 200)][:-1]
    # points += [y + 1j for y in np.linspace(1, -1, 200)]
    # points = np.array(points) * .7
    ###########################################################################################
    # Example 3: a random bezier curve (closed-path)
    # ccw_sorted_points = get_random_points(n=7, scale=1)
    # ccw_sorted_points = np.vstack([ccw_sorted_points, ccw_sorted_points[0]])
    # x, y, _ = get_bezier_curve(ccw_sorted_points)
    # points = x - .5 + 1j * (y - .5)
    ###########################################################################################
    # Example 4: from an svg file
    points = get_points("data/fourier.pts")  # set max_components=300, num_components_step =1, this will write 300 frames with one new frequency component added with each frame
    ###########################################################################################

    FOUR = FourierDrawer(write_path="temp")
    FOUR.evolve(points, num_components_step=1, max_components=300)
