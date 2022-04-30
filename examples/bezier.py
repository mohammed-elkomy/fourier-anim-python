import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt


# I found that on stackoverflow, but reformatted the code a bit to be more clear to me
# https://stackoverflow.com/questions/50731785/create-random-shape-contour-using-matplotlib/50732357


def bernstein(resolution, itr, interval):
    """
    direct application of the bezier curve (3rd order) https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Explicit_definition
    :param resolution: number of points to represent the resolution
    :param itr: the step or the point obtained by the function explained on wikipedia
    :param interval: the interval points [0,1] as a range
    :return: a term in the function explained on wikipedia, for both x and y
    """
    return binom(resolution, itr) * interval ** itr * (1. - interval) ** (resolution - itr)


def bezier(anchors, resolution):
    """
    direct application of the bezier curve (3rd order) https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Explicit_definition
    :param anchors: the anchor points of the bezier curve
    :param resolution: number of points to represent the resolution (how many points are generated between two successive points)
    :return: the curve points
    """
    num_points = len(anchors)
    interval = np.linspace(0, 1, num=resolution)  # this is t R: [0,1], spanning the curve
    curve = np.zeros((resolution, 2))
    for i in range(num_points):
        curve += np.outer(bernstein(num_points - 1, i, interval), anchors[i])
    return curve


class Segment:
    """
    A class to model a bezier segment of a 3rd order polynomail between to successive points
    the mathematical model is here: https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Explicit_definition
    """

    def __init__(self, p1, p2, angle1, angle2, **kw):
        """
        :param p1: the first point of the segment
        :param p2: the second point of the segment
        :param angle1: departure angle of the first point
        :param angle2: departure angle of the second point
        :param kw: extra parameters, mainly "numpoints", and "r"
        # numpoints: number of points to represent the resolution (how many points are generated between two successive points)
        # r: a random value r used to set the internal points
        """
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)  # number of points to represent the resolution (how many points are generated between two successive points)
        radius = kw.get("r", 0.3)  # a random value r used to set the internal points
        dist = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        self.radius = radius * dist
        self.anchors = np.zeros((4, 2))

        # outer points
        self.anchors[0, :] = self.p1[:]
        self.anchors[3, :] = self.p2[:]

        # inner points
        self.anchors[1, :] = self.p1 + np.array([self.radius * np.cos(self.angle1), self.radius * np.sin(self.angle1)])
        self.anchors[2, :] = self.p2 + np.array([self.radius * np.cos(self.angle2 + np.pi), self.radius * np.sin(self.angle2 + np.pi)])
        self.curve = bezier(self.anchors, self.numpoints)


def get_curve(points, **kw):
    """
    computes the curve in a form of segments, a segment is a 3rd order polynomial between every two successive points
    :param points: the sequence of points
    :param kw: extra parameters, mainly "numpoints", and "r" passed to Segment object
    :return: the segments and the curve points
    """
    segments = []
    for i in range(len(points) - 1):
        seg = Segment(points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], **kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve


def ccw_sort(points):
    """
    sorting points on counter clockwise order
    :param points: numpy points [n,2]
    :return: sorted numpy points [n,2], using the angle
    """
    shifted_points = points - np.mean(points, axis=0)
    slop_angle = np.arctan2(shifted_points[:, 0], shifted_points[:, 1])
    return points[np.argsort(slop_angle), :]


def positive_angle(ang):
    """
    :param ang: input angle
    :return: if the angle is negative, add a full cycle
    """
    return (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)


def get_bezier_curve(random_points, STEER=0.2, smooth=0, numpoints=100):
    """
    given an array of points *a*, create a curve through those points.
    :param random_points: counter clockwise sorted points
    :param STEER: is a number between 0 and 1 to steer the distance of control points.
    :param smooth: is a parameter which controls how "edgy" the curve is, edgy=0 is smoothest.
    :param numpoints: the curve resolution between between every two successive points
    :return: the bezier curve for the ccw_sorted_points
    """
    p = np.arctan(smooth) / np.pi + .5
    # random_points = ccw_sort(random_points)
    # ccw_sorted_points = np.append(ccw_sorted_points, np.atleast_2d(ccw_sorted_points[0, :]), axis=0)
    pairwise_differences = np.diff(random_points, axis=0)
    pairwise_angles = np.arctan2(pairwise_differences[:, 1], pairwise_differences[:, 0])

    pairwise_angles = positive_angle(pairwise_angles)
    ang1 = pairwise_angles
    ang2 = np.roll(pairwise_angles, 1)
    pairwise_angles = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    pairwise_angles = np.append(pairwise_angles, [pairwise_angles[0]])
    random_points = np.append(random_points, np.atleast_2d(pairwise_angles).T, axis=1)
    segments, curve = get_curve(random_points, r=STEER, numpoints=numpoints)
    x, y = curve.T
    return x, y, random_points


def get_random_points(n=5, scale=0.8, min_dst=None, recur=0):
    """ create n random points in the unit square, which are *mindst* apart, then scale them."""
    min_dst = min_dst or .7 / n
    random_points = np.random.rand(n, 2)  # zero centered
    ccw_sorted_points = ccw_sort(random_points)
    pairwise_differences = np.diff(ccw_sorted_points, axis=0)
    pairwise_distances = np.linalg.norm(pairwise_differences, axis=1)
    if np.all(pairwise_distances >= min_dst) or recur >= 200:
        return ccw_sorted_points * scale
    else:
        return get_random_points(n=n, scale=scale, min_dst=min_dst, recur=recur + 1)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    rad = 0.2
    edgy = 0.05

    for c in np.array([[0, 0], [0, 1], [1, 0], [1, 1]]):
        ccw_sorted_points = get_random_points(n=5, scale=1) + c
        x, y, _ = get_bezier_curve(ccw_sorted_points, STEER=rad, smooth=edgy)
        plt.plot(x, y)

    plt.show()
