"""
This script is responsible for
1) reading a svg file with paths,
2) those paths are represented as a complex function and saved to the disk with the help of joblib
3) the drawing code for arrow animation and evolution can work with points returned by "get_points" funtion here
"""
import random
from xml.dom import minidom

import cv2
import joblib
import numpy as np
from svg.path import parse_path


def get_point_at(path, distance, scale, offset):
    pos = path.point(distance)
    pos += offset
    pos *= scale
    return pos.real, pos.imag


def points_from_path(path, density, scale, offset):
    step = int(path.length() * density)
    last_step = step - 1

    if last_step == 0:
        yield get_point_at(path, 0, scale, offset)
        return

    for distance in range(step):
        yield get_point_at(
            path, distance / last_step, scale, offset)


def points_from_doc(doc, density=5, scale=1, offset=0):
    offset = offset[0] + offset[1] * 1j
    points = []
    for element in doc.getElementsByTagName("path"):
        for path in parse_path(element.getAttribute("d")):
            points.extend(points_from_path(
                path, density, scale, offset))

    return points


def generate_points_from_svg(svg_file_path):
    # read the SVG file
    doc = minidom.parse(svg_file_path)
    points = points_from_doc(doc, density=1, scale=5, offset=(0, 5))
    doc.unlink()

    points = np.array(points)
    min_p, max_p = points.min(axis=0), points.max(axis=0)
    points = (points - min_p) / (max_p - min_p)
    points = points[:, ::-1]
    points[:, 0] = points[:, 0] * .9 + .05
    points[:, 1] = points[:, 1] * .7 + .15
    joblib.dump((points, 1.0), svg_file_path.replace(".svg", ".pts"), compress=5)


def animate_points(pts_path):
    cv2.namedWindow("Animation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Animation", 800, 800)

    path, aspect = joblib.load(pts_path)

    dim = 1000
    draw = np.zeros((int(aspect * dim), dim, 3)).astype(np.uint8)
    path = (path * (aspect * dim, dim)).astype(int)
    for p in path:
        color = (random.choices(range(256), k=3))

        cv2.circle(draw, tuple(p[::-1]), 5, color=color, thickness=-1)

        cv2.imshow("Animation", draw)
        cv2.waitKey(1)


def get_points(pts_path):
    path, aspect = joblib.load(pts_path)
    path = path * 2 - 1
    path = (path * (1, 1 / aspect))
    return 1j * path[:, 0] + path[:, 1]

