"""
This holds a simple class for drawing our animations
1) uses opencv for 2d drawing
2) creates a canves of 2500x2500 with normalized coordiantes for graphing functions
3) uses scipy.fft to obtain the discrete fourier transform to be fed to arrow animation function  (check "animate" method here)
4) uses FourierApproximator class to generate the evolution animation (check "evolve" method here)
"""
import os
from math import atan2
from random import shuffle

import cmapy
import cv2
import numpy as np
from numpy import pi, cos, sin
from scipy.fft import fft

from core.fourier_numerical_approximator import FourierApproximator


class FourierDrawer:
    """
    A very simple rendering class for fourier animation, a 2D square screen limited by lower_limit and upper_limit
    this class can animate complex functions (assumed to be a closed loop),
    since fourier series operates on periodic functions, we can interpret a closed-loop as a periodic function with the angle repeating every 2*pi, for example a point on a circle at angle 0 is the same as any point at n*2*pi
    """

    def __init__(self, dpi=2500, lower_limit=-1, upper_limit=1, trace=True, write_path="temp"):
        """
        lower_limit and upper_limit represents a normalized coordinates that will be mapped to dpi resolution
        :param dpi: the resolution of the rendered image
        :param lower_limit: the lower left limit for the 2D drawing screen
        :param upper_limit: the upper right limit for the 2D drawing screen
        :param trace: caching the points with every frame (needed for vector animation)
        :param write_path: path to write images to
        """
        self.dpi = dpi
        self.dim_min = lower_limit
        self.dim_max = upper_limit
        self.diff = upper_limit - lower_limit
        self.trace = trace
        self.history = []
        self.source_zoom = .05
        self.dist_zoom = .2
        self.write_path = write_path
        if not os.path.exists(write_path):
            os.mkdir(write_path)

        cv2.namedWindow("Animation", cv2.WINDOW_NORMAL)  # the drawing window
        cv2.resizeWindow("Animation", 800, 800)  # my poor HD screen

    def un_normalized_coords(self, normalized):
        """
        from the normalized coordinates to the screen coordinates
        :param normalized: the point in normalized coordinates,
         if it's a np.ndarray (point_x, point_y)
         else it's a magnitude
        :return: scale and return the point based on the dpi
        """
        if type(normalized) is np.ndarray:  # a pair
            return (normalized - self.dim_min) / self.diff * self.dpi
        else:
            return normalized * self.dpi * .5  # (magnitude) value ranges from 0 to 1 to fit in screen

    def normalized_coords(self, un_normalized):
        """
        convert a  (point_x, point_y) to normalized coordinates
        :param un_normalized: scaled point to be normalized
        :return: normalized point
        """
        return (un_normalized - .5 * self.dpi) / self.dpi * self.diff

    @staticmethod
    def get_arrow_hooks(point1, point2):
        """
        for a line defined by 2 points, draw an arrow hook at point2
        :param point1: start point of a line
        :param point2: end point of a line
        :return: both hooks of => with point2 as head
        """
        # create the arrow hooks
        angle = atan2(point1[1] - point2[1], point1[0] - point2[0])  # angle in radians

        hook1 = (int(point2[0] + 8 * cos(angle + pi / 8)),
                 int(point2[1] + 8 * sin(angle + pi / 8)))

        hook2 = (int(point2[0] + 8 * cos(angle - pi / 8)),
                 int(point2[1] + 8 * sin(angle - pi / 8)))
        return hook1, hook2

    def draw_component(self, canvas, magnitude, phase_rad, old_shift, circle_color, text_color):
        """
        draw fourier component
        :param canvas: the drawing canvas to draw a component to
        :param magnitude: the magnitude of the component
        :param phase_rad: the phase for the component in radians
        :param old_shift: the last point position
        :param circle_color: BGR color of a circle
        :param text_color: BGR the color of the text
        :return:
        """
        # un normalize for drawing
        old_shift = self.un_normalized_coords(old_shift)
        magnitude = self.un_normalized_coords(magnitude)

        # 1) magnitude of a frequency component (a circle with radius of magnitude)
        thickness = int(np.log(100000 * magnitude / self.dpi)) + 1  # thickness based on strength
        cv2.circle(canvas, self.as_cv_tuple(old_shift), int(magnitude), color=circle_color, thickness=thickness, lineType=cv2.LINE_AA)

        # the new point, also the end of the arrow
        new_shift = np.array([(old_shift[0] + magnitude * cos(phase_rad)),
                              (old_shift[1] + magnitude * sin(phase_rad))])

        # 2) an arrow representing the phase shift of each component + time shift (a linear phase shift)
        thickness = max(int(np.log(10000 * magnitude / self.dpi)), 1)  # thickness based on strength
        cv2.line(canvas, self.as_cv_tuple(old_shift), self.as_cv_tuple(new_shift), color=text_color, thickness=thickness, lineType=cv2.LINE_AA)

        hook1, hook2 = self.get_arrow_hooks(old_shift, new_shift)
        cv2.line(canvas, hook1, self.as_cv_tuple(new_shift), color=text_color, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.line(canvas, hook2, self.as_cv_tuple(new_shift), color=text_color, thickness=thickness, lineType=cv2.LINE_AA)

        return self.normalized_coords(new_shift)

    @staticmethod
    def as_cv_tuple(point):
        """
        convert to opencv tuble
        :param point: floating np.array
        :return: convert to int tuple
        """
        return tuple(point.astype(int))

    def draw_components(self, magnitudes, phases, frequencies, normalized_time, colors):
        """
        draw the whole frame at a given point in time (this is used for the animation), the animation starts at normalized_time = 0 to normalized_time = 1
        :param magnitudes: the k components used to animate the drawing (more components = more circles = slower animation)
        :param phases: the phase shift of each component (to match the animation timing and draw the exact shape) in radian
        :param frequencies: the frequency of every component (integral multiple of fundamental frequency 1,2,3,4,5)
        :param normalized_time: from 0 to 1, start to the end of the animation video frames
        :param colors: list of consistent colors
        :return: the rendered frame

        magnitudes, phases, frequencies are of the same length
        """
        canvas = np.zeros((self.dpi, self.dpi, 3), dtype=np.uint8)  # init the frame

        current_shift = np.array((0, 0))  # starting at the center
        for color_idx, (frequency, magnitude, phase_rad) in enumerate(zip(frequencies, magnitudes, phases)):
            # for every frequency, magnitude, phase_rad draw the circle, the arrow based on the current shift + phase shift added plus the frequency
            angular_frequency = 2 * pi * frequency
            time_shift = angular_frequency * normalized_time  # at time normalized_time scale by the frequency: 2*pi*f*t
            current_shift = self.draw_component(canvas,
                                                magnitude,
                                                phase_rad + time_shift,
                                                current_shift, colors[color_idx], colors[color_idx * 2])

        if self.trace:
            # keep track of the points at time < normalized_time, cache them in history
            background = np.zeros_like(canvas)
            canvas = cv2.addWeighted(canvas, .5, background, .5, 0.0)  # make the arrows + circles half transparent
            self.history.append(
                self.as_cv_tuple(
                    self.un_normalized_coords(current_shift)  # the function value at time = normalized_time = f(t) = value
                ))

            # draw every point in history, to draw the 2d function in the red color
            for point1, point2 in zip(self.history[:-1], self.history[1:]):
                cv2.line(canvas, point1, point2, color=(0, 0, 255), thickness=10, lineType=cv2.LINE_AA)

            self.add_zoom_box(canvas)

        return canvas

    def add_zoom_box(self, canvas):
        """
        add the zoom box on the lower left, to zoom at the point being drawn
        :param canvas: the drawing canvas
        """
        # zoomed
        target_size = int(self.dist_zoom * self.dpi)
        source_zoom = self.source_zoom * self.dpi

        # make a crop
        left_zoomed = int(self.history[-1][0] - source_zoom)
        right_zoomed = int(self.history[-1][0] + source_zoom)
        top_zoomed = int(self.history[-1][1] - source_zoom)
        bottom_zoomed = int(self.history[-1][1] + source_zoom)

        # clip if it goes beyond
        left_zoomed = max(0, left_zoomed)
        top_zoomed = max(0, top_zoomed)
        right_zoomed = min(self.dpi, right_zoomed)
        bottom_zoomed = min(self.dpi, bottom_zoomed)

        # crop it
        zoom_box = canvas[top_zoomed:bottom_zoomed, left_zoomed:right_zoomed]

        # zero padding if the crop goes outside the boundaries
        hor_padding = int(source_zoom * 2 - zoom_box.shape[0])
        ver_padding = int(source_zoom * 2 - zoom_box.shape[1])

        zoom_box = np.pad(zoom_box,
                          [(ver_padding // 2, ver_padding // 2 + ver_padding % 2)
                              , (hor_padding // 2, hor_padding // 2 + hor_padding % 2),
                           (0, 0)])  # zero padding

        zoom_box = cv2.resize(zoom_box, (target_size, target_size))
        zoom_box[0:10, :] = (255, 255, 255)
        zoom_box[-11:-1, :] = (255, 255, 255)
        zoom_box[:, 0:10] = (255, 255, 255)
        zoom_box[:, -11:-1] = (255, 255, 255)
        canvas[-target_size:, -target_size:] = zoom_box  # place the zoom box at the bottom right

    def animate(self, time_signal, top_perc=1.0, cmap='hsv'):
        """
        EXAMPLE OUTPUT VIDEO IN heart.mp4
        This function draw the animated vector plot, where fourier components are vectors added cumulatively,
         to produce it we get the discrete fourier transform for the 2d complex function and we animate the components as time goes
         NOTE: discrete fourier transform takes n points to produce n frequency components,
         to make it efficient we only consider the top k components for reconstruction

        :param time_signal: signal in time domain (a sequence of points (complex for 2d))
        :param top_perc: percentage of the strongest frequency components to keep , this value ranges from 0 --> 1, 0 means take nothing, 1 takes all components
        :param cmap: any matplotlib cmap
        """

        freq_desc = fft(time_signal) / len(time_signal)  # complex array
        magnitudes = np.abs(freq_desc)
        phases = np.angle(freq_desc)
        time_steps = len(magnitudes)
        ###########################################################
        # dropping zero components for better color distribution
        tok_k_count = int(len(magnitudes) * top_perc)  # select the top k strongest components and neglecting the rest
        frequencies = np.sort(magnitudes.argsort()[-tok_k_count:][::-1])  # top k frequencies, frequency multiples actually
        magnitudes = magnitudes[frequencies]  # top k magnitudes
        phases = phases[frequencies]  # top k phases
        ############################################################
        # using the color map to color the components for a better visual experience
        colors = [cmapy.color(cmap, round(idx))
                  for idx in np.linspace(0, 255, len(frequencies) * 2)
                  ]  # linearly spaced colors
        shuffle(colors)
        ###########################################################
        # the animation simulation
        for time in range(time_steps + 100):  # extra 100 static frames in order to give some time for the viewer to see the full image in a video
            if time < time_steps:
                canvas = self.draw_components(magnitudes, phases, frequencies, time / time_steps, colors)
                cv2.imshow("Animation", canvas)
                if self.write_path:
                    cv2.imwrite(os.path.join(self.write_path, f"{time:05d}.png"), canvas)
                cv2.waitKey(10)
            else:
                cv2.imwrite(os.path.join(self.write_path, f"{time:05d}.png"), canvas)

        print("PRESS ANY KEY TO EXIT")
        cv2.waitKey()

    def evolve(self, time_signal, num_components_step=5, draw_original=False, cmap='hsv', max_components=400):
        """
        evolve animation uses the continuous fourier series approximation we made in core i.e FourierApproximator
        why we can't use fft from scipy?
        since fft is a discrete fast fourier transform it expects n input points to produce n frequency components, but what we need here to produce the evolution animation
        i.e. to draw the reconstructed function for the first component,
         then draw it for the first 2 components and so on, until we have a fair approximation of the function
        but using fft produces n frequency components for exactly n input points in time and they aren't guaranteed to reconstruct the function iteratively, you may try it :),
        discrete fourier transform when used to reconstruct the original input it only guarantees the original points when we used ALL OF THE FREQUENCY COMPONENTS

        the images will be saved to the write path for every frame rendered, the reconstruction will eventually diverge
        :param cmap: the color map during evolution
        :param draw_original: whether to draw the original curve
        :param num_components_step: number of components taken in each step
        :param time_signal: signal in time domain (a sequence of points (complex for 2d))
        :param max_components: max components to calculate
        :return:
        """
        f_approx = FourierApproximator(time_signal)
        # f_approx = BigFourierApproximator(time_signal)
        # f_approx can bring the top k frequency components and it also caches them for further calls

        orginal_function_canvas = np.zeros((self.dpi, self.dpi, 3), dtype=np.uint8)
        x_time = np.real(time_signal)
        y_time = np.imag(time_signal)

        # draw the original input function
        if draw_original:
            self.draw_to_canvas(orginal_function_canvas, x_time, y_time, color=(0, 255, 0))

        # draw the reconstructed function for 10 to 2000 components
        for i in range(0, max_components, num_components_step):
            temp_canvas = orginal_function_canvas.copy()
            restored_points = f_approx.restore_up_to(i)

            x_time = [point.real for point in restored_points]
            y_time = [point.imag for point in restored_points]
            self.draw_to_canvas(temp_canvas, x_time, y_time, color=cmapy.color(cmap, (i + 1) / max_components))
            cv2.putText(temp_canvas, f"Harmonics: {i}",
                        (int(temp_canvas.shape[0] * .75), int(temp_canvas.shape[1] * .05)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 7, cv2.LINE_AA)

            cv2.imshow("Animation", temp_canvas)
            if self.write_path:
                cv2.imwrite(os.path.join(self.write_path, f"{i:05d}.png"), temp_canvas)

            cv2.waitKey(1)

        print("PRESS ANY KEY TO EXIT")
        cv2.waitKey()

    def draw_to_canvas(self, canvas, x_time, y_time, color):
        for point in zip(x_time, y_time):
            point = self.un_normalized_coords(np.array(point))

            cv2.circle(canvas, self.as_cv_tuple(point), 5, color=color, thickness=-1)
