"""
This file holds the mathematical core for generating fourier series animation,
this involves
1) numerical integration
2) computing coefficients c_n
3) caching coefficients
4) restores the function up to the k-th coefficient
"""
from functools import partial

import numpy as np
from scipy import integrate


def complex_quadrature(func, a, b, ):
    """
    complex function integration (done numerically), taken from https://stackoverflow.com/questions/5965583/use-scipy-integrate-quad-to-integrate-complex-numbers
    :param func: any complex function (in our case the f function for fourier transform)
    :param a: the lower limit for our integration
    :param b: the upper limit for our integration
    :return: the integral evaluation value
    """

    def real_func(x):
        """
        evaluate the real part of a complex function at x
        :param x: the input points
        :return: the real part of func(x)
        """
        return np.real([func(x_) for x_ in x])

    def imag_func(x):
        """
        evaluate the imaginary part of a complex function at x
        :param x: the input points
        :return: the imaginary part of func(x)
        """
        return np.imag([func(x_) for x_ in x])

    real_integral = integrate.quadrature(real_func, a, b, maxiter=1500)  # integrate real
    imag_integral = integrate.quadrature(imag_func, a, b, maxiter=1500)  # integrate imaginary

    # integrate.quadrature()

    if real_integral[1:][0] > 1e-3:
        print("High integration error", real_integral[1:][0])

    if imag_integral[1:][0] > 1e-3:
        print("High integration error", imag_integral[1:][0])


    return real_integral[0] + 1j * imag_integral[0]  # the complex output


class FourierApproximator:
    """
    coefficient function for fourier transform, view it on any latex viewer https://latex.codecogs.com/eqneditor/editor.php
    c_n = \frac{1}{P} \int_{t_0}^{t_0 + P} f(t) \exp \left( \frac{-i 2\pi n t}{P}\right) \mathrm{d}t
    P is the period, since we are given points to approximate using fourier we can simply set P = 1
    """

    def __init__(self, points):
        self.P = 1  # any value will work

        self.points = points  # samples from the complex 2d function
        self.cache = {}  # caching fourier coefficients as we go

    def fourier_integral_sample(self, t, n):
        """
        in order to compute c_n numerically (numerical integration) we need to get samples of the function being integrated at different points t
        this expression
         \frac{1}{P}  f(t) \exp \left( \frac{-i 2\pi n t}{P}\right)
        :param t: for point t in time
        :param n: for coefficient k
        :return: a sample d_n(t)
        """
        mapped_index = t / self.P  # for point t in time, get the index of the nearest point of self.points
        f_t = self.points[int(len(self.points) * mapped_index)]

        exp_part = np.exp(-2j * np.pi * n * t / self.P)
        return (1 / self.P) * f_t * exp_part

    def fourier_series(self, c_n_list, t, ):
        """
        this applies the fourier series reconstruction by computing 
        S_N(t) = \sum_{n=-N}^N c_n \exp \left( \frac{i 2\pi n t}{P}\right)
        using c_n fourier coefficients
        :param c_n_list:  a list of top 2*k + 1 fourier exponential coefficients (2*k complex conjugates) (returned from get_coefficients)
        :param t: at specific point t get the value of the function
        :return: the value of the function resembled by points self.points, simply this is the fourier approximation for f(t) using top k fourier coefficients
        """
        return sum(c_n * np.exp(2j * np.pi * n * t / self.P)
                   for n, c_n in c_n_list)

    def get_coefficients(self, k):
        """
        get top k fourier coefficients for the discrete function have self.points samples
        :param k: the number of components, more k = better approximation, in general :)
        :return: a list of top 2*k + 1 fourier exponential coefficients (2*k complex conjugates)
        """
        cn = []
        for i in range(-k, k + 1):
            if i in self.cache:
                c_n = self.cache[i]  # don't recompute the same coefficient, get if from the cache
            else:

                # c_n = \frac{1}{P} \int_{t_0}^{t_0 + P} f(t) \exp \left( \frac{-i 2\pi n t}{P}\right) \mathrm{d}t
                c_n = complex_quadrature(partial(self.fourier_integral_sample, n=i),
                                         0,
                                         self.P)
                self.cache[i] = c_n
            cn.append((i, c_n))
        return cn

    def restore_up_to(self, k):
        """
        we can use fourier coefficients to restore the signal using first k components
        :param k: the number of components, more k = better approximation, theoretically, but here it diverges because of numerical stability issues
        :return: the points reconstructed from fourier coefficients computed using the self.points
        """
        cn = self.get_coefficients(k)  # get top k coefficients
        return [
            self.fourier_series(cn, t)
            for t in np.linspace(0, self.P, len(self.points))
        ]
