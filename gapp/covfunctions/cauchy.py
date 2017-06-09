"""
    GaPP: Gaussian Processes in Python
    Copyright (C) 2012, 2013  Marina Seikel
    University of Cape Town
    University of Western Cape
    marina [at] jorrit.de

    This file is part of GaPP.

    GaPP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GaPP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""




import cov
import numpy as np
from numpy import array, exp, reshape, sqrt
import warnings


class Cauchy(cov.CovarianceFunction):
    # initialize class with initial hyperparameter theta
    def __init__(self, theta, X=None, Y=None):
        if (theta == None):
            # automatically provide initial theta if none is given
            sigmaf = (max(Y) - min(Y))/2.0
            l = np.min(np.max(X, axis=0) - np.min(X, axis=0))/2.0
            theta = [sigmaf, l]
        cov.CovarianceFunction.__init__(self, theta)
        if (self.theta[0] <= 0.0 or self.theta[1] < 0.0):
            warnings.warn("Illegal hyperparameters in the" + 
                          " initialization of Cauchy.")

    # definition of the cauchy covariance function
    def covfunc(self):
        sigmaf = self.theta[0]
        l = self.theta[1]
        r2 = float(np.sum((self.x1 - self.x2)**2))
        a = l**2 + r2
        covariance = sigmaf**2 * l/a
        return covariance

    # gradient of the cauchy with respect to the hyperparameters
    # (d/dsigmaf,d/dl)k
    def gradcovfunc(self):
        sigmaf = self.theta[0]
        l = self.theta[1]
        r2 = float(np.sum((self.x1 - self.x2)**2))
        a = l**2 + r2
        dk_dsigmaf = float(2 * sigmaf * l/a)
        dk_dl = float(sigmaf**2 * (r2 - l**2)/a**2)
        grad = array([dk_dsigmaf, dk_dl])
        return grad

    # derivative of the cauchy with respect to x2
    def dcovfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r2 = (self.x1 - self.x2)**2
        a = l**2 + r2
        dcov = 2 * sigmaf**2 * l * (self.x1 - self.x2)/a**2
        return float(dcov)

    # derivative of the cauchy with respect to x1 and x2
    # dk/(dx1 dx2)
    def ddcovfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r2 = (self.x1 - self.x2)**2
        a = l**2 + r2
        dcov = 2 * sigmaf**2 * l * (l**2 - 3 * r2)/a**3
        return float(dcov)

    # second derivative of the cauchy with respect to x2
    def d2covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r2 = (self.x1 - self.x2)**2
        a = l**2 + r2
        dcov = - 2 * sigmaf**2 * l *(l**2 - 3 * r2)/a**3
        return float(dcov)

    # second derivative of the cauchy with respect to x1 and x2
    # d^4k/(dx1^2 dx2^2)
    def d2d2covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r2 = (self.x1 - self.x2)**2
        a = l**2 + r2
        dcov = 24 * sigmaf**2 * l * (l**4 - 10 * l**2 * r2 + 5 * r2**2)/a**5
        return float(dcov)

    # d^5/(dx1^2 dx2^3)
    def d2d3covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r2 = (self.x1 - self.x2)**2
        a = l**2 + r2
        dcov = 240 * sigmaf**2 * l * (3 * l**4 - 10 * l**2 * r2 + 3 * r2**2) *\
            (self.x1 - self.x2)/a**6
        return float(dcov)

    # d^3k/(dx1 dx2^2)
    def dd2covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r2 = (self.x1 - self.x2)**2
        a = l**2 + r2
        dcov = 24 * sigmaf**2 * l * (l**2 - r2) * (self.x1 - self.x2)/a**4
        return float(dcov)

    # d^3k/dx2^3
    def d3covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r2 = (self.x1 - self.x2)**2
        a = l**2 + r2
        dcov = 24 * sigmaf**2 * l * (-l**2 + r2) * (self.x1 - self.x2)/a**4
        return float(dcov)

    # d^6k/dx1^3dx2^3
    def d3d3covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r2 = (self.x1 - self.x2)**2
        a = l**2 + r2
        dcov = 720 * sigmaf**2 * l * (l**6 - 21 * l**4 * r2 + 
                                      35 * l**2 * r2**2 - 7 * r2**3)/a**7
        return dcov

    # d^4k/dx1dx2^3
    def dd3covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r2 = (self.x1 - self.x2)**2
        a = l**2 + r2
        dcov = -24 * sigmaf**2 * l * (l**4 - 10 * l**2 * r2 + 5 * r2**2)/a**5
        return dcov

    # derivative of the gradient of the cauchy with respect to x2
    def dgradcovfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r2 = float((self.x1 - self.x2)**2)
        a = l**2 + r2
        dgrad_s = float(4 * sigmaf * l * (self.x1 - self.x2)/a**2)
        dgrad_l = float(2 * sigmaf**2 * (-3 * l**2 + r2) * 
                        (self.x1 - self.x2)/a**3)
        dgrad = array([dgrad_s, dgrad_l])
        return dgrad

    # derivative of the gradient of the cauchy with 
    # respect to x1 and x2
    # dk/(d1 d2)
    def ddgradcovfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r2 = float((self.x1 - self.x2)**2)
        a = l**2 + r2
        ddgrad_s = float(4 * sigmaf * l * (l**2 - 3 * r2)/a**3)
        ddgrad_l = float(-6 * sigmaf**2 * (l**4 - 6 * l**2 * r2 + r2**2)/a**4)
        ddgrad = array([ddgrad_s, ddgrad_l])
        return ddgrad

