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
from numpy import array, concatenate, exp, insert, reshape, sqrt, zeros
import warnings

class MultiDDoubleSquaredExponential(cov.CovarianceFunction):
    # initialize class with initial hyperparameter theta
    def __init__(self, theta, X=None, Y=None):
        if (theta == None):
            # automatically provide initial theta if none is given
            sigmaf1 = (max(Y) - min(Y))/2.0
            l1 = array((np.max(X, axis=0) - np.min(X, axis=0))/2.0)
            sigmaf2 = (max(Y) - min(Y))/10.0
            l2 = array((np.min(X, axis=0) - np.min(X, axis=0))/10.0)
            theta1 = insert(l1, 0, sigmaf1)
            theta2 = insert(l2, 0, sigmaf2)
            theta = concatenate(theta1, theta2)
        cov.CovarianceFunction.__init__(self, theta)
        if (np.min(self.theta) <= 0.0):
            warnings.warn("Illegal hyperparameters in the " +
                          "initialization of MultiDDoubleSquaredExponential")


    # definition of the squared exponential covariance function
    def covfunc(self):
        nt = len(self.theta)
        sigmaf1 = self.theta[0]
        l1 = self.theta[1:nt/2]
        sigmaf2 = self.theta[nt/2]
        l2 = self.theta[nt/2 + 1:]
        xxl1 = np.sum(((self.x1 - self.x2)/l1)**2)
        xxl2 = np.sum(((self.x1 - self.x2)/l2)**2)
        covariance1 = sigmaf1**2 * exp(-xxl1/2.)
        covariance2 = sigmaf2**2 * exp(-xxl2/2.)
        covariance = covariance1 + covariance2
        return covariance


    # gradient of the squared exponential with respect to the hyperparameters
    # (d/dsigmaf,d/dl)k
    def gradcovfunc(self):
        nt = len(self.theta)
        sigmaf1 = self.theta[0]
        l1 = self.theta[1:nt/2]
        sigmaf2 = self.theta[nt/2]
        l2 = self.theta[nt/2 + 1:]
        grad = zeros(len(self.theta))
        xxl1 = np.sum(((self.x1 - self.x2)/l1)**2)
        xxl2 = np.sum(((self.x1 - self.x2)/l2)**2)
        grad[0] = 2 * sigmaf1 * exp(-xxl1/2.)
        grad[1:nt/2] = sigmaf1**2 * (self.x1[:] - self.x2[:])**2/l1[:]**3 * \
            exp(-xxl1/2.)
        grad[nt/2] = 2 * sigmaf2 * exp(-xxl2/2.)
        grad[nt/2 + 1:] = sigmaf2**2 * (self.x1[:] - self.x2[:])**2/l2[:]**3 * \
            exp(-xxl2/2.)
        return grad

    # derivative of the squared exponential with respect to x2
    def dcovfunc(self):
        raise RuntimeError("Derivative calculations are only implemented" + 
                           " for 1-dimensional inputs x.")

    # derivative of the squared exponential with respect to x1 and x2
    # dk/(dx1 dx2)
    def ddcovfunc(self):
        raise RuntimeError("Derivative calculations are only implemented" + 
                           " for 1-dimensional inputs x.")

    # second derivative of the squared exponential with respect to x2
    def d2covfunc(self):
        raise RuntimeError("Derivative calculations are only implemented" + 
                           " for 1-dimensional inputs x.")

    # second derivative of the squared exponential with respect to x1 and x2
    # d^4k/(dx1^2 dx2^2)
    def d2d2covfunc(self):
        raise RuntimeError("Derivative calculations are only implemented" + 
                           " for 1-dimensional inputs x.")

    # d^5/(dx1^2 dx2^3)
    def d2d3covfunc(self):
        raise RuntimeError("Derivative calculations are only implemented" + 
                           " for 1-dimensional inputs x.")

    # d^3k/(dx1 dx2^2)
    def dd2covfunc(self):
        raise RuntimeError("Derivative calculations are only implemented" + 
                           " for 1-dimensional inputs x.")

    # d^3k/dx2^3
    def d3covfunc(self):
        raise RuntimeError("Derivative calculations are only implemented" + 
                           " for 1-dimensional inputs x.")

    # d^6k/dx1^3dx2^3
    def d3d3covfunc(self):
        raise RuntimeError("Derivative calculations are only implemented" + 
                           " for 1-dimensional inputs x.")

    # d^4k/dx1dx2^3
    def dd3covfunc(self):
        raise RuntimeError("Derivative calculations are only implemented" + 
                           " for 1-dimensional inputs x.")

    # derivative of the gradient of the squared exponential with respect to x2
    def dgradcovfunc(self):
        raise RuntimeError("Derivative calculations are only implemented" + 
                           " for 1-dimensional inputs x.")

    # derivative of the gradient of the squared exponential with 
    # respect to x1 and x2
    # dk/(d1 d2)
    def ddgradcovfunc(self):
        raise RuntimeError("Derivative calculations are only implemented" + 
                           " for 1-dimensional inputs x.")



