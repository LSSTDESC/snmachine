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




from . import cov
import numpy as np
from numpy import array, exp, reshape, sqrt
import warnings

class DoubleSquaredExponential(cov.CovarianceFunction):
    # initialize class with initial hyperparameter theta
    def __init__(self, theta, X=None, Y=None):
        if (theta == None):
            # automatically provide initial theta if none is given
            sigmaf1 = (max(Y) - min(Y))/2.0
            l1 = np.min(np.max(X, axis=0) - np.min(X, axis=0))/2.0
            sigmaf2 = (max(Y) - min(Y))/10.0
            l2 = np.min(np.max(X, axis=0) - np.min(X, axis=0))/10.0
            theta = [sigmaf1, l1, sigmaf2, l2]
        cov.CovarianceFunction.__init__(self, theta)
        if (np.min(self.theta) < 0.0):
            warnings.warn("Illegal hyperparameters in the " +
                          "initialization of DoubleSquaredExponential.")



    # definition of the squared exponential covariance function
    def covfunc(self):
        sigmaf1 = self.theta[0]
        l1 = self.theta[1]
        sigmaf2 = self.theta[2]
        l2 = self.theta[3]
        xxl1 = np.sum(((self.x1 - self.x2)/l1)**2)
        xxl2 = np.sum(((self.x1 - self.x2)/l2)**2)
        covariance1 = sigmaf1**2 * exp(-xxl1/2.)
        covariance2 = sigmaf2**2 * exp(-xxl2/2.)
        covariance = covariance1 + covariance2
        return covariance


    # gradient of the squared exponential with respect to the hyperparameters
    # (d/dsigmaf,d/dl)k
    def gradcovfunc(self):
        sigmaf1 = self.theta[0]
        l1 = self.theta[1]
        sigmaf2 = self.theta[2]
        l2 = self.theta[3]
        xxl1 = np.sum(((self.x1 - self.x2)/l1)**2)
        xxl2 = np.sum(((self.x1 - self.x2)/l2)**2)
        dk_dsigmaf1 = 2 * sigmaf1 * exp(-xxl1/2.)
        dk_dl1 = sigmaf1**2/l1 * xxl1 * exp(-xxl1)
        dk_dsigmaf2 = 2 * sigmaf2 * exp(-xxl2/2.)
        dk_dl2 = sigmaf2**2/l2 * xxl2 * exp(-xxl2/2.)
        grad = array([dk_dsigmaf1, dk_dl1, dk_dsigmaf2, dk_dl2])
        return grad

    # derivative of the squared exponential with respect to x2
    def dcovfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf1 = self.theta[0]
        l1 = self.theta[1]
        sigmaf2 = self.theta[2]
        l2 = self.theta[3]
        xxl1 = ((self.x1 - self.x2)/l1)**2
        xxl2 = ((self.x1 - self.x2)/l2)**2
        dcov1 = (sigmaf1/l1)**2 * exp(-xxl1/2.) * (self.x1 - self.x2)
        dcov2 = (sigmaf2/l2)**2 * exp(-xxl2/2.) * (self.x1 - self.x2)
        dcov = dcov1 + dcov2
        return float(dcov)

    # derivative of the squared exponential with respect to x1 and x2
    # dk/(dx1 dx2)
    def ddcovfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf1 = self.theta[0]
        l1 = self.theta[1]
        sigmaf2 = self.theta[2]
        l2 = self.theta[3]
        xxl1 = ((self.x1 - self.x2)/l1)**2
        xxl2 = ((self.x1 - self.x2)/l2)**2
        dcov1 = (sigmaf1/l2)**2 * exp(-xxl1/2.) * (1 - xxl1)
        dcov2 = (sigmaf1/l2)**2 * exp(-xxl2/2.) * (1 - xxl2)
        dcov = dcov1 + dcov2
        return float(dcov)

    # second derivative of the squared exponential with respect to x2
    def d2covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf1 = self.theta[0]
        l1 = self.theta[1]
        sigmaf2 = self.theta[2]
        l2 = self.theta[3]
        xxl1 = ((self.x1 - self.x2)/l1)**2
        xxl2 = ((self.x1 - self.x2)/l2)**2
        dcov1 = (sigmaf1/l1)**2 * exp(-xxl1/2.) * (xxl1 - 1.)
        dcov2 = (sigmaf2/l2)**2 * exp(-xxl2/2.) * (xxl2 - 1.)
        dcov = dcov1 + dcov2
        return float(dcov)

    # second derivative of the squared exponential with respect to x1 and x2
    # d^4k/(dx1^2 dx2^2)
    def d2d2covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf1 = self.theta[0]
        l1 = self.theta[1]
        sigmaf2 = self.theta[2]
        l2 = self.theta[3]
        xxl1 = ((self.x1 - self.x2)/l1)**2
        xxl2 = ((self.x1 - self.x2)/l2)**2
        dcov1 = sigmaf1**2/l1**4 * exp(-xxl1/2.) * (3. - 6 * xxl1 + xxl1**2)
        dcov2 = sigmaf2**2/l2**4 * exp(-xxl2/2.) * (3. - 6 * xxl2 + xxl2**2)
        dcov = dcov1 + dcov2
        return float(dcov)

    # d^5/(dx1^2 dx2^3)
    def d2d3covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf1 = self.theta[0]
        l1 = self.theta[1]
        sigmaf2 = self.theta[2]
        l2 = self.theta[3]
        xxl1 = ((self.x1 - self.x2)/l1)**2
        xxl2 = ((self.x1 - self.x2)/l2)**2
        dcov1 = sigmaf1**2/l1**6 * exp(-xxl1/2.) * \
            (15. - 10 * xxl1 + xxl1**2) * (self.x1 - self.x2)
        dcov2 = sigmaf2**2/l2**6 * exp(-xxl2/2.) * \
            (15. - 10 * xxl2 + xxl2**2) * (self.x1 - self.x2)
        dcov = dcov1 + dcov2
        return float(dcov)

    # d^3k/(dx1 dx2^2)
    def dd2covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf1 = self.theta[0]
        l1 = self.theta[1]
        sigmaf2 = self.theta[2]
        l2 = self.theta[3]
        xxl1 = ((self.x1 - self.x2)/l1)**2
        xxl2 = ((self.x1 - self.x2)/l2)**2
        dcov1 = -sigmaf1**2/l1**4 * exp(-xxl1/2.) * (xxl1 - 3.) * \
            (self.x1 - self.x2)
        dcov2 = -sigmaf2**2/l2**4 * exp(-xxl2/2.) * (xxl2 - 3.) * \
            (self.x1 - self.x2)
        dcov = dcov1 + dcov2
        return float(dcov)

    # d^3k/dx2^3
    def d3covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf1 = self.theta[0]
        l1 = self.theta[1]
        sigmaf2 = self.theta[2]
        l2 = self.theta[3]
        xxl1 = ((self.x1 - self.x2)/l1)**2
        xxl2 = ((self.x1 - self.x2)/l2)**2
        dcov1 = sigmaf1**2/l1**4 * exp(-xxl1/2.) * (xxl1 - 3.) * \
            (self.x1 - self.x2)
        dcov2 = sigmaf2**2/l2**4 * exp(-xxl2/2.) * (xxl2 - 3.) * \
            (self.x1 - self.x2)
        dcov = dcov1 + dcov2
        return float(dcov)

    # d^6k/dx1^3dx2^3
    def d3d3covfunc(self):
        if (self.multiD=='True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf1 = self.theta[0]
        l1 = self.theta[1]
        sigmaf2 = self.theta[2]
        l2 = self.theta[3]
        xxl1 = ((self.x1 - self.x2)/l1)**2
        xxl2 = ((self.x1 - self.x2)/l2)**2
        dcov1 = sigmaf1**2/l1**6 * exp(-xxl1/2.) * (15. - 45 * xxl1 +
                                                    15 * xxl1**2 - xxl1**3)
        dcov2 = sigmaf2**2/l2**6 * exp(-xxl2/2.) * (15. - 45 * xxl2 + 
                                                    15 * xxl2**2 - xxl2**3)
        dcov = dcov1 + dcov2
        return dcov

    # d^4k/dx1dx2^3
    def dd3covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf1 = self.theta[0]
        l1 = self.theta[1]
        sigmaf2 = self.theta[2]
        l2 = self.theta[3]
        xxl1 = ((self.x1 - self.x2)/l1)**2
        xxl2 = ((self.x1 - self.x2)/l2)**2
        dcov1 = sigmaf1**2/l1**4 * exp(-xxl1/2.) * (-3. + 6 * xxl1 - xxl1**2)
        dcov2 = sigmaf2**2/l2**4 * exp(-xxl2/2.) * (-3. + 6 * xxl2 - xxl2**2)
        dcov = dcov1 + dcov2
        return dcov

    # derivative of the gradient of the squared exponential with respect to x2
    def dgradcovfunc(self):
        if (self.multiD=='True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf1 = self.theta[0]
        l1 = self.theta[1]
        sigmaf2 = self.theta[2]
        l2 = self.theta[3]
        xxl1 = float(((self.x1 - self.x2)/l1)**2)
        xxl2 = float(((self.x1 - self.x2)/l2)**2)
        dgrad_s1 = float(2 * sigmaf1/l1**2 * exp(-xxl1/2.) * 
                         (self.x1 - self.x2))
        dgrad_l1 = sigmaf1**2/l1**3 * exp(-xxl1/2.) * (self.x1 - self.x2) * \
            (xxl1 - 2)
        dgrad_s2 = float(2 * sigmaf2/l2**2 * exp(-xxl2/2.) * 
                         (self.x1 - self.x2))
        dgrad_l = sigmaf2**2/l2**3 * exp(-xxl2/2.) * (self.x1 - self.x2) * \
            (xxl2 - 2)
        dgrad = array([dgrad_s1, dgrad_l1, dgrad_s2, dgrad_l2])
        return dgrad

    # derivative of the gradient of the squared exponential with 
    # respect to x1 and x2
    # dk/(d1 d2)
    def ddgradcovfunc(self):
        if (self.multiD=='True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf1 = self.theta[0]
        l1 = self.theta[1]
        sigmaf2 = self.theta[2]
        l2 = self.theta[3]
        xxl1 = float(((self.x1 - self.x2)/l1)**2)
        xxl2 = float(((self.x1 - self.x2)/l2)**2)
        ddgrad_s1 = 2 * sigmaf1/l1**2 * exp(-xxl1/2.) * (1 - xxl1)
        ddgrad_l1 = sigmaf1**2/l1**3 * exp(-xxl1/2.) * (-2 + 5 * xxl1 - xxl1**2)
        ddgrad_s2 = 2 * sigmaf2/l2**2 * exp(-xxl2/2.) * (1 - xxl2)
        ddgrad_l2 = sigmaf2**2/l2**3 * exp(-xxl2/2.) * (-2 + 5 * xxl2 - xxl2**2)
        ddgrad = array([ddgrad_s1, ddgrad_l1, ddgrad_s2, ddgrad_l2])
        return ddgrad


