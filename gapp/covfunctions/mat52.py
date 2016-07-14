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


class Matern52(cov.CovarianceFunction):
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
                          " initialization of Matern52.")



    # definition of the matern52 covariance function
    def covfunc(self):
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = sqrt(np.sum((self.x1 - self.x2)**2))
        covariance = sigmaf**2 * (1 + sqrt(5.) * r/l + 5./3. * (r/l)**2) * \
            exp(-sqrt(5.) * r/l)
        return covariance

    # gradient of the matern52 with respect to the hyperparameters
    # (d/dsigmaf,d/dl)k
    def gradcovfunc(self):
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = sqrt(np.sum((self.x1 - self.x2)**2))
        dk_dsigmaf = float(2 * sigmaf * (1 + sqrt(5.) * r/l + 5./3. * 
                                         (r/l)**2) * exp(-sqrt(5.) * r/l))
        dk_dl = float(5/3. * sigmaf**2/l**3 * (1 + sqrt(5.) * r/l) * r**2 * 
                      exp(-sqrt(5.) * r/l))
        grad = array([dk_dsigmaf, dk_dl])
        return grad

    # derivative of the matern52 with respect to x2
    def dcovfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = sqrt((self.x1 - self.x2)**2)
        dcov = 5./3. * sigmaf**2/l**2 * exp(-sqrt(5.) * r/l) * \
            (self.x1 - self.x2) * (1. + sqrt(5.) * r/l)
        return float(dcov)

    # derivative of the matern52 with respect to x1 and x2
    # dk/(dx1 dx2)
    def ddcovfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = sqrt((self.x1 - self.x2)**2)
        dcov = 5./3 * sigmaf**2/l**2 * exp(-sqrt(5.) * r/l) * \
            (1. + sqrt(5.) * r/l - 5 * r**2/l**2)
        return float(dcov)

    # second derivative of the matern52 with respect to x2
    def d2covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = sqrt((self.x1 - self.x2)**2)
        dcov = 5./3 * sigmaf**2/l**2 * exp(-sqrt(5.) * r/l) * \
            (-1. - sqrt(5.) * r/l + 5 * r**2/l**2)
        return float(dcov)

    # second derivative of the matern52 with respect to x1 and x2
    # d^4k/(dx1^2 dx2^2)
    def d2d2covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = sqrt((self.x1 - self.x2)**2)
        dcov = 25./3 * sigmaf**2/l**4 * exp(-sqrt(5.) * r/l) * \
            (3. - 5 * sqrt(5.) * r/l + 5 * r**2/l**2)
        return float(dcov)

    # d^5/(dx1^2 dx2^3)
    def d2d3covfunc(self):
        raise RuntimeError("Error: Matern52 cannot reconstruct third " +
                           "(or higher) derivatives")


    # d^3k/(dx1 dx2^2)
    def dd2covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = sqrt((self.x1 - self.x2)**2)
        dcov = 25./3 * sigmaf**2/l**4 * exp(-sqrt(5.) * r/l) * \
            (self.x1 - self.x2) * (3. - sqrt(5.) * r/l)
        return float(dcov)

    # d^3k/dx2^3
    def d3covfunc(self):
        raise RuntimeError("Error: Matern52 cannot reconstruct third " +
                           "(or higher) derivatives")

    # d^6k/dx1^3dx2^3
    def d3d3covfunc(self):
        raise RuntimeError("Error: Matern52 cannot reconstruct third " +
                           "(or higher) derivatives")

    # d^4k/dx1dx2^3
    def dd3covfunc(self):
        raise RuntimeError("Error: Matern52 cannot reconstruct third " +
                           "(or higher) derivatives")

    # derivative of the gradient of the matern52 with respect to x2
    def dgradcovfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = float((self.x1 - self.x2)**2)
        dgrad_s = float(10. * sigmaf/l**2 * exp(-sqrt(5.) * r/l) * 
                        (self.x1 - self.x2) * (1./3. + sqrt(5.) * r/l))
        dgrad_l = float(5./3 * sigmaf**2/l**3 * exp(-sqrt(5.) * r/l) * 
                        (self.x1 - self.x2) * (-2. - 2 * sqrt(5.) * r/l + 
                                                5 * (r/l)**2))
        dgrad = array([dgrad_s, dgrad_l])
        return dgrad

    # derivative of the gradient of the matern52 with respect to x1 and x2
    def ddgradcovfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = float((self.x1 - self.x2)**2)
        ddgrad_s = float(10./3 * sigmaf/l**2 * exp(-sqrt(5.) * r/l) * 
                         (1. + sqrt(5.) * r/l - 5 * r**2/l**2))
        ddgrad_l = float(5./3 * sigmaf**2/l**3 * exp(-sqrt(5.) * r/l) * 
                         (-2. - 2 * sqrt(5.) * r/l + 25 * (r/l)**2 - 
                           5*sqrt(5.) * (r/l)**3))
        ddgrad = array([ddgrad_s, ddgrad_l])
        return ddgrad

