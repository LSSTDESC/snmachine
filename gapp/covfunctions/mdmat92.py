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
from numpy import array, exp, insert, reshape, sqrt, zeros
import warnings


class MultiDMatern92(cov.CovarianceFunction):
    # initialize class with initial hyperparameter theta
    def __init__(self, theta, X=None, Y=None):
        if (theta == None):
            # automatically provide initial theta if none is given
            sigmaf = (max(Y) - min(Y))/2.0
            l = array((np.max(X, axis=0) - np.min(X, axis=0))/2.0)
            theta = insert(l, 0, sigmaf)
        cov.CovarianceFunction.__init__(self, theta)
        if (np.min(self.theta) <= 0.0):
            warnings.warn("Illegal hyperparameters in the" + 
                          " initialization of MultiDMatern92.")


    # definition of the matern92 covariance function
    def covfunc(self):
        sigmaf = self.theta[0]
        l = self.theta[1:]
        rl = sqrt(np.sum(((self.x1 - self.x2)/l)**2))
        erl = exp(-3.0 * rl)
        covariance = sigmaf**2 * (1 + 3.0 * rl + 27./7. * rl**2 +
                                  18./7. * rl**3 + 27./35. * rl**4) * erl
        return covariance

    # gradient of the matern92 with respect to the hyperparameters
    # (d/dsigmaf,d/dl)k
    def gradcovfunc(self):
        sigmaf = self.theta[0]
        l = self.theta[1:]
        grad = zeros(len(self.theta))
        r = self.x1 - self.x2
        rl = sqrt(np.sum((r/l)**2))
        erl = exp(-3.0 * rl)
        dk_dsigmaf = float(2 * sigmaf * (1 + 3.0 * rl + 27./7. * rl**2 +
                                  18./7. * rl**3 + 27./35. * rl**4) * erl)
        grad[0] = dk_dsigmaf
        grad[1:] = 9./35. * sigmaf**2 * r[:]**2/l[:]**3 * erl * \
                      (5. + 15. * rl + 18. * rl**2 + 9. * rl**3)
        return grad

    # derivative of the matern92 with respect to x2
    def dcovfunc(self):
        raise RuntimeError("Derivative calculations are only implemented" + 
                           " for 1-dimensional inputs x.")

    # derivative of the matern92 with respect to x1 and x2
    # dk/(dx1 dx2)
    def ddcovfunc(self):
        raise RuntimeError("Derivative calculations are only implemented" + 
                           " for 1-dimensional inputs x.")

    # second derivative of the matern92 with respect to x2
    def d2covfunc(self):
        raise RuntimeError("Derivative calculations are only implemented" + 
                           " for 1-dimensional inputs x.")

    # second derivative of the matern92 with respect to x1 and x2
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

    # derivative of the gradient of the matern92 with respect to x2
    def dgradcovfunc(self):
        raise RuntimeError("Derivative calculations are only implemented" + 
                           " for 1-dimensional inputs x.")

    # derivative of the gradient of the matern92 with respect to x1 and x2
    def ddgradcovfunc(self):
        raise RuntimeError("Derivative calculations are only implemented" + 
                           " for 1-dimensional inputs x.")


