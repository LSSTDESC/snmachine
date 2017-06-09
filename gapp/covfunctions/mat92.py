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


class Matern92(cov.CovarianceFunction):
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
                          " initialization of Matern92.")





    # definition of the matern92 covariance function
    def covfunc(self):
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = sqrt(np.sum((self.x1 - self.x2)**2))
        rl = r/l
        erl = exp(-3.0 * rl)
        covariance = sigmaf**2 * (1 + 3.0 * rl + 27./7. * rl**2 +
                                  18./7. * rl**3 + 27./35. * rl**4) * erl
        return covariance

    # gradient of the matern92 with respect to the hyperparameters
    # (d/dsigmaf,d/dl)k
    def gradcovfunc(self):
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = sqrt(np.sum((self.x1 - self.x2)**2))
        rl = r/l
        erl = exp(-3.0 * rl)
        dk_dsigmaf = float(2 * sigmaf * (1 + 3.0 * rl + 27./7. * rl**2 +
                                  18./7. * rl**3 + 27./35. * rl**4) * erl)
        dk_dl = float(9./35. * sigmaf**2 * rl**2/l * erl *
                      (5. + 15. * rl + 18. * rl**2 + 9. * rl**3))
        grad = array([dk_dsigmaf, dk_dl])
        return grad

    # derivative of the matern92 with respect to x2
    def dcovfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = sqrt((self.x1 - self.x2)**2)
        rl = r/l
        erl = exp(-3.0 * rl)
        dcov = 9./35. * sigmaf**2/l**2 * erl * (self.x1 - self.x2) * \
            (5. + 15. * rl + 18. * rl**2 + 9. * rl**3)
        return float(dcov)

    # derivative of the matern92 with respect to x1 and x2
    # dk/(dx1 dx2)
    def ddcovfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = sqrt((self.x1 - self.x2)**2)
        rl = r/l
        erl = exp(-3.0 * rl)
        dcov = 9./35. * sigmaf**2/l**2 * erl * \
            (5. + 15. * rl + 9. * rl**2 - 18. * rl**3 - 27. * rl**4)
        return float(dcov)

    # second derivative of the matern92 with respect to x2
    def d2covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = sqrt((self.x1 - self.x2)**2)
        rl = r/l
        erl = exp(-3.0 * rl)
        dcov = 9./35. * sigmaf**2/l**2 * erl * \
            (-5. - 15. * rl - 9. * rl**2 + 18. * rl**3 + 27. * rl**4)
        return float(dcov)

    # second derivative of the matern92 with respect to x1 and x2
    # d^4k/(dx1^2 dx2^2)
    def d2d2covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = sqrt((self.x1 - self.x2)**2)
        rl = r/l
        erl = exp(-3.0 * rl)
        dcov = 243./35. * sigmaf**2/l**4 * erl * \
            (1. + 3. * rl - 3. * rl**2 - 18. * rl**3 + 9. * rl**4)
        return float(dcov)

    # d^5/(dx1^2 dx2^3)
    def d2d3covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = sqrt((self.x1 - self.x2)**2)
        rl = r/l
        erl = exp(-3.0 * rl)
        dcov = 729./35. * sigmaf**2/l**6 * erl * (self.x1 - self.x2) * \
            (5. + 15. * rl  - 30. * rl**2 + 9. * rl**3)
        return float(dcov)

    # d^3k/(dx1 dx2^2)
    def dd2covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = sqrt((self.x1 - self.x2)**2)
        rl = r/l
        erl = exp(-3.0 * rl)
        dcov = 243./35. * sigmaf**2/l**4 * erl * (self.x1 - self.x2) * \
            (1. + 3. * rl + 2. * rl**2 - 3. * rl**3)
        return float(dcov)

    # d^3k/dx2^3
    def d3covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = sqrt((self.x1 - self.x2)**2)
        rl = r/l
        erl = exp(-3.0 * rl)
        dcov = 243./35. * sigmaf**2/l**4 * erl * (self.x1 - self.x2) * \
            (-1. - 3. * rl - 2. * rl**2 + 3. * rl**3)
        return float(dcov)
        return float(dcov)

    # d^6k/dx1^3dx2^3
    def d3d3covfunc(self):
        if (self.multiD=='True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = sqrt((self.x1 - self.x2)**2)
        rl = r/l
        erl = exp(-3.0 * rl)
        dcov = 729./35. * sigmaf**2/l**6 * erl * \
            (5. + 15. * rl - 135. * rl**2 + 126. * rl**3 - 27 * rl**4)
        return float(dcov)

    # d^4k/dx1dx2^3
    def dd3covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = sqrt((self.x1 - self.x2)**2)
        rl = r/l
        erl = exp(-3.0 * rl)
        dcov = 243./35. * sigmaf**2/l**4 * erl * \
            (-1. - 3. * rl + 3. * rl**2 + 18. * rl**3 - 9. * rl**4)
        return float(dcov)

    # derivative of the gradient of the matern92 with respect to x2
    def dgradcovfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = float((self.x1 - self.x2)**2)
        rl = r/l
        erl = exp(-3.0 * rl)
        dgrad_s = float(18./35. * sigmaf/l**2 * erl * (self.x1 - self.x2) * 
                        (5. + 15. * rl + 18. * rl**2 + 9. * rl**3))
        dgrad_l = float(9./35. * sigmaf/l**3 * erl * (self.x1 - self.x2) *
                        (-10. - 30. * rl - 27. * rl**2 + 9. * rl**3 + 
                          27. * rl**4))
        dgrad = array([dgrad_s, dgrad_l])
        return dgrad

    # derivative of the gradient of the matern92 with respect to x1 and x2
    def ddgradcovfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        r = float((self.x1 - self.x2)**2)
        rl = r/l
        erl = exp(-3.0 * rl)
        ddgrad_s = float(18./35. * sigmaf/l**2 * erl * 
                         (5. + 15. * rl + 9. * rl**2 - 18. * rl**3 - 27. * rl**4))
        ddgrad_l = float(9./35. * sigmaf/l**3 * erl *
                         (-10. - 30. * rl + 9. * rl**2 + 117. * rl**3 - 
                           108. * rl**4 - 81. * rl**5))
        ddgrad = array([ddgrad_s, ddgrad_l])
        return ddgrad

