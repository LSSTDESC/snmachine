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
from numpy import array, exp, log, reshape, sqrt
import warnings


class RationalQuadratic(cov.CovarianceFunction):
    # initialize class with initial hyperparameter theta
    def __init__(self, theta, X=None, Y=None):
        if (theta == None): 
            # automatically provide initial theta if none is given
            sigmaf = (max(Y) - min(Y))/2.0
            l = np.min(np.max(X, axis=0) - np.min(X, axis=0))/2.0
            alpha = 1.0
            theta = [sigmaf, l, alpha]
        cov.CovarianceFunction.__init__(self, theta)
        if (np.min(self.theta) <= 0.0):
            warnings.warn("Illegal hyperparameters in the" + 
                          " initialization of RationalQuadratic")


    # definition of the rational quadratic covariance function
    def covfunc(self):
        sigmaf = self.theta[0]
        l = self.theta[1]
        alpha = self.theta[2]
        rl2 = np.sum(((self.x1 - self.x2)/l)**2)
        covariance = sigmaf**2 / (1.0 + rl2/(2.0 * alpha))**alpha
        return covariance


    # gradient of the rational quadratic with respect to the hyperparameters
    # (d/dsigmaf,d/dl)k
    def gradcovfunc(self):
        sigmaf = self.theta[0]
        l = self.theta[1]
        alpha = self.theta[2]
        r2 = np.sum((self.x1 - self.x2)**2)
        rl2 = r2/l**2
        dk_dsigmaf = 2 * sigmaf / (1.0 + rl2/(2.0 * alpha))**alpha
        dk_dl = sigmaf**2/l**3 * r2 / (1. + rl2/(2. * alpha))**(1. + alpha)
        dk_dalpha = sigmaf**2/(alpha * l**2) * 2.**alpha * \
            (alpha * l**2 / (2. * alpha * l**2 + r2))**(1. + alpha) * \
            (r2 - (2. * alpha * l**2 + r2) * log(1. + rl2/(2. * alpha)))
        grad = array([dk_dsigmaf, dk_dl, dk_dalpha])
        return grad

    # derivative of the rational quadratic with respect to x2
    def dcovfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        alpha = self.theta[2]
        rl2 = np.sum(((self.x1 - self.x2)/l)**2)
        dcov = (sigmaf/l)**2 / (1. + rl2/(2. * alpha))**(1. + alpha) * \
            (self.x1 - self.x2) 
        return float(dcov)

    # derivative of the rational quadratic with respect to x1 and x2
    # dk/(dx1 dx2)
    def ddcovfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        alpha = self.theta[2]
        r2 = np.sum((self.x1 - self.x2)**2)
        rl2 = r2/l**2
        dcov = sigmaf**2 * 2. * alpha * (2. * alpha * (l**2 - r2) - r2)/ \
            ((2. * alpha * l**2 + r2)**2 * (1. + rl2/(2. * alpha))**alpha)
        return float(dcov)

    # second derivative of the rational quadratic with respect to x2
    def d2covfunc(self):
        if (self.multiD=='True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        alpha = self.theta[2]
        r2 = np.sum((self.x1 - self.x2)**2)
        rl2 = r2/l**2
        dcov = - sigmaf**2 * 2. * alpha * (2. * alpha * (l**2 - r2) - r2)/ \
            ((2. * alpha * l**2 + r2)**2 * (1. + rl2/(2. * alpha))**alpha)
        return float(dcov)

    # second derivative of the rational quadratic with respect to x1 and x2
    # d^4k/(dx1^2 dx2^2)
    def d2d2covfunc(self):
        if (self.multiD=='True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        alpha = self.theta[2]
        r2 = np.sum((self.x1 - self.x2)**2)
        rl2 = r2/l**2
        t1 = 4. * alpha**3 * (3. * l**4 - 6. * l**2 * r2 + r2**2)
        t2 = 12 * alpha**2 * (l**4 - 5. * l**2 * r2 + r2**2)
        t3 = alpha * (- 36. * l**2  + 11. * r2) * r2 + 3. * r2**2
        dcov = sigmaf**2 * 4. * alpha * (t1 + t2 + t3) / \
            ((2. * alpha * l**2 + r2)**4 * (1. + rl2/(2. * alpha))**alpha)
        return float(dcov)

    # d^5/(dx1^2 dx2^3)
    def d2d3covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        alpha = self.theta[2]
        r2 = np.sum((self.x1 - self.x2)**2)
        rl2 = r2/l**2
        t1 = 4. * alpha**4 * (15. * l**4 - 10. * l**2 * r2 + r2**2)
        t2 = 20. * alpha**3 * (9. * l**4 - 9. * l**2 * r2 + r2**2)
        t3 = 5. * alpha**2 * (24. * l**4 - 52. * l**2 * r2 + 7. * r2**2)
        t4 = -5. * alpha * (24. * l**2  - 5. * r2) * r2 + 6. * r2**2
        dcov = sigmaf**2 * 8. * alpha * (t1 + t2 + t3 + t4) / \
            ((2. * alpha * l**2 + r2)**5 * (1. + rl2/(2. * alpha))**alpha) * \
            (self.x1 - self.x2)
        return float(dcov)

    # d^3k/(dx1 dx2^2)
    def dd2covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        alpha = self.theta[2]
        r2 = np.sum((self.x1 - self.x2)**2)
        rl2 = r2/l**2
        dcov =  sigmaf**2 * 4. * alpha * (1. + alpha) * (self.x1 - self.x2) * \
            (2. * alpha * (3. * l**2 - r2) - r2)/ \
            ((2. * alpha * l**2 + r2)**3 * (1. + rl2/(2. * alpha))**alpha)
        return float(dcov)

    # d^3k/dx2^3
    def d3covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        alpha = self.theta[2]
        r2 = np.sum((self.x1 - self.x2)**2)
        rl2 = r2/l**2
        dcov = sigmaf**2 * 4. * alpha * (1. + alpha) * (self.x1 - self.x2) * \
            (2. * alpha * (-3. * l**2 + r2) + r2)/ \
            ((2. * alpha * l**2 + r2)**3 * (1. + rl2/(2. * alpha))**alpha)
        return float(dcov)

    # d^6k/dx1^3dx2^3
    def d3d3covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        alpha = self.theta[2]
        r2 = np.sum((self.x1 - self.x2)**2)
        rl2 = r2/l**2
        t1 = 8. * alpha**5 * (15. * l**6 - 45. * l**4 * r2 + 
                              15. * l**2 * r2**2 - r2**3)
        t2 = 60. * alpha**4 * (6. * l**6 - 33. * l**4 * r2 + 
                              14. * l**2 * r2**2 - r2**3)
        t3 = 10. * alpha**3 * (24. * l**6 - 342. * l**4 * r2 + 
                              213. * l**2 * r2**2 - 17. * r2**3)
        t4 = -15. * alpha**2 * (120. * l**4 - 154. * l**2 * r2 + 
                              15. * r2**2) * r2
        t5 = alpha * (900. * l**2 - 137. * r2) * r2**2 - 30. * r2**3
        dcov = sigmaf**2 * 8. * alpha * (t1 + t2 + t3 + t4 + t5) / \
            ((2. * alpha * l**2 + r2)**6 * (1. + rl2/(2. * alpha))**alpha)
        return dcov

    # d^4k/dx1dx2^3
    def dd3covfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        alpha = self.theta[2]
        r2 = np.sum((self.x1 - self.x2)**2)
        rl2 = r2/l**2
        t1 = 4. * alpha**3 * (3. * l**4 - 6. * l**2 * r2 + r2**2)
        t2 = 12 * alpha**2 * (l**4 - 5. * l**2 * r2 + r2**2)
        t3 = alpha * (- 36. * l**2  + 11. * r2) * r2 + 3. * r2**2
        dcov = - sigmaf**2 * 4. * alpha * (t1 + t2 + t3) / \
            ((2. * alpha * l**2 + r2)**4 * (1. + rl2/(2. * alpha))**alpha)
        return dcov

    # derivative of the gradient of the rational quadratic with respect to x2
    def dgradcovfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        alpha = self.theta[2]
        r2 = np.sum((self.x1 - self.x2)**2)
        rl2 = r2/l**2
        dgrad_s = float(2. * sigmaf/l**2 * (self.x1 - self.x2) /
                        (1. + rl2/(2. * alpha))**(1. + alpha))
        dgrad_l = float(sigmaf**2 * 4. * alpha**2/l * (self.x1 - self.x2) *
                        (-2. * l**2 + r2)/
                        ((2. * alpha * l**2 + r2)**2 * 
                         (1. + rl2/(2. * alpha))**alpha))
        dgrad_a = float(sigmaf**2 * 2. * (self.x1 - self.x2) * 
                        ((1. + alpha) * r2 - alpha * (2. * l**2 + r2) * 
                         log(1. + rl2/(2. * alpha))) /
                        ((2. * alpha * l**2 + r2)**2 * 
                         (1. + rl2/(2. * alpha))**alpha))
        dgrad = array([dgrad_s, dgrad_l, dgrad_a])
        return dgrad

    # derivative of the gradient of the rational quadratic with 
    # respect to x1 and x2
    # dk/(d1 d2)
    def ddgradcovfunc(self):
        if (self.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        sigmaf = self.theta[0]
        l = self.theta[1]
        alpha = self.theta[2]
        r2 = np.sum((self.x1 - self.x2)**2)
        rl2 = r2/l**2
        ddgrad_s = float(sigmaf * 4. * alpha * (2. * alpha * (l**2 - r2) - r2)/
                         ((2. * alpha * l**2 + r2)**2 * 
                          (1. + rl2/(2. * alpha))**alpha))
        t1 = 2. * alpha * (2. * l**4 - 5. * l**2 * r2 + r2**2)
        t2 = (-6. * l**2 + r2) * r2
        ddgrad_l = float(-sigmaf**2 * 4. * alpha**2/l * (t1 + t2) /
                          ((2. * alpha * l**2 + r2)**3 * 
                           (1. + rl2/(2. * alpha))**alpha))
        t3 = -2. * alpha**2 * (l**2 - r2) * r2 
        t4 = (alpha * (-6. * l**2 + 5. * r2) + r2) * r2
        f1 = 4. * alpha**2 * l**2 * (l**2 - r2) - r2**2 - 2. * alpha * r2**2
        t5 = alpha * f1 * log(1. + rl2/(2. * alpha))
        dgrad_a = float(-sigmaf**2 * 2. * (t3 + t4 + t5)/
                         ((2. * alpha * l**2 + r2)**3 * 
                           (1. + rl2/(2. * alpha))**alpha))
        ddgrad = array([ddgrad_s, ddgrad_l, dgrad_a])
        return ddgrad

