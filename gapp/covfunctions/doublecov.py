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
from numpy import concatenate, reshape
import warnings

class DoubleCovariance(cov.CovarianceFunction):
    # initialize class with initial hyperparameter theta
    def __init__(self, covfunction1, covfunction2, theta, X, Y):
        self.covf1 = covfunction1(None, X, Y)
        self.covf2 = covfunction2(None, X, Y)
        self.lth1 = len(self.covf1.theta)
        self.lth2 = len(self.covf2.theta)
        if(theta == None):
            theta = concatenate((self.covf1.theta, self.covf2.theta))
        else:
            self.covf1.theta = theta[:self.lth1]
            self.covf2.theta = theta[self.lth1:]
        cov.CovarianceFunction.__init__(self, theta)
        if (np.min(self.theta) < 0.0):
            warnings.warn("Illegal hyperparameters in the " +
                          "initialization of DoubleCovariance.")

    def set_x1x2(self, x1, x2):
        if ((type(x1) and type(x2) in [type(1), type(1.0)]) or 
            (len(x1) == 1 and len(x2) == 1)):
            # 1 dimensional data
            self.x1 = reshape(x1, (1, 1))
            self.x2 = reshape(x2, (1, 1))
            self.covf1.x1 = reshape(x1, (1, 1))
            self.covf1.x2 = reshape(x2, (1, 1))
            self.covf2.x1 = reshape(x1, (1, 1))
            self.covf2.x2 = reshape(x2, (1, 1))
            self.covf1.multiD = 'False'
            self.covf2.multiD = 'False'
        elif (len(x1) == len(x2)):
            # multi-dimensional data
            self.x1 = x1
            self.x2 = x2
            self.covf1.x1 = x1
            self.covf1.x2 = x2
            self.covf2.x1 = x1
            self.covf2.x2 = x2
            self.covf1.multiD = 'True'
            self.covf2.multiD = 'True'
        else:
            print ("ERROR: wrong data type of (x1, x2)")
            exit


    def covfunc(self):
        self.set_x1x2(self.x1, self.x2)
        covariance = self.covf1.covfunc() + self.covf2.covfunc()
        return covariance


    def gradcovfunc(self):
        self.set_x1x2(self.x1, self.x2)
        grad = concatenate((self.covf1.gradcovfunc(), self.covf2.gradcovfunc()))
        return grad

    def dcovfunc(self):
        self.set_x1x2(self.x1, self.x2)
        if (self.covf1.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        dcov = self.covf1.dcovfunc() + self.covf2.dcovfunc()
        return float(dcov)


    def ddcovfunc(self):
        self.set_x1x2(self.x1, self.x2)
        if (self.covf1.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        dcov = self.covf1.ddcovfunc() + self.covf2.ddcovfunc()
        return float(dcov)

    def d2covfunc(self):
        self.set_x1x2(self.x1, self.x2)
        if (self.covf1.multiD=='True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        dcov = self.covf1.d2covfunc() + self.covf2.d2covfunc()
        return float(dcov)

    def d2d2covfunc(self):
        self.set_x1x2(self.x1, self.x2)
        if (self.covf1.multiD=='True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        dcov = self.covf1.d2d2covfunc() + self.covf2.d2d2covfunc()
        return float(dcov)

    def d2d3covfunc(self):
        self.set_x1x2(self.x1, self.x2)
        if (self.covf1.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        dcov = self.covf1.d2d3covfunc() + self.covf2.d2d3covfunc()
        return float(dcov)

    def dd2covfunc(self):
        self.set_x1x2(self.x1, self.x2)
        if (self.covf1.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        dcov = self.covf1.dd2covfunc() + self.covf2.dd2covfunc()
        return float(dcov)

    def d3covfunc(self):
        self.set_x1x2(self.x1, self.x2)
        if (self.covf1.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        dcov = self.covf1.d3covfunc() + self.covf2.d3covfunc()
        return float(dcov)

    def d3d3covfunc(self):
        self.set_x1x2(self.x1, self.x2)
        if (self.covf1.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        dcov = self.covf1.d3d3covfunc() + self.covf2.d3d3covfunc()
        return dcov

    def dd3covfunc(self):
        self.set_x1x2(self.x1, self.x2)
        if (self.covf1.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        dcov = self.covf1.dd3covfunc() + self.covf2.dd3covfunc()
        return dcov

    def dgradcovfunc(self):
        self.set_x1x2(self.x1, self.x2)
        if (self.covf1.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        dgrad = concatenate((self.covf1.dgradcovfunc(), 
                            self.covf2.dgradcovfunc()))
        return dgrad

    def ddgradcovfunc(self):
        self.set_x1x2(self.x1, self.x2)
        if (self.covf1.multiD == 'True'): 
            raise RuntimeError("Derivative calculations are only implemented" + 
                               " for 1-dimensional inputs x.")
        ddgrad = concatenate((self.covf1.ddgradcovfunc(), 
                             self.covf2.ddgradcovfunc()))
        return ddgrad


