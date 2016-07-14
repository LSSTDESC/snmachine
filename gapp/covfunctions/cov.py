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

import numpy as np
from numpy import append, array, flatnonzero, reshape, take, zeros

class CovarianceFunction(object):
    def __init__(self, theta):
        self.theta = array(theta)
        # initial theta. 
        # in contrast to theta, initheta will not change during the gp
        self.initheta = array(theta) 

    def set_x1x2(self, x1, x2):
        if ((type(x1) and type(x2) in [type(1), type(1.0)]) or 
            (len(x1) == 1 and len(x2) == 1)):
            # 1 dimensional data
            self.x1 = reshape(x1, (1, 1))
            self.x2 = reshape(x2, (1, 1))
            self.multiD = 'False'
        elif (len(x1) == len(x2)):
            # multi-dimensional data
            self.x1 = x1
            self.x2 = x2
            self.multiD = 'True'
        else:
            raise TypeError("Wrong data type of (x1, x2).")


                    

    # return the constraints on the hyperparameters of the covariance function
    # for the function scipy.optimize.fmin_cobyla
    def constraints(self, thetatrain='True'):
        if (thetatrain == 'True'):
            inith = self.initheta
        else:
            if(thetatrain == 'False'):
                thetatrain = zeros(len(self.initheta))
            indices = flatnonzero(thetatrain)
            inith = take(self.initheta, indices)
        if(self.scaletrain == 'True'):
            inith = append(inith, self.iniscale)
        def const(theta):
            return float(np.min(theta - inith/1.0e15))
        return (const,)



    # return the bounds of the hyperparameters of the covariance function
    # for the function scipy.optimize.fmin_tnc
    def bounds(self):
        inith = self.initheta
        if(self.scaletrain == 'True'):
            inith = append(inith, self.iniscale)
        bounds = []
        for i in range(len(inith)):
            bounds.append((inith[i]/1.0e15, None))
        return bounds




    # return the constraints on the hyperparameters of the covariance function
    # for the function scipy.optimize.fmin_cobyla
    # when using derivative mearurements
    def dmconstraints(self, thetatrain='True'):
        if (thetatrain == 'True'):
            inith = self.initheta
        else:
            if(thetatrain == 'False'):
                thetatrain = zeros(len(self.initheta))
            indices = flatnonzero(thetatrain)
            inith = take(self.initheta, indices)
        if(self.scaletrain == 'True'):
            inith = append(inith, self.iniscale)
        if(self.dscaletrain == 'True'):
            inith = append(inith, self.inidscale)
        def const(theta):
            return float(np.min(theta - inith/1.0e15))
        return (const,)



                    
    # return the bounds of the hyperparameters of the covariance function
    # for the function scipy.optimize.fmin_tnc
    # when using derivative mearurements
    def dmbounds(self):
        inith = self.initheta
        if(self.scaletrain == 'True'):
            inith = append(inith, self.iniscale)
        if(self.dscaletrain == 'True'):
            inith = append(inith, self.inidscale)
        bounds = []
        for i in range(len(inith)):
            bounds.append((inith[i]/1.0e15, None))
        return bounds
