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


# dgp: reconstruction of f'(x) using measurements of f(x)
# d2gp: reconstruction of f''(x) using measurements of f(x)
# d3gp: reconstruction of f'''(x) using measurements of f(x)
# dm_gp: reconstruction of f(x) using measurements of f(x) and f'(x)
# dm_dgp: reconstruction of f'(x) using measurements of f(x) and f'(x)
# dm_d2gp: reconstruction of f''(x) using measurements of f(x) and f'(x)

import gp
import covariance
import numpy as np
from numpy import append, array, concatenate, diagonal, dot, eye, \
    flatnonzero, linalg, log, pi, reshape, resize, shape, sqrt, take, \
    transpose, zeros
import scipy.optimize as opt


class DGaussianProcess(gp.GaussianProcess):
    def __init__(self, X, Y, Sigma, covfunction=covariance.SquaredExponential,
                 theta=None, dX=None, dY=None, dSigma=None, Xstar=None,
                 cXstar=None, mu=None, dmu=None, d2mu=None, d3mu=None,
                 muargs=(), prior=None, gradprior=None, priorargs=(),
                 thetatrain='True', scale=None, scaletrain='True', 
                 grad='True'):
        assert (shape(X) == () or shape(X) == (len(X), ) or 
                shape(X) == (len(X), 1)), \
            "X must be 1-dimensional."
        gp.GaussianProcess.__init__(self, X, Y, Sigma, covfunction, 
                                    theta, Xstar, cXstar, mu, muargs,
                                    prior, gradprior, priorargs,
                                    thetatrain, scale, scaletrain, grad)
        # observational data of the derivative of f(x)
        self.set_ddata(dX, dY, dSigma)
        # first derivative of the a priori mean function of f(x)
        self.set_dmu(dmu)
        # second derivative of the a priori mean function of f(x)
        self.set_d2mu(d2mu)
        # second derivative of the a priori mean function of f(x)
        self.set_d3mu(d3mu)

    # set observational data of the derivative of f(x)
    def set_ddata(self, dX, dY, dSigma):
        if (dX == None or dY == None or dSigma == None):
            self.dX = None
            self.dY = None
            self.dY_dmu = None
            self.dSigma = None
            self.dmuptodate = 'False'
        elif (type(dX) and type(dY) and type(dSigma) in [type(1), type(1.0)]):
            self.dX = array(reshape(dX, (1, 1)))
            self.dY = array(reshape(dY, (1, )))          
            # if Sigma is a number, it is considered to be an error which is 
            # turned into a covariance matrix
            self.dSigma = array(reshape(dSigma**2, (1, 1))) 
            self.dn = 1           
            try:
                if (self.dmu != None):
                    self.subtract_dmu()
            except AttributeError:
                pass
            self.dmuptodate = 'False'
        else:
            dn = len(dX)
            assert (len(dY) == dn and len(dSigma) == dn), \
                "dX, dY and dSigma must have same length."
            if(shape(dX) == (dn, )):
                dX = reshape(dX, (dn, 1))
            self.dX = array(dX)   
            self.dY = array(dY)
            # data covariance matrix
            if (shape(dSigma) == (dn, dn)):
                self.dSigma = array(dSigma)
            elif (shape(dSigma) in [(dn, ), (dn, 1)]):
                # turn vector into diagonal covariance matrix
                self.dSigma = dSigma * eye(dn) * dSigma
            else:
                raise AssertionError("dSigma must be vector or nxn matrix.")
            self.dn = dn                 # number of data points
            try:
                if (self.dmu != None):
                    self.subtract_dmu()
            except AttributeError:
                pass
            self.dmuptodate = 'False'



    # set the first derivative of an a priori mean function
    def set_dmu(self, dmu):
        self.uptodate = 'False'
        self.dmuptodate = 'False'
        if (self.mu == None and dmu != None):
            warnings.warn("dmu given, but mu=None. dmu will be ignored.")
            self.dmu = None
        else:
            self.dmu = dmu
        if (self.dY != None):
            if(dmu == None):
                self.dY_dmu = self.dY[:]
            else:
                self.subtract_dmu()

    # set the second derivative of an a priori mean function
    def set_d2mu(self, d2mu):
        self.uptodate = 'False'
        self.dmuptodate = 'False'
        if (self.mu == None and d2mu != None):
            warnings.warn("d2mu given, but mu=None. d2mu will be ignored.")
            self.d2mu = None
        else:
            self.d2mu = d2mu

    # set the third derivative of an a priori mean function
    def set_d3mu(self, d3mu):
        self.uptodate = 'False'
        self.dmuptodate = 'False'
        if (self.mu == None and d3mu != None):
            warnings.warn("d3mu given, but mu=None. d3mu will be ignored.")
            self.d3mu = None
        else:
            self.d3mu = d3mu

    # subtract the derivative of the a priori mean from the derivative data
    def subtract_dmu(self):
        self.dY_dmu = zeros(self.dn)
        for i in range(self.dn):
            self.dY_dmu[i] = self.dY[i] - self.dmu(self.dX[i], *self.muargs)
        self.dmuptodate = 'False'


    def log_likelihood(self, theta=None, dX='False', dY='False', 
                       dSigma='False', mu='False', dmu='False', muargs=(),
                       prior='False', priorargs=(), scale='False'):
        # set new attributes
        if (theta != None): 
            self.set_theta(theta)
        if (dX != 'False' and dY != 'False' and dSigma != 'False'):
            self.set_ddata(dX, dY, dSigma)
        elif (dX != 'False' or dY != 'False' or dSigma != 'False'):
            if (dX != None or dY != None or dSigma != None):
                self.set_ddata(None, None, None)
            else:
                warnings.warn("dX, dY and dSigma have to be changed " +
                              "simultaneously. Old values of dX, dY " +
                              "and dSigma will be used.")
        if (mu != 'False'): 
            self.set_mu(mu, muargs)
        if (dmu != 'False'): 
            self.set_dmu(dmu)
        if (prior != 'False'): 
            self.set_prior(prior, self.gradprior, priorargs)
        if (scale != 'False'): 
            self.set_scale(scale)
        if (self.dX == None):
            return(-self.nlog_likelihood())
        else:
            return(-self.dm_nlog_likelihood())
        

    # train the hyperparameters
    def hypertrain(self, theta=None, dX='False', dY='False', dSigma='False',
                   mu='False', dmu='False', muargs=(), prior='False',
                   gradprior=None, priorargs=(), thetatrain=None,
                   scale='False', scaletrain=None, grad=None):
        # set new attributes
        if (theta != None): 
            self.set_theta(theta)
        if (dX != 'False' and dY != 'False' and dSigma != 'False'):
            self.set_ddata(dX, dY, dSigma)
        elif (dX != 'False' or dY != 'False' or dSigma != 'False'):
            if (dX != None or dY != None or dSigma != None):
                self.set_ddata(None, None, None)
            else:
                warnings.warn("dX, dY and dSigma have to be changed " +
                              "simultaneously. Old values of dX, dY " +
                              "and dSigma will be used.")
        if (mu != 'False'): 
            self.set_mu(mu, muargs)
        if (dmu != 'False'): 
            self.set_dmu(dmu)
        if (prior != 'False'): 
            self.set_prior(prior, gradprior, priorargs)
        if (thetatrain != None): 
            self.set_thetatrain(thetatrain)
        if (scale != 'False'): 
            if (scaletrain == None): 
                scaletrain = 'True'
            self.set_scale(scale)
            self.set_scaletrain(scaletrain)
        elif (scaletrain != None):
            self.set_scaletrain(scaletrain)
        if (grad != None): 
            self.set_grad(grad)
        if (self.thetatrain == 'False' and self.covf.scaletrain == 'False'):
            raise RuntimeError("thetatrain='False' and scaletrain=='False', " +
                               "i.e. no parameters are to be trained.")
        # train the hyperparameters
        if (self.dX == None):
            return(self.fhypertrain())
        else:
            return(self.dm_hypertrain())



    ############
    # function #
    ############

    # full Gaussian process run
    def gp(self, theta=None, dX='False', dY='False', dSigma='False', 
           Xstar=None, cXstar=None, mu='False', dmu='False', muargs=(),
           prior='False', gradprior=None, priorargs=(), thetatrain=None,
           scale='False', scaletrain=None, grad=None, unpack='False'):
        # set new attributes
        if (theta != None): 
            self.set_theta(theta)
        if (dX != 'False' and dY != 'False' and dSigma != 'False'):
            self.set_ddata(dX, dY, dSigma)
        elif (dX != 'False' or dY != 'False' or dSigma != 'False'):
            if (dX != None or dY != None or dSigma != None):
                self.set_ddata(None, None, None)
            else:
                warnings.warn("dX, dY and dSigma have to be changed " +
                              "simultaneously. Old values of dX, dY " +
                              "and dSigma will be used.")
        if (Xstar != None): 
            self.set_Xstar(Xstar)
            if (cXstar != None):
                warnings.warn("Xstar and cXstar given in dpg.gp. \n" +
                              "cXstar will be ignored.")
        elif (cXstar != None): 
            self.create_Xstar(cXstar[0], cXstar[1], cXstar[2])
        if (mu != 'False'): 
            self.set_mu(mu, muargs)
        if (dmu != 'False'):
            self.set_dmu(dmu)
        if (prior != 'False'):
            self.set_prior(prior, gradprior, priorargs)
        if (thetatrain != None): 
            self.set_thetatrain(thetatrain)
        if (scale != 'False'): 
            if (scaletrain == None):
                scaletrain = 'True'
            self.set_scale(scale)
            self.set_scaletrain(scaletrain)
        if (grad != None):
            self.set_grad(grad)
        # GP run
        if (self.dX == None):
            return(self.fgp(unpack=unpack))
        else:
            return(self.dm_gp(unpack=unpack))
        


    ####################
    # first derivative #
    ####################

    # calculate vector of covariances dkstar between one test point
    # and the n training points
    def d_covariance_vector(self, xstar):
        dkstar = zeros(self.n)
        self.covf.set_x1x2(self.X[0, :], xstar)
        for i in range(self.n):
            self.covf.x1 = self.X[i, :]
            dkstar[i] = self.covf.dcovfunc()
        return dkstar

    # calculate the predictive mean and standard deviation of (f'-mu') at 
    # test point xstar
    def dprediction(self, xstar):
        if (self.uptodate == 'False'):
            self.input_covariance()
            self.alpha_L()
            if (self.grad == 'True'):
                self.grad_covariance()
            self.uptodate = 'True'
        if (self.alpha == None):
            raise RuntimeError("Invalid hyperparameters. Covariance matrix" +
                               " not positive definit.")
        # calculate covariance vector kstar
        kstar = self.d_covariance_vector(xstar)
        # predictive mean
        mean = dot(transpose(kstar), self.alpha)
        # calculate predictive standard deviation
        v = linalg.solve(self.L, kstar)
        self.covf.x1 = xstar
        self.covf.x2 = xstar
        covstar = self.covf.ddcovfunc()
        stdev = sqrt(covstar - dot(transpose(v), v))
        return(mean, stdev)

    def dgp(self, theta=None, dX='False', dY='False', dSigma='False', 
            Xstar=None, cXstar=None, mu='False', dmu='False', muargs=(),
            prior='False', gradprior=None, priorargs=(), thetatrain=None,
            scale='False', scaletrain=None, grad=None, unpack='False'):
        # set new attributes
        if (theta != None):
            self.set_theta(theta)
        if (dX != 'False' and dY != 'False' and dSigma != 'False'):
            self.set_ddata(dX, dY, dSigma)
        elif (dX != 'False' or dY != 'False' or dSigma != 'False'):
            if (dX != None or dY != None or dSigma != None):
                self.set_ddata(None, None, None)
            else:
                warnings.warn("dX, dY and dSigma have to be changed " + 
                              "simultaneously. Old values of dX, dY and " +
                              "dSigma will be used.")
        if (Xstar != None): 
            self.set_Xstar(Xstar)
            if (cXstar != None):
                warnings.warn("Xstar and cXstar given. cXstar will be" +
                              " ignored.")
        elif (cXstar != None): 
            self.create_Xstar(cXstar[0], cXstar[1], cXstar[2])
        if (mu != 'False'): 
            self.set_mu(mu, muargs)
        if (dmu != 'False'): 
            self.set_dmu(dmu)
        if (self.dmu == None and self.mu != None):
            warnings.warn("mu given, but dmu=None. mu will be ignored")
            self.unset_mu()
        if (prior != 'False'): 
            self.set_prior(prior, gradprior, priorargs)
        if (thetatrain != None):
            self.set_thetatrain(thetatrain)
        if (scale != 'False'): 
            if (scaletrain == None): 
                scaletrain = 'True'
            self.set_scale(scale)
            self.set_scaletrain(scaletrain)
        elif (scaletrain != None):
            self.set_scaletrain(scaletrain)
        if (grad != None):
            self.set_grad(grad)
        # GP run
        if (self.dX == None):
            return(self.fdgp(unpack=unpack))
        else:
            return(self.dm_dgp(unpack=unpack))
        

    def fdgp(self, unpack):
        # train the hyperparameters
        if (self.thetatrain != 'False' or self.covf.scaletrain == 'True' or 
            self.covf.dscaletrain == 'True'):
            self.hypertrain()
        # reconstruct f'(x)
        dfmean_mu = zeros(self.nstar)
        dfmean = zeros(self.nstar)
        dfstd = zeros(self.nstar)
        for i in range(self.nstar):
            (dfmean_mu[i], dfstd[i]) = self.dprediction(self.Xstar[i, :])
        if (self.mu != None):
            for i in range(self.nstar):
                dfmean[i] = dfmean_mu[i] + self.dmu(self.Xstar[i], 
                                                    *self.muargs)
        else:
            dfmean[:] = dfmean_mu[:]
        self.dfmean_mu = dfmean_mu
        self.dfstd_mu = dfstd
        self.dfmean = dfmean
        self.dfstd = dfstd
        self.dreconstruction = concatenate((self.Xstar, 
                                            reshape(dfmean, (self.nstar, 1)),
                                            reshape(dfstd, (self.nstar, 1))), 
                                           axis=1)
        if (self.scale == None):
            if (unpack == 'False'):
                return(self.dreconstruction, self.covf.theta)
            else:
                return(self.Xstar, self.dfmean, self.dfstd, self.covf.theta)
        else:
            if (unpack == 'False'):
                return(self.dreconstruction, self.covf.theta, self.scale)
            else:
                return(self.Xstar, self.dfmean, self.dfstd, self.covf.theta,
                       self.scale)


    #####################
    # second derivative #
    #####################

    # calculate vector of covariances d2kstar between one test point
    # and the n training points
    def d2_covariance_vector(self, xstar):
        d2kstar = zeros(self.n)
        self.covf.set_x1x2(self.X[0,:], xstar)
        for i in range(self.n):
            self.covf.x1 = self.X[i, :]
            d2kstar[i] = self.covf.d2covfunc()
        return d2kstar

    # calculate the predictive mean and standard deviation of (f''-mu'') 
    # at test point xstar
    def d2prediction(self, xstar):
        if (self.uptodate == 'False'):
            self.input_covariance()
            self.alpha_L()
            if (self.grad == 'True'):
                self.grad_covariance()
            self.uptodate = 'True'
        if (self.alpha == None):
            raise RuntimeError("Invalid hyperparameters. Covariance matrix " +
                               "not positive definit")
        # calculate covariance vector kstar
        kstar = self.d2_covariance_vector(xstar)
        # predictive mean
        mean = dot(transpose(kstar), self.alpha)
        # calculate predictive standard deviation
        v = linalg.solve(self.L, kstar)
        self.covf.x1 = xstar
        self.covf.x2 = xstar
        covstar = self.covf.d2d2covfunc()
        stdev = sqrt(covstar - dot(transpose(v), v))
        return(mean, stdev)

    def d2gp(self, theta=None, dX='False', dY='False', dSigma='False', 
             Xstar=None, cXstar=None, mu='False', dmu='False', d2mu='False',
             muargs=(), prior='False', gradprior=None, priorargs=(),
             thetatrain=None, scale='False', scaletrain=None, grad=None,
             unpack='False'):
        # set new attributes
        if (theta != None): 
            self.set_theta(theta)
        if (dX != 'False' and dY != 'False' and dSigma != 'False'):
            self.set_ddata(dX, dY, dSigma)
        elif (dX != 'False' or dY != 'False' or dSigma != 'False'):
            if (dX != None or dY != None or dSigma != None):
                self.set_ddata(None, None, None)
            else:
                warnings.warn("dX, dY and dSigma have to be changed " +
                              "simultaneously. Old values of dX, dY " +
                              "and dSigma will be used.")
        if (Xstar != None): 
            self.set_Xstar(Xstar)
            if (cXstar != None):
                warnings.warn("Xstar and cXstar given. cXstar will " +
                              "be ignored.")
        elif (cXstar != None): 
            self.create_Xstar(cXstar[0], cXstar[1], cXstar[2])
        if (mu != 'False'):
            self.set_mu(mu, muargs)
        if (dmu != 'False'): 
            self.set_dmu(dmu)
        if (d2mu != 'False'):
            self.set_d2mu(d2mu)
        if (self.dX != None and self.dmu == None and self.mu != None):
            warnings.warn("mu given, but dmu=None. mu will be ignored")
            self.unset_mu()
        if (self.d2mu == None and self.mu != None):
            warnings.warn("mu given, but d2mu=None. mu will be ignored")
            self.unset_mu()
        if (prior != 'False'): 
            self.set_prior(prior, gradprior, priorargs)
        if (thetatrain != None):
            self.set_thetatrain(thetatrain)
        if (scale != 'False'): 
            if (scaletrain == None):
                scaletrain = 'True'
            self.set_scale(scale)
            self.set_scaletrain(scaletrain)
        if (grad != None):
            self.set_grad(grad)
        # GP run
        if (self.dX == None):
            return(self.fd2gp(unpack=unpack))
        else:
            return(self.dm_d2gp(unpack=unpack))

    

    def fd2gp(self, unpack):
        # train the hyperparameters
        if (self.thetatrain != 'False' or self.covf.scaletrain == 'True' or 
            self.covf.dscaletrain == 'True'):
            self.hypertrain()
        # reconstruct f'(x)
        d2fmean_mu = zeros(self.nstar)
        d2fmean = zeros(self.nstar)
        d2fstd = zeros(self.nstar)
        for i in range(self.nstar):
            (d2fmean_mu[i], d2fstd[i]) = self.d2prediction(self.Xstar[i, :])
        if (self.mu != None):
            for i in range(self.nstar):
                d2fmean[i] = d2fmean_mu[i] + self.d2mu(self.Xstar[i],
                                                       *self.muargs)
        else:
            d2fmean[:] = d2fmean_mu[:]
        self.d2fmean_mu = d2fmean_mu
        self.d2fstd_mu = d2fstd
        self.d2fmean = d2fmean
        self.d2fstd = d2fstd
        self.d2reconstruction = concatenate((self.Xstar,
                                             reshape(d2fmean, (self.nstar, 1)),
                                             reshape(d2fstd, (self.nstar, 1))),
                                            axis=1)
        if (self.scale == None):
            if (unpack == 'False'):
                return(self.d2reconstruction, self.covf.theta)
            else:
                return(self.Xstar, self.d2fmean, self.d2fstd, self.covf.theta)
        else:
            if (unpack == 'False'):
                return(self.d2reconstruction, self.covf.theta, self.scale)
            else:
                return(self.Xstar, self.d2fmean, self.d2fstd, self.covf.theta,
                       self.scale)


    ####################
    # third derivative #
    ####################

    # calculate vector of covariances d3kstar between one test point
    # and the n training points
    def d3_covariance_vector(self, xstar):
        d3kstar = zeros(self.n)
        self.covf.set_x1x2(self.X[0, :], xstar)
        for i in range(self.n):
            self.covf.x1 = self.X[i, :]
            d3kstar[i] = self.covf.d3covfunc()
        return d3kstar

    # calculate the predictive mean and standard deviation of (f'''-mu''') at 
    # test point xstar
    def d3prediction(self, xstar):
        if (self.uptodate == 'False'):
            self.input_covariance()
            self.alpha_L()
            if (self.grad == 'True'):
                self.grad_covariance()
            self.uptodate = 'True'
        if (self.alpha == None):
            raise RuntimeError("Invalid hyperparameters. Covariance matrix " +
                               "not positive definit.")
        # calculate covariance vector kstar
        kstar = self.d3_covariance_vector(xstar)
        # predictive mean
        mean = dot(transpose(kstar), self.alpha)
        # calculate predictive standard deviation
        v = linalg.solve(self.L, kstar)
        self.covf.x1 = xstar
        self.covf.x2 = xstar
        covstar = self.covf.d3d3covfunc()
        stdev = sqrt(covstar - dot(transpose(v), v))
        return (mean, stdev)


    def d3gp(self, theta=None, dX='False', dY='False', dSigma='False', 
             Xstar=None, cXstar=None, mu='False', dmu='False', d3mu='False',
             muargs=(), prior='False', gradprior=None, priorargs=(),
             thetatrain=None, scale='False', scaletrain=None, grad=None,
             unpack='False'):
        # set new attributes
        if (theta != None):
            self.set_theta(theta)
        if (dX != 'False' and dY != 'False' and dSigma != 'False'):
            self.set_ddata(dX, dY, dSigma)
        elif (dX != 'False' or dY != 'False' or dSigma != 'False'):
            if (dX != None or dY != None or dSigma != None):
                self.set_ddata(None, None, None)
            else:
                warnings.warn("dX, dY and dSigma have to be changed " +
                              "simultaneously. Old values of dX, dY "+
                              "and dSigma will be used.")
        if (Xstar != None): 
            self.set_Xstar(Xstar)
            if (cXstar != None):
                warnings.warn("Xstar and cXstar given. cXstar will be" +
                              " ignored.")
        elif (cXstar != None):
            self.create_Xstar(cXstar[0], cXstar[1], cXstar[2])
        if (mu != 'False'):
            self.set_mu(mu, muargs)
        if (dmu != 'False'): 
            self.set_dmu(dmu)
        if (d3mu != 'False'):
            self.set_d3mu(d3mu)
        if (self.dX != None and self.dmu == None and self.mu != None):
            warnings.warn("mu given, but dmu=None. mu will be ignored")
            self.unset_mu()
        if (self.d3mu == None and self.mu != None):
            warnings.warn("mu given, but d3mu=None. mu will be ignored")
            self.unset_mu()
        if (prior != 'False'):
            self.set_prior(prior, gradprior, priorargs)
        if (thetatrain != None):
            self.set_thetatrain(thetatrain)
        if (scale != 'False'): 
            if (scaletrain == None):
                scaletrain = 'True'
            self.set_scale(scale)
            self.set_scaletrain(scaletrain)
        if (grad != None):
            self.set_grad(grad)
        # GP run
        if (self.dX == None):
            return(self.fd3gp(unpack=unpack))
        else:
            return(self.dm_d3gp(unpack=unpack))

    def fd3gp(self, unpack):
        # train the hyperparameters
        if (self.thetatrain != 'False' or self.covf.scaletrain == 'True' or 
            self.covf.dscaletrain == 'True'):
            self.hypertrain()
        # reconstruct f'(x)
        d3fmean_mu = zeros(self.nstar)
        d3fmean = zeros(self.nstar)
        d3fstd = zeros(self.nstar)
        for i in range(self.nstar):
            (d3fmean_mu[i], d3fstd[i]) = self.d3prediction(self.Xstar[i, :])
        if (self.mu != None):
            for i in range(self.nstar):
                d3fmean[i] = d3fmean_mu[i] + self.d3mu(self.Xstar[i],
                                                       *self.muargs)
        else:
            d3fmean[:] = d3fmean_mu[:]
        self.d3fmean_mu = d3fmean_mu
        self.d3fstd_mu = d3fstd
        self.d3fmean = d3fmean
        self.d3fstd = d3fstd
        self.d3reconstruction = concatenate((self.Xstar,
                                             reshape(d3fmean, (self.nstar, 1)),
                                             reshape(d3fstd, (self.nstar, 1))),
                                            axis=1)
        if (self.scale == None):
            if (unpack == 'False'):
                return(self.d3reconstruction, self.covf.theta)
            else:
                return(self.Xstar, self.d3fmean, self.d3fstd, self.covf.theta)
        else:
            if (unpack == 'False'):
                return(self.d3reconstruction, self.covf.theta, self.scale)
            else:
                return(self.Xstar, self.d3fmean, self.d3fstd, self.covf.theta,
                       self.scale)


##################################################
## covariances between f(x) and its derivatives ##
##################################################

    # var(f)
    def f_var(self, v):
        covstar = self.covf.covfunc()
        fvar = covstar - dot(v, v)
        return fvar

    # var(f')
    def df_var(self, dv):
        covstar = self.covf.ddcovfunc()
        dfvar = covstar - dot(dv, dv)
        return dfvar

    # var(f'')
    def d2f_var(self, d2v):
        covstar = self.covf.d2d2covfunc()
        d2fvar = covstar - dot(d2v, d2v)
        return d2fvar

    # var(f''')
    def d3f_var(self, d3v):
        covstar = self.covf.d3d3covfunc()
        d3fvar = covstar - dot(d3v, d3v)
        return d3fvar

    # cov(f,f')
    def f_df_cov(self, v, dv):
        covstar = self.covf.dcovfunc()
        fdf = covstar - dot(dv, v)
        return fdf

    # cov(f,f'')
    def f_d2f_cov(self, v, d2v):
        covstar = self.covf.d2covfunc()
        fd2f = covstar - dot(d2v, v)
        return fd2f

    # cov(f,f''')
    def f_d3f_cov(self, v, d3v):
        covstar = self.covf.d3covfunc()
        fd3f = covstar - dot(d3v, v)
        return fd3f

    # cov(f',f'')
    def df_d2f_cov(self, dv, d2v):
        covstar = self.covf.dd2covfunc()
        dfd2f = covstar - dot(d2v, dv)
        return dfd2f

    # cov(f',f''')
    def df_d3f_cov(self, dv, d3v):
        covstar = self.covf.dd3covfunc()
        dfd3f = covstar - dot(d3v, dv)
        return dfd3f

    # cov(f'',f''')
    def d2f_d3f_cov(self, d2v, d3v):
        covstar = self.covf.d2d3covfunc()
        d2fd3f = covstar - dot(d3v, d2v)
        return d2fd3f


    # covariances between f and its derivatives at Xstar
    def f_covariances(self, fclist=[0,1,2,3]):
        if (self.dX == None):
            return(self.ff_covariances(fclist))
        else:
            return(self.dm_f_covariances(fclist))


    def ff_covariances(self, fclist):
        if (self.uptodate == 'False'):
            self.input_covariance()
            self.alpha_L()
            if (self.grad == 'True'):
                self.grad_covariance()
            self.uptodate = 'True'
        fclist.sort()
        Xstar = self.Xstar
        nstar = self.nstar
        fcov = zeros((nstar, len(fclist), len(fclist)))
        for i in range(nstar):
            fcmatrix = zeros((4, 4))
            xstar = Xstar[i, :]
            if (0 in fclist):
                kstar = self.covariance_vector(xstar)
                v = linalg.solve(self.L,kstar)
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                fcmatrix[0, 0] = self.f_var(v)
            if (1 in fclist):
                dkstar = self.d_covariance_vector(xstar)
                dv = linalg.solve(self.L, dkstar)
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                fcmatrix[1, 1] = self.df_var(dv)
            if (2 in fclist):
                d2kstar = self.d2_covariance_vector(xstar)
                d2v = linalg.solve(self.L, d2kstar)
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                fcmatrix[2, 2] = self.d2f_var(d2v)
            if (3 in fclist):
                d3kstar = self.d3_covariance_vector(xstar)
                d3v = linalg.solve(self.L, d3kstar)
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                fcmatrix[3, 3] = self.d3f_var(d3v)
            if (0 in fclist and 1 in fclist):
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                fdf = self.f_df_cov(v, dv)
                fcmatrix[0, 1] = fdf
                fcmatrix[1,0] = fdf
            if (0 in fclist and 2 in fclist):
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                fd2f = self.f_d2f_cov(v, d2v)
                fcmatrix[0, 2] = fd2f
                fcmatrix[2, 0] = fd2f
            if (0 in fclist and 3 in fclist):
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                fd3f = self.f_d3f_cov(v, d3v)
                fcmatrix[0, 3] = fd3f
                fcmatrix[3, 0] = fd3f
            if (1 in fclist and 2 in fclist):
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                dfd2f = self.df_d2f_cov(dv, d2v)
                fcmatrix[1, 2] = dfd2f
                fcmatrix[2, 1] = dfd2f
            if (1 in fclist and 3 in fclist):
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                dfd3f = self.df_d3f_cov(dv, d3v)
                fcmatrix[1, 3] = dfd3f
                fcmatrix[3, 1] = dfd3f
            if (2 in fclist and 3 in fclist):
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                d2fd3f = self.d2f_d3f_cov(d2v, d3v)
                fcmatrix[2, 3] = d2fd3f
                fcmatrix[3, 2] = d2fd3f
            fcov[i, :, :] = take(take(fcmatrix, fclist, axis=0), fclist, axis=1)
        return fcov


################
##   DM_GP   ##
################

    def dm_input_covariance(self):
        n = self.n
        dn = self.dn
        self.input_covariance()
        A = zeros((n + dn, n + dn))
        A[:n, :n] = self.K[:, :]
        if (self.scale == None):
            A[:n, :n] = A[:n, :n] + self.Sigma[:, :]
        else:
            A[:n, :n] = A[:n, :n] + self.scale**2 * self.Sigma[:, :]
        for i in range(dn):
            for j in range(dn):
                if (j >= i):
                    self.covf.x1 = self.dX[i]
                    self.covf.x2 = self.dX[j]
                    A[n+i, n+j] = self.covf.ddcovfunc()
                else:
                    A[n+i, n+j] = A[n+j, n+i]
        for i in range(n):
            for j in range(dn):
                self.covf.x1 = self.X[i]
                self.covf.x2 = self.dX[j]
                A[i, n+j] = self.covf.dcovfunc()
                A[n+j, i] = A[i, n+j]
        if (self.covf.dscale == None):
            A[n:, n:] = A[n:, n:] + self.dSigma[:, :]
        else:
            A[n:, n:] = A[n:, n:] + self.covf.dscale**2 * self.dSigma[:, :]
        self.dmA = A

    # calculate the gradient of the covariance matrix with respect to theta
    def dm_grad_covariance(self):
        n = self.n
        dn = self.dn
        gradK = zeros((len(self.covf.theta), n + dn, n + dn))
        for i in range(n):
            for j in range(n):
                if (j >= i):
                    self.covf.x1 = self.X[i]
                    self.covf.x2 = self.X[j]
                    gradK[:, i, j] = self.covf.gradcovfunc()
                else:
                    gradK[:, i, j] = gradK[:, j, i]
        for i in range(dn):
            for j in range(dn):
                if (j >= i):
                    self.covf.x1 = self.dX[i]
                    self.covf.x2 = self.dX[j]
                    gradK[:, n+i, n+j] = self.covf.ddgradcovfunc()
                else:
                    gradK[:, n+i, n+j] = gradK[:, n+j, n+i]
        for i in range(n):
            for j in range(dn):
                self.covf.x1 = self.X[i]
                self.covf.x2 = self.dX[j]
                gradK[:, i, n+j] = self.covf.dgradcovfunc()
                gradK[:, n+j, i] = gradK[:, i, n+j]
        if (self.covf.scaletrain=='True'):
            gradscale = zeros((1, n+dn, n+dn))
            for i in range(n):
                gradscale[0, i, i] = 2 * self.scale * self.Sigma[i, i]
            gradK = concatenate((gradK, gradscale))
        if (self.covf.dscaletrain == 'True'):
            graddscale = zeros((1, n + dn, n + dn))
            for i in range(dn):
                graddscale[0, n + i, n + i] = 2 * self.covf.dscale *\
                    self.dSigma[i, i]
            gradK = concatenate((gradK, graddscale))
        self.dmgradK = gradK


    # calculates alpha = (K + sigma)^{-1}Y 
    # and the Cholesky decomposition L
    def dm_alpha_L(self):
        A = self.dmA
        YdY = concatenate((self.Y_mu, self.dY_dmu))
        try:
            L = linalg.cholesky(A)
            b = linalg.solve(L, YdY)
            self.dmalpha = linalg.solve(transpose(L), b)
            self.dmL = L
        except np.linalg.linalg.LinAlgError: 
            # A not positive definit
            self.dmalpha = None
            self.dmL = None

    # dm_gp: calculate vector of covariances kstar between one test point
    # and the n+dn training points
    def dm_covariance_vector(self, xstar):
        kstar = zeros(self.n + self.dn)
        self.covf.x2 = xstar
        for i in range(self.n):
            self.covf.x1 = self.X[i, :]
            kstar[i] = self.covf.covfunc()
        self.covf.x1 = xstar
        for i in range(self.dn):
            self.covf.x2 = self.dX[i]
            kstar[self.n + i] = self.covf.dcovfunc()
        return kstar

    # calculate the log likelihood
    def dm_nlog_likelihood(self):
        if (self.dmuptodate == 'False'):
            self.dm_input_covariance()
            self.dm_alpha_L()
            if (self.grad == 'True'):
                self.dm_grad_covariance()
            self.dmuptodate = 'True'
        # log likelihood of prior
        if (self.prior != None):
            priorp = self.prior(self.covf.theta, *self.priorargs)
            if (priorp < 0.0):
                warnings.warn("Invalid prior. Negative prior values " +
                              "will be treated as prior=0.")
                return 1.0e+20
            if (priorp == 0.0):
                return 1.0e+20
            priorlogp = log(priorp)
        else:
            priorlogp = 0.0
        YdY = concatenate((self.Y_mu, self.dY_dmu))
        if (self.dmalpha == None):
            logp = 1.0e+20 - priorlogp
        else:
            logp = -(-0.5 * dot(YdY, self.dmalpha) - 
                      np.sum(log(diagonal(self.dmL))) - 
                      (self.n + self.dn)/2 * log(2 * pi) + priorlogp)
        return logp
        
    # calculate the grad log likelihood
    def dm_grad_nlog_likelihood(self):
        if (self.dmuptodate == 'False'):
            self.dm_input_covariance()
            self.dm_alpha_L()
            self.dm_grad_covariance()
            self.dmuptodate = 'True'
        logp = self.dm_nlog_likelihood()
        if (self.dmalpha == None):
            try:
                self.dmgradlogp = 0.9 * self.dmgradlogp
                return (logp, array(-self.dmgradlogp))
            except:
                raise RuntimeError("Invalid hyperparameters. " +
                                   "Covariance matrix not positive definit")
        # number of hyperparameters (plus 1 or 2 if scale!=None)
        nh = len(self.dmgradK)
        # calculate trace of alpha*alpha^T gradK
        traaTgradK = zeros(nh)
        for t in range(nh):
            aTgradK = zeros(self.n + self.dn)
            for i in range(self.n + self.dn):
                aTgradK[i] = np.sum(self.dmalpha[:] * self.dmgradK[t, :, i])
            traaTgradK[t] = np.sum(self.dmalpha[:] * aTgradK[:])
        # calculate trace of A^{-1} gradK
        invL = linalg.inv(self.dmL)
        invA = dot(transpose(invL), invL)
        trinvAgradK = zeros(nh)
        for t in range(nh):
            for i in range(self.n + self.dn):
                trinvAgradK[t] = trinvAgradK[t] + np.sum(invA[i, :] * 
                                                         self.dmgradK[t, :, i])
        # gradient of the prior log likelihood
        gradpriorlogp = zeros(nh)
        if (self.gradprior != None):
            gradpriorp = self.gradprior(self.covf.theta, *self.priorargs)
            if (self.prior == None):
                warnings.warn("No prior given. gradprior will be ignored")
            else:
                priorp = self.prior(self.covf.theta, *self.priorargs)
                for t in range(nh):
                    if (priorp == 0.0 and gradpriorp[t] == 0.0):
                        gradpriorlogp[t] = 0.0
                    elif (priorp <= 0.0):
                        gradpriorlogp[t] = sign(gradpriorp[t]) * 1.0e20
                    else:
                        gradpriorlogp[t] = gradpriorp[t]/priorp
        # gradient of the negative log likelihood
        gradlogp = array(-0.5 * (traaTgradK[:] - trinvAgradK[:]) - 
                          gradpriorlogp)
        self.dmgradlogp = gradlogp
        return (logp, gradlogp)

    # calculate the predictive mean and standard deviation of (f-mu) at 
    # test point xstar
    def dm_prediction(self, xstar):
        if (self.dmuptodate == 'False'):
            self.dm_input_covariance()
            self.dm_alpha_L()
            if (self.grad == 'True'):
                self.dm_grad_covariance()
            self.dmuptodate = 'True'
        if (self.dmalpha == None):
            raise RuntimeError("Invalid hyperparameters. Covariance matrix " +
                               "not positive definit")
        # calculate covariance vector kstar
        kstar = self.dm_covariance_vector(xstar)
        # predictive mean
        mean = dot(transpose(kstar), self.dmalpha)
        # calculate predictive standard deviation
        v = linalg.solve(self.dmL, kstar)
        self.covf.x1 = xstar
        self.covf.x2 = xstar
        covstar = self.covf.covfunc()
        stdev = sqrt(covstar - dot(transpose(v), v))
        return (mean, stdev)


    # train the hyperparameters
    def dm_hypertrain(self):
        # train the hyperparameters
        initheta = self.covf.theta
        if (self.grad == 'True'):
            if (self.prior != None and self.gradprior == None):
                raise RuntimeError("no gradprior given in " + 
                                   "grad_nlog_likelihood \n" + 
                                   "Possible solutions: \n" + 
                                   "(1) provide gradient of the prior, " + 
                                   "gradprior \n" + 
                                   "(2) set prior=None, i.e. no prior on" + 
                                   " the hyperparameters will be used \n" + 
                                   "(3) set grad='False', i.e. prior will" + 
                                   " be used, but Gaussian process is slower")
            bounds = self.covf.dmbounds()
            if (self.covf.scaletrain == 'False' and 
                self.covf.dscaletrain == 'False'):
                # all hyperparameters will be trained
                if (self.thetatrain == 'True'):
                    def logpfunc(theta):
                        self.set_theta(theta)
                        return self.dm_grad_nlog_likelihood()
                    theta = opt.fmin_tnc(logpfunc, initheta, bounds=bounds, 
                                         messages=8)[0]
                # some hyperparameters will be trained
                else:
                    # indices of the hyperparameters that are to be trained
                    indices = flatnonzero(self.thetatrain)
                    # array of the initial values of these hyperparameters
                    inith = take(initheta, indices)
                    bound = []
                    for i in range(len(indices)):
                        bound.append(bounds[indices[i]])
                    theta = initheta   # initialize theta
                    def logpfunc(th):
                        for i in range(len(indices)):
                            theta[indices[i]] = th[i]
                        self.set_theta(theta)
                        (logp, gradlogp) = self.dm_grad_nlog_likelihood()
                        gradlogp = take(gradlogp, indices)
                        return (logp, gradlogp)
                    th = opt.fmin_tnc(logpfunc, inith, bounds=bound,
                                      messages=8)[0]
                    for i in range(len(indices)):
                        theta[indices[i]] = th[i]
                self.set_theta(theta)
                print ("")
                print ("Optimized hyperparameters:")
                print ("theta = " + str(theta))
                return theta
            else:
                # scale and all hyperparameters will be trained
                if (self.thetatrain == 'True'):
                    # initheta_s contains initheta and (scale and/or dscale)
                    initheta_s = array(initheta)
                    if (self.covf.scaletrain == 'True'):
                        initheta_s = append(initheta_s, self.scale)
                    if (self.covf.dscaletrain == 'True'):
                        initheta_s = append(array(initheta_s), 
                                            self.covf.dscale)
                    sds = [self.scale, self.covf.dscale]
                    def logpfunc(theta_s): 
                        if (self.covf.scaletrain == 'True'):
                            sds[0] = theta_s[len(initheta)]
                        if (self.covf.dscaletrain == 'True'):
                            sds[1] = theta_s[len(theta_s) - 1]
                        self.set_scale(sds)
                        self.set_theta(resize(theta_s, len(initheta)))
                        return self.dm_grad_nlog_likelihood()
                    # determine theta containing s
                    theta_s = opt.fmin_tnc(logpfunc, initheta_s, bounds=bounds,
                                           messages=8)[0]
                    if (self.covf.scaletrain == 'True'):
                        sds[0] = theta_s[len(initheta)]
                    if (self.covf.dscaletrain == 'True'):
                        sds[1] = theta_s[len(theta_s) - 1]
                    self.set_scale(sds)
                    theta = resize(theta_s, len(initheta))
                    self.set_theta(theta)
                # only scale will be trained. hyperparameters are fixed
                elif (self.thetatrain == 'False'):
                    theta = array(self.covf.theta)
                    sc = []
                    bound = []
                    if (self.covf.scaletrain == 'True'):
                        sc.append(self.scale)
                        sb = (self.covf.iniscale/1.0e15, None)
                        bound.append(sb)
                    if (self.covf.dscaletrain == 'True'):
                        sb = (self.covf.inidscale/1.0e15, None)
                        bound.append(sb)
                        sc.append(self.covf.dscale)
                    sds = [self.scale, self.covf.dscale]
                    def logpfunc(sc):
                        if (self.covf.scaletrain == 'True'):
                            sds[0] = sc[0]
                        if (self.covf.dscaletrain == 'True'):
                            sds[1] = sc[len(sc) - 1]
                        self.set_scale(sds)
                        (logp, gradlogp) = self.dm_grad_nlog_likelihood()
                        gradlogp = gradlogp[len(theta):]
                        return (logp, gradlogp)
                    sc = opt.fmin_tnc(logpfunc, sc, bounds=bound, messages=8)[0]
                    sds = [self.scale, self.covf.dscale]
                    if (self.covf.scaletrain == 'True'):
                        sds[0] = sc[0]
                    if (self.covf.dscaletrain == 'True'):
                        sds[1] = sc[len(sc)-1]
                    self.set_scale(sds)
                 # scale and some hyperparameters will be trained
                else:
                    # indices of the hyperparameters that are to be trained
                    indices = flatnonzero(self.thetatrain)
                    indices_s = array(indices)
                    bound = []
                    for i in range(len(indices_s)):
                        bound.append(bounds[indices_s[i]])
                    if (self.covf.scaletrain == 'True'):
                        # add index for scale
                        indices_s = append(indices_s, len(initheta))
                        bound.append(bounds[len(initheta)])
                    if (self.covf.dscaletrain == 'True'):
                        # add index for dscale
                        indices_s = append(indices_s, len(initheta) + 1)
                        if (self.covf.scaletrain == 'True'):
                            bound.append(bounds[len(initheta) + 1])
                        else:
                            bound.append(bounds[len(initheta)])
                    # array of the initial values of these hyperparameters
                    initheta_s = append(initheta, (self.scale, 
                                                   self.covf.dscale))
                    inith_s = take(initheta_s, indices_s)
                    theta = array(self.covf.theta) 
                    sds = [self.scale, self.covf.dscale]
                    def logpfunc(th_s):
                        for i in range(len(indices)):
                            theta[indices[i]] = th_s[i]
                        self.set_theta(theta)
                        if (self.covf.scaletrain == 'True'):
                            if (self.covf.dscaletrain == 'True'):
                                sds[0] = th_s[len(th_s) - 2]
                            else:
                                sds[0] = th_s[len(th_s) - 1]
                        if (self.covf.dscaletrain == 'True'):
                            sds[1] = th_s[len(th_s) - 1]
                        self.set_scale(sds)
                        (logp, gradlogp) = self.dm_grad_nlog_likelihood()
                        gradlogp = append(take(gradlogp, indices), 
                                          gradlogp[len(initheta):])
                        return (logp, gradlogp)
                    th_s = opt.fmin_tnc(logpfunc, inith_s, bounds=bound,
                                        messages=8)[0]
                    for i in range(len(indices)):
                        theta[indices[i]] = th_s[i]
                    self.set_theta(theta)
                    sds = [self.scale, self.covf.dscale]
                    if (self.covf.scaletrain == 'True'):
                        if (self.covf.dscaletrain == 'True'):
                            sds[0] = th_s[len(th_s) - 2]
                        else:
                            sds[0] = th_s[len(th_s) - 1]
                    if (self.covf.dscaletrain == 'True'):
                        sds[1] = th_s[len(th_s) - 1]
                    self.set_scale(sds)
                print ("")
                print ("Optimized hyperparameters:")
                print ("theta = " + str(theta))
                print ("scale = "+str([self.scale, self.covf.dscale]))
                return (self.covf.theta, sds)
        else:
            constraints = self.covf.dmconstraints(self.thetatrain)
            if (self.covf.scaletrain == 'False' and 
                self.covf.dscaletrain == 'False'):
                # all hyperparameters will be trained
                if (self.thetatrain == 'True'):
                    def logpfunc(theta):
                        self.set_theta(theta)
                        return self.dm_nlog_likelihood()
                    theta = opt.fmin_cobyla(logpfunc, initheta, constraints)
                    self.set_theta(theta)

                # some hyperparameters will be trained
                else:
                    # indices of the hyperparameters that are to be trained
                    indices = flatnonzero(self.thetatrain)
                    # array of the initial values of these hyperparameters
                    inith = take(initheta, indices)
                    theta = initheta 
                    def logpfunc(th):
                        for i in range(len(indices)):
                            theta[indices[i]] = th[i]
                        self.set_theta(theta)
                        return self.dm_nlog_likelihood()
                    th = opt.fmin_cobyla(logpfunc, inith, constraints)
                    for i in range(len(indices)):
                        theta[indices[i]] = th[i]
                self.set_theta(theta)
                print ("")
                print ("Optimized hyperparameters:")
                print ("theta = " + str(theta))
                return theta
            else:
                # all hyperparameters and the scale will be trained
                if (self.thetatrain == 'True'):
                    # initheta_s contains initheta and (scale and/or dscale)
                    initheta_s = array(initheta)
                    if (self.covf.scaletrain == 'True'):
                        initheta_s = append(initheta_s, self.scale)
                    if (self.covf.dscaletrain == 'True'):
                        initheta_s = append(array(initheta_s), self.covf.dscale)
                    sds = [self.scale, self.covf.dscale]
                    def logpfunc(theta_s): 
                        if (self.covf.scaletrain == 'True'):
                            sds[0] = theta_s[len(initheta)]
                        if (self.covf.dscaletrain == 'True'):
                            sds[1] = theta_s[len(theta_s) - 1]
                        self.set_scale(sds)
                        self.set_theta(resize(theta_s, len(initheta)))
                        return self.dm_nlog_likelihood()
                    # determine theta containing s
                    theta_s = opt.fmin_cobyla(logpfunc, initheta_s, constraints,
                                              args=(), consargs=())
                    if (self.covf.scaletrain == 'True'):
                        sds[0] = theta_s[len(initheta)]
                    if (self.covf.dscaletrain == 'True'):
                        sds[1] = theta_s[len(theta_s) - 1]
                    self.set_scale(sds)
                    theta = resize(theta_s, len(initheta))
                    self.set_theta(theta)
                # only scale will be trained. hyperparameters are fixed
                elif (self.thetatrain == 'False'): 
                    theta = array(self.covf.theta)
                    sc = []
                    if (self.covf.scaletrain == 'True'):
                        sc.append(self.scale)
                    if (self.covf.dscaletrain == 'True'):
                        sc.append(self.covf.dscale)
                    sds = [self.scale, self.covf.dscale]
                    def logpfunc(sc):
                        if (self.covf.scaletrain == 'True'):
                            sds[0] = sc[0]
                        if (self.covf.dscaletrain == 'True'):
                            sds[1] = sc[len(sc) - 1]
                        self.set_scale(sds)
                        return self.dm_nlog_likelihood()
                    sc = opt.fmin_cobyla(logpfunc, sc, constraints, args=(),
                                         consargs=())
                    if (self.covf.scaletrain == 'True'):
                        sds[0] = sc[0]
                    if (self.covf.dscaletrain == 'True'):
                        sds[1] = sc[len(sc) - 1]
                    self.set_scale(sds)
                # some hyperparameters and the scale will be trained
                else:
                    # indices of the hyperparameters that are to be trained
                    indices = flatnonzero(self.thetatrain)
                    indices_s = array(indices)
                    if (self.covf.scaletrain == 'True'):
                        # add index for scale
                        indices_s = append(indices_s, len(initheta))
                    if (self.covf.dscaletrain == 'True'):
                        # add index for dscale
                        indices_s = append(indices_s, len(initheta) + 1)
                    # array of the initial values of these hyperparameters
                    initheta_s = append(initheta, (self.scale, 
                                                   self.covf.dscale))
                    inith_s = take(initheta_s, indices_s)
                    theta = array(self.covf.theta)
                    sds = [self.scale, self.covf.dscale]
                    def logpfunc(th_s):
                        for i in range(len(indices)):
                            theta[indices[i]] = th_s[i]
                        self.set_theta(theta)
                        if (self.covf.scaletrain == 'True'):
                            if (self.covf.dscaletrain == 'True'):
                                sds[0] = th_s[len(th_s) - 2]
                            else:
                                sds[0] = th_s[len(th_s) - 1]
                        if (self.covf.dscaletrain == 'True'):
                            sds[1] = th_s[len(th_s) - 1]
                        self.set_scale(sds)
                        return self.dm_nlog_likelihood()
                    th_s = opt.fmin_cobyla(logpfunc, inith_s, constraints)
                    for i in range(len(indices)):
                        theta[indices[i]] = th_s[i]
                    self.set_theta(theta)
                    sds = [self.scale,self.covf.dscale]
                    if (self.covf.scaletrain == 'True'):
                        if (self.covf.dscaletrain == 'True'):
                            sds[0] = th_s[len(th_s) - 2]
                        else:
                            sds[0] = th_s[len(th_s) - 1]
                    if (self.covf.dscaletrain == 'True'):
                        sds[1] = th_s[len(th_s) - 1]
                    self.set_scale(sds)
                print ("")
                print ("Optimized hyperparameters:")
                print ("theta = " + str(theta))
                print ("scale = " + str([self.scale, self.covf.dscale]))
                return (self.covf.theta, sds)





    # full Gaussian process run
    def dm_gp(self, unpack):
        # train the hyperparameters
        if (self.thetatrain != 'False' or self.covf.scaletrain == 'True' or 
            self.covf.dscaletrain == 'True'):
            self.dm_hypertrain()
        # reconstruct f(x)
        fmean_mu = zeros(self.nstar)
        fmean = zeros(self.nstar)
        fstd = zeros(self.nstar)
        for i in range(self.nstar):
            (fmean_mu[i], fstd[i]) = self.dm_prediction(self.Xstar[i, :])
        if (self.mu!=None):
            for i in range(self.nstar):
                fmean[i] = fmean_mu[i] + self.mu(self.Xstar[i], *self.muargs)
        else:
            fmean[:] = fmean_mu[:]
        self.fmean_mu = fmean_mu
        self.fstd_mu = fstd
        self.fmean = fmean
        self.fstd = fstd
        self.reconstruction = concatenate((self.Xstar,
                                           reshape(fmean, (self.nstar, 1)),
                                           reshape(fstd, (self.nstar, 1))),
                                          axis=1)
        if (self.covf.scale == None and self.covf.dscale == None):
            if (unpack == 'False'):
                return(self.reconstruction, self.covf.theta)
            else:
                return(self.Xstar, self.fmean, self.fstd, self.covf.theta)
        else:
            sds = [self.scale, self.covf.dscale]
            if (unpack == 'False'):
                return(self.reconstruction, self.covf.theta, sds)
            else:
                return(self.Xstar, self.fmean, self.fstd, self.covf.theta, sds)




################
##   DM_DGP   ##
################

    # dm_dgp: calculate vector of covariances kstar between one test point
    # and the n+dn training points
    def dm_d_covariance_vector(self, xstar):
        kstar = zeros(self.n + self.dn)
        self.covf.x2 = xstar
        for i in range(self.n):
            self.covf.x1 = self.X[i, :]
            kstar[i] = self.covf.dcovfunc()
        for i in range(self.dn):
            self.covf.x1 = self.dX[i]
            kstar[self.n + i] = self.covf.ddcovfunc()
        return kstar

    # calculate the predictive mean and standard deviation of (f'-mu') at 
    # test point xstar
    def dm_d_prediction(self, xstar):
        if (self.dmuptodate == 'False'):
            self.dm_input_covariance()
            self.dm_alpha_L()
            if (self.grad == 'True'):
                self.dm_grad_covariance()
            self.dmuptodate = 'True'
        if (self.dmalpha == None):
            raise RuntimeError("Invalid hyperparameters. Covariance matrix " +
                               "not positive definit.")
        # calculate covariance vector kstar
        kstar = self.dm_d_covariance_vector(xstar)
        # predictive mean
        mean = dot(transpose(kstar), self.dmalpha)
        # calculate predictive standard deviation
        v = linalg.solve(self.dmL, kstar)
        self.covf.x1 = xstar
        self.covf.x2 = xstar
        covstar = self.covf.ddcovfunc()
        stdev = sqrt(covstar - dot(transpose(v), v))
        return (mean, stdev)


    # full Gaussian process run
    def dm_dgp(self, theta=None, Xstar=None, cXstar=None, mu='False', 
               dmu='False', muargs=(), prior='False', gradprior=None,
               priorargs=(), thetatrain=None, scale='False', scaletrain=None,
               grad=None, unpack='False'):
        # train the hyperparameters
        if (self.thetatrain != 'False' or self.covf.scaletrain == 'True' or 
            self.covf.dscaletrain == 'True'):
            self.dm_hypertrain()
        # reconstruct f(x)
        dfmean_mu = zeros(self.nstar)
        dfmean = zeros(self.nstar)
        dfstd = zeros(self.nstar)
        for i in range(self.nstar):
            (dfmean_mu[i], dfstd[i]) = self.dm_d_prediction(self.Xstar[i, :])
        if (self.mu != None):
            for i in range(self.nstar):
                dfmean[i] = dfmean_mu[i] + self.dmu(self.Xstar[i], *self.muargs)
        else:
            dfmean[:] = dfmean_mu[:]
        self.dfmean_mu = dfmean_mu
        self.dfstd_mu = dfstd
        self.dfmean = dfmean
        self.dfstd = dfstd
        self.dreconstruction = concatenate((self.Xstar,
                                            reshape(dfmean, (self.nstar, 1)),
                                            reshape(dfstd, (self.nstar, 1))),
                                           axis=1)
        if (self.covf.scale == None and self.covf.dscale == None):
            if (unpack == 'False'):
                return(self.dreconstruction, self.covf.theta)
            else:
                return(self.Xstar, self.dfmean, self.dfstd, self.covf.theta)
        else:
            sds = [self.scale, self.covf.dscale]
            if (unpack == 'False'):
                return(self.dreconstruction, self.covf.theta, sds)
            else:
                return(self.Xstar, self.dfmean, self.dfstd, self.covf.theta, 
                       sds)


################
##   DM_D2GP  ##
################

    # calculate vector of covariances kstar between one test point
    # and the n+dn training points
    def dm_d2_covariance_vector(self, xstar):
        kstar = zeros(self.n + self.dn)
        self.covf.x2 = xstar
        for i in range(self.n):
            self.covf.x1 = self.X[i, :]
            kstar[i] = self.covf.d2covfunc()
        for i in range(self.dn):
            self.covf.x1 = self.dX[i]
            kstar[self.n + i] = self.covf.dd2covfunc()
        return kstar

    # calculate the predictive mean and standard deviation of (f''-mu'') at 
    # test point xstar
    def dm_d2_prediction(self, xstar):
        if (self.dmuptodate == 'False'):
            self.dm_input_covariance()
            self.dm_alpha_L()
            if (self.grad == 'True'):
                self.dm_grad_covariance()
            self.dmuptodate = 'True'
        if (self.dmalpha == None):
            raise RuntimeError("Invalid hyperparameters. Covariance matrix " +
                               "not positive definit")
        # calculate covariance vector kstar
        kstar = self.dm_d2_covariance_vector(xstar)
        # predictive mean
        mean = dot(transpose(kstar), self.dmalpha)
        # calculate predictive standard deviation
        v = linalg.solve(self.dmL, kstar)
        self.covf.x1 = xstar
        self.covf.x2 = xstar
        covstar = self.covf.d2d2covfunc()
        stdev = sqrt(covstar - dot(transpose(v), v))
        return (mean, stdev)


    # full Gaussian process run
    def dm_d2gp(self, theta=None, Xstar=None, cXstar=None, mu='False',
                dmu='False', d2mu='False', muargs=(), prior='False',
                gradprior=None, priorargs=(), thetatrain=None, scale='False',
                scaletrain=None, grad=None, unpack='False'):
        # train the hyperparameters
        if (self.thetatrain != 'False' or self.covf.scaletrain == 'True' or 
            self.covf.dscaletrain == 'True'):
            self.dm_hypertrain()
        # reconstruct f(x)
        d2fmean_mu = zeros(self.nstar)
        d2fmean = zeros(self.nstar)
        d2fstd = zeros(self.nstar)
        for i in range(self.nstar):
            (d2fmean_mu[i], d2fstd[i]) = self.dm_d2_prediction(self.Xstar[i, :])
        if (self.mu != None):
            for i in range(self.nstar):
                d2fmean[i] = d2fmean_mu[i] + self.d2mu(self.Xstar[i], 
                                                       *self.muargs)
        else:
            d2fmean[:] = d2fmean_mu[:]
        self.d2fmean_mu = d2fmean_mu
        self.d2fstd_mu = d2fstd
        self.d2fmean = d2fmean
        self.d2fstd = d2fstd
        self.d2reconstruction = concatenate((self.Xstar,
                                             reshape(d2fmean, (self.nstar, 1)),
                                             reshape(d2fstd, (self.nstar,1))),
                                            axis=1)
        if (self.covf.scale == None and self.covf.dscale == None):
            if (unpack == 'False'):
                return(self.d2reconstruction, self.covf.theta)
            else:
                return(self.Xstar, self.d2fmean, self.d2fstd, self.covf.theta)
        else:
            sds = [self.scale, self.covf.dscale]
            if (unpack == 'False'):
                return(self.d2reconstruction, self.covf.theta, sds)
            else:
                return(self.Xstar, self.d2fmean, self.d2fstd, self.covf.theta,
                       sds)



################
##   DM_D3GP  ##
################

    # calculate vector of covariances kstar between one test point
    # and the n+dn training points
    def dm_d3_covariance_vector(self, xstar):
        kstar = zeros(self.n + self.dn)
        self.covf.x2 = xstar
        for i in range(self.n):
            self.covf.x1 = self.X[i, :]
            kstar[i] = self.covf.d3covfunc()
        for i in range(self.dn):
            self.covf.x1 = self.dX[i]
            kstar[self.n + i] = self.covf.dd3covfunc()
        return kstar

    # calculate the predictive mean and standard deviation of (f'''-mu''') at 
    # test point xstar
    def dm_d3_prediction(self, xstar):
        if (self.dmuptodate == 'False'):
            self.dm_input_covariance()
            self.dm_alpha_L()
            if (self.grad == 'True'):
                self.dm_grad_covariance()
            self.dmuptodate = 'True'
        if (self.dmalpha == None):
            raise RuntimeError("Invalid hyperparameters. Covariance matrix " +
                               "not positive definit")
        # calculate covariance vector kstar
        kstar = self.dm_d3_covariance_vector(xstar)
        # predictive mean
        mean = dot(transpose(kstar), self.dmalpha)
        # calculate predictive standard deviation
        v = linalg.solve(self.dmL, kstar)
        self.covf.x1 = xstar
        self.covf.x2 = xstar
        covstar = self.covf.d3d3covfunc()
        stdev = sqrt(covstar - dot(transpose(v), v))
        return (mean, stdev)


    # full Gaussian process run
    def dm_d3gp(self, theta=None, Xstar=None, cXstar=None, mu='False',
                dmu='False', d3mu='False', muargs=(), prior='False',
                gradprior=None, priorargs=(), thetatrain=None, scale='False',
                scaletrain=None, grad=None, unpack='False'):
        # train the hyperparameters
        if (self.thetatrain != 'False' or self.covf.scaletrain == 'True' or 
            self.covf.dscaletrain == 'True'):
            self.dm_hypertrain()
        # reconstruct f(x)
        d3fmean_mu = zeros(self.nstar)
        d3fmean = zeros(self.nstar)
        d3fstd = zeros(self.nstar)
        for i in range(self.nstar):
            (d3fmean_mu[i], d3fstd[i]) = self.dm_d3_prediction(self.Xstar[i, :])
        if (self.mu != None):
            for i in range(self.nstar):
                d3fmean[i] = d3fmean_mu[i] + self.d3mu(self.Xstar[i], 
                                                       *self.muargs)
        else:
            d3fmean[:] = d3fmean_mu[:]
        self.d3fmean_mu = d3fmean_mu
        self.d3fstd_mu = d3fstd
        self.d3fmean = d3fmean
        self.d3fstd = d3fstd
        self.d3reconstruction = concatenate((self.Xstar,
                                             reshape(d3fmean, (self.nstar, 1)),
                                             reshape(d3fstd, (self.nstar, 1))),
                                            axis=1)
        if (self.covf.scale == None and self.covf.dscale == None):
            if (unpack == 'False'):
                return(self.d3reconstruction, self.covf.theta)
            else:
                return(self.Xstar, self.d3fmean, self.d3fstd, self.covf.theta)
        else:
            sds = [self.scale, self.covf.dscale]
            if (unpack == 'False'):
                return(self.d3reconstruction, self.covf.theta, sds)
            else:
                return(self.Xstar, self.d3fmean, self.d3fstd, self.covf.theta,
                       sds)



##################################################
## covariances between f(x) and its derivatives ##
##################################################

    # covariances between f and its derivatives at Xstar
    def dm_f_covariances(self, fclist=[0,1,2,3]):
        if (self.dmuptodate == 'False'):
            self.dm_input_covariance()
            self.dm_alpha_L()
            if (self.grad == 'True'):
                self.dm_grad_covariance()
            self.dmuptodate = 'True'
        fclist.sort()
        Xstar = self.Xstar
        nstar = self.nstar
        fcov = zeros((nstar, len(fclist), len(fclist)))
        for i in range(nstar):
            fcmatrix = zeros((4, 4))
            xstar = Xstar[i, :]
            if (0 in fclist):
                kstar = self.dm_covariance_vector(xstar)
                v = linalg.solve(self.dmL, kstar)
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                fcmatrix[0, 0] = self.f_var(v)
            if (1 in fclist):
                dkstar = self.dm_d_covariance_vector(xstar)
                dv = linalg.solve(self.dmL, dkstar)
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                fcmatrix[1, 1] = self.df_var(dv)
            if (2 in fclist):
                d2kstar = self.dm_d2_covariance_vector(xstar)
                d2v = linalg.solve(self.dmL, d2kstar)
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                fcmatrix[2, 2] = self.d2f_var(d2v)
            if (3 in fclist):
                d3kstar = self.dm_d3_covariance_vector(xstar)
                d3v = linalg.solve(self.dmL, d3kstar)
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                fcmatrix[3, 3] = self.d3f_var(d3v)
            if (0 in fclist and 1 in fclist):
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                fdf = self.f_df_cov(v, dv)
                fcmatrix[0, 1] = fdf
                fcmatrix[1, 0] = fdf
            if (0 in fclist and 2 in fclist):
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                fd2f = self.f_d2f_cov(v, d2v)
                fcmatrix[0, 2] = fd2f
                fcmatrix[2, 0] = fd2f
            if (0 in fclist and 3 in fclist):
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                fd3f = self.f_d3f_cov(v, d3v)
                fcmatrix[0, 3] = fd3f
                fcmatrix[3, 0] = fd3f
            if (1 in fclist and 2 in fclist):
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                dfd2f = self.df_d2f_cov(dv, d2v)
                fcmatrix[1, 2] = dfd2f
                fcmatrix[2, 1] = dfd2f
            if (1 in fclist and 3 in fclist):
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                dfd3f = self.df_d3f_cov(dv, d3v)
                fcmatrix[1, 3] = dfd3f
                fcmatrix[3, 1] = dfd3f
            if (2 in fclist and 3 in fclist):
                self.covf.x1 = xstar
                self.covf.x2 = xstar
                d2fd3f = self.d2f_d3f_cov(d2v, d3v)
                fcmatrix[2, 3] = d2fd3f
                fcmatrix[3, 2] = d2fd3f
            fcov[i, :, :] = take(take(fcmatrix, fclist, axis=0), fclist,
                                 axis=1)
        return fcov


