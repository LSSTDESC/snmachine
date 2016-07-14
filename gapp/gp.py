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




"""

Attributes of GaussianProcess:

covfunction    Covariance function
covf.theta     Vector of hyperparameters
X              Vector (or matrix) of the inputs
Y              Vector of observations of f(X)
Sigma          Data covariance matrix
Xstar          Points where the function is to be reconstructed
mu             A priori mean function that is subtracted from the data before 
               the GP is performed
muargs         List of arguments to be passed to mu
prior          Prior on the hyperparameters
gradprior      Gradient of the prior on the hyperparameters
priorargs      List of arguments to be passed to prior and gradprior
grad           'True' if the gradient of the covariance function is used for 
               the GP, 'False' else
thetatrain     Determines which hyperparameters are to be trained
scale          None if the covariance matrix not to be scaled, else a float 
               or int
K              Covariance matrix of the inputs
gradK          Gradient of the covariance matrix
A              K + Sigma
alpha          A^(-1)Y
L              Cholesky decomposition of A
uptodate       Specifies whether the values of (K, gradK, A, alpha, L) are up 
               to date
fmean          Mean value of the reconstructed function
fstd           Standard deviation of the reconstructed function
reconstruction Array containing Xstar, fmean and fstd
"""

import covariance
import numpy as np
from numpy import append, array, concatenate, diagonal, dot, eye, exp, \
    flatnonzero, loadtxt, log, mean, ones, pi, reshape, resize, shape, sign, \
    sqrt, std, take, trace, transpose, zeros, linalg
import scipy.optimize as opt
import warnings


class GaussianProcess(object):
    def __init__(self, X, Y, Sigma, covfunction=covariance.SquaredExponential,
                 theta=None, Xstar=None, cXstar=None, mu=None, muargs=(),
                 prior=None, gradprior=None, priorargs=(), thetatrain='True',
                 scale=None, scaletrain='True', grad='True'):
        # covariance function
        try:
            self.covnumber = len(covfunction)
        except:
            self.covnumber = 1
        if(self.covnumber == 1):
            if (theta == None):
                # theta determined automatically
                self.covf = covfunction(theta, X, Y)  
            else:
                self.covf = covfunction(theta)
        elif(self.covnumber == 2):
            self.covf = covariance.DoubleCovariance(covfunction[0], 
                                                    covfunction[1], theta, 
                                                    X, Y)
        else:
            raise AssertionError("Number of covariance functions is not supported.")
            
        # observational data
        self.set_data(X, Y, Sigma)
        # vector (or matrix) of the locations where f(x) is to be reconstructed
        if (Xstar != None): 
            self.set_Xstar(Xstar)
            if (cXstar != None):
                warnings.warn("Xstar and cXstar given in the " +
                              "initialization of GaussianProcess. \n" +
                              "cXstar will be ignored.")
        elif (cXstar != None):
            # create Xstar with cXstar = (xmin,xmax,nstar)
            xmin = cXstar[0]                    
            xmax = cXstar[1]
            nstar = cXstar[2]
            d = len(self.X[0, :])
            assert (shape(xmin) in [(), (1, ), (d, )]), \
                "xmin does not fit shape of X."
            assert (shape(xmax) in [(), (1, ), (d, )]), \
                "xmax does not fit shape of X."
            if(d > 1):
                if(xmin != None and shape(xmin) in [(), (1, )]):
                    xmin = xmin * ones(d)
                if(xmax != None and shape(xmax) in [(), (1, )]):
                    xmax = xmax * ones(d)
            if (xmin == None or xmax == None):
                self.auto_create_Xstar(xmin, xmax, nstar)
            else:
                self.create_Xstar(xmin, xmax, nstar)
        else:
            # automatically create Xstar
            self.auto_create_Xstar()
        # a priori mean function of f(x)
        self.set_mu(mu, muargs)
        # prior on the hyperparameters
        self.set_prior(prior, gradprior, priorargs)
        # scale of the measurement errors
        self.set_scale(scale)
        self.set_scaletrain(scaletrain)
        # 'True' if the gradient of the covariance function is used for the GP
        self.grad = grad
        # 'True', 'False' or a list with entries !=0 for hyperparameters that 
        # are to be trained
        self.set_thetatrain(thetatrain)

    # set observational data
    def set_data(self, X, Y, Sigma):
        n = len(X)
        assert (len(Y) == n and len(Sigma) == n), \
            "X, Y and Sigma must have the same length."
        if(shape(X) == (n,)):
            X = reshape(X, (n, 1))
        if(shape(X) == (n, 1)):
            # 1-dimensional data
            self.covf.multiD = 'False'
        else:
            # multi-dimensional data
            self.covf.multiD = 'True'
        self.X = array(X)
        self.Y = array(Y)
        if (shape(Sigma) == (n, n)):
            # data covariance matrix
            self.Sigma = array(Sigma)  
        elif (shape(Sigma) in [(n,), (n, 1)]):
            # turn vector into diagonal covariance matrix
            self.Sigma = Sigma * eye(n) * Sigma
        else:
            raise AssertionError("Sigma must be vector or nxn matrix.")
        # number of data points
        self.n = n
        try:
            if (self.dmu != None):
                self.subtract_dmu()
        except AttributeError:
            pass
        self.uptodate = 'False'
        self.dmuptodate = 'False'
            



    # set hyperparameter theta
    def set_theta(self, theta):
        self.covf.theta = array(theta)
        self.uptodate = 'False'
        self.dmuptodate = 'False'
        if(self.covnumber == 2):
            self.covf.covf1.theta = theta[:self.covf.lth1]
            self.covf.covf2.theta = theta[self.covf.lth1:]

    # subtract the a priori mean from the data
    def subtract_mu(self):
        self.Y_mu = zeros(self.n)
        for i in range(self.n):
            self.Y_mu[i] = self.Y[i] - self.mu(self.X[i], *self.muargs)
        self.uptodate = 'False'
        self.dmuptodate = 'False'

    # set an a priori mean function
    def set_mu(self, mu, muargs=()):
        if (mu == None):
            self.mu = None
            self.Y_mu = self.Y[:]
            self.muargs = ()
        else:
            self.mu = mu
            try:
                len(muargs)
            except TypeError:
                muargs = (muargs,)
            self.muargs = muargs
            self.subtract_mu()
        self.uptodate = 'False'
        self.dmuptodate = 'False'

    # unset the a priori mean function
    def unset_mu(self):
        self.mu = None
        self.dmu = None
        self.d2mu = None
        self.d3mu = None
        self.Y_mu = self.Y[:]
        self.dY_dmu = self.dY[:]
        self.muargs = ()
        self.uptodate = 'False'
        self.dmuptodate = 'False'

    # set prior for the hyperparameters
    def set_prior(self, prior, gradprior=None, priorargs=()):
        self.prior = prior
        try:
            len(priorargs)
        except TypeError:
            priorargs = (priorargs,)
        self.priorargs = priorargs
        if (gradprior == None):
            self.gradprior = None
        elif (len(gradprior(self.covf.initheta)) == len(self.covf.initheta)):
            self.gradprior = gradprior
        else:
            warnings.warn("Wrong data type in gradprior. \n" +
                          "gradprior(theta) must return array of " +
                          "length theta. \n" +
                          "gradprior will be ignored.")
            self.gradprior = None
        self.uptodate = 'False'
        self.dmuptodate = 'False'

    # unset prior for the hyperparameters
    def unset_prior(self):
        self.prior = None
        self.priorargs = ()
        self.gradprior = None
        self.uptodate = 'False'
        self.dmuptodate = 'False'

    # set scale of the measurement errors
    def set_scale(self, scale):
        self.uptodate = 'False'
        self.dmuptodate = 'False'
        if (scale == None):
            self.scale = None
            self.covf.scale = None
            self.covf.iniscale = None
            self.covf.dscale = None
            self.covf.inidscale = None
            self.set_scaletrain('False')
        elif (len(array([scale])) == 1):
            self.scale = scale
            self.covf.scale = scale
            self.covf.iniscale = scale
            self.covf.dscale = None
            self.covf.inidscale = None
        elif(len(scale) == 2):
            self.scale = scale[0]
            self.covf.scale = scale[0]
            self.covf.iniscale = scale[0]
            self.dscale = scale[1]
            self.covf.dscale = scale[1]
            self.covf.inidscale = scale[1]

    # define which values of scale are to be trained
    def set_scaletrain(self, scaletrain):
        if (scaletrain in ['True', 'False']):
            if (self.covf.scale == None): 
                self.covf.scaletrain = 'False'
            else: 
                self.covf.scaletrain = scaletrain
            if (self.covf.dscale == None): 
                self.covf.dscaletrain = 'False'
            else: 
                self.covf.dscaletrain = scaletrain
        elif (len(scaletrain) == 2):
            if (self.covf.scale == None): 
                self.covf.scaletrain = 'False'
            else:
                if (scaletrain[0] == 0):
                    self.covf.scaletrain = 'False'
                else:
                    self.covf.scaletrain = 'True'
            if (self.covf.dscale == None): 
                self.covf.dscaletrain = 'False'
            else:
                if (scaletrain[1] == 0):
                    self.covf.dscaletrain = 'False'
                else:
                    self.covf.dscaletrain = 'True'

        
        
    # unset scale of the measurement errors
    def unset_scale(self):
        self.scale = None
        self.covf.scale = None
        self.covf.iniscale = None
        self.covf.dscale = None
        self.covf.inidscale = None
        self.set_scaletrain('False')
        self.uptodate = 'False'
        self.dmuptodate = 'False'

    # set which of the hyperparameters are to be trained
    def set_thetatrain(self, thetatrain):
        if (thetatrain in ['False', 'True']):
            self.thetatrain = thetatrain
        elif (len(thetatrain) == len(self.covf.theta)):
            if (np.all(thetatrain)):
                self.thetatrain = 'True'
            elif (np.any(thetatrain) == False):
                self.thetatrain = 'False'
            else:
                self.thetatrain = array(thetatrain)
        else:
            raise TypeError("Wrong data type in thetatrain.")

    # gradient of the covariance function will be used for the GP
    def set_grad(self, grad='True'):
        self.grad = grad
        self.uptodate = 'False'
        self.dmuptodate = 'False'

    # gradient of the covariance function will not be used for the GP
    def unset_grad(self):
        self.grad = 'False'
        self.dmuptodate = 'False'


    # set Xstar
    def set_Xstar(self, Xstar):
        if (shape(Xstar) == ()):
            Xstar = reshape(Xstar, (1, 1))
        nstar = len(Xstar)
        if(shape(Xstar) == (len(Xstar),)):
            Xstar = reshape(Xstar, (nstar, 1))
        self.Xstar = Xstar
        self.nstar = nstar


    # create vector Xstar with nstar values between xmin and xmax
    def create_Xstar(self, xmin, xmax, nstar):
        if (xmin == None or xmax == None):
            self.auto_create_Xstar(xmin, xmax, nstar)
        else:
            if (shape(xmin) in [(), (1, ), (1, 1)]):
                Xstar = zeros(nstar)
                self.nstar = nstar
                for i in range(nstar):
                    Xstar[i] = xmin + i * (xmax - xmin)/float(nstar - 1)
                self.Xstar = reshape(Xstar, (nstar, 1))
            else:
                Nstar = nstar * ones((len(xmin)), dtype=np.int)
                self.create_md_Xstar(xmin, xmax, Nstar)

    # create vector Xstar with nstar values. xmin and xmax are determined
    # automatically.
    # X is the array of x values from the data set
    def auto_create_Xstar(self, xmin=None, xmax=None, nstar=200):
        xmi = np.min(self.X, axis=0)
        xma = np.max(self.X, axis=0)
        diff = xma - xmi
        if (xmin == None):
            xmin = xmi - diff/10.
        if (xmax == None):
            xmax = xma + diff/10.
        if (shape(self.X) in [(self.n, ), (self.n, 1)]):
            self.create_Xstar(xmin, xmax, nstar)
        else:
            Nstar = nstar * ones((len(xmin)), dtype=np.int)
            self.create_md_Xstar(xmin, xmax, Nstar)

    # create multi-D Xstar with Nstar[i] values between xmin[i] and xmax[i]
    def create_md_Xstar(self, xmin, xmax, Nstar):
        D = len(xmin)
        self.nstar = np.prod(Nstar)
        Xstar = zeros((self.nstar, D))
        self.k = 0
        def xsloop(d):
            for i in range(Nstar[d]):
                ul = self.k + int(np.prod(Nstar[d+1:D]))
                Xstar[self.k:ul, d] = xmin[d] + \
                    i * (xmax[d] - xmin[d])/float(Nstar[d] - 1)
                if ((d + 1) < D):
                    xsloop(d + 1)
                else:
                    self.k += 1
        xsloop(0)
        self.Xstar = Xstar


    # calculate covariance matrix of the inputs
    def input_covariance(self):
        K = zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if (j >= i):
                    self.covf.x1 = self.X[i, :]
                    self.covf.x2 = self.X[j, :]
                    K[i, j] = self.covf.covfunc()
                else:
                    K[i, j] = K[j, i]
        self.K = K

    # calculate the gradient of the covariance matrix with respect to theta
    def grad_covariance(self):
        gradK = zeros((len(self.covf.theta), self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if (j >= i):
                    self.covf.x1 = self.X[i, :]
                    self.covf.x2 = self.X[j, :]
                    gradK[:, i, j] = self.covf.gradcovfunc()
                else:
                    gradK[:, i, j] = gradK[:, j, i]
        if (self.covf.scaletrain == 'True'):
            gradscale = zeros((1, self.n, self.n))
            for i in range(self.n):
                gradscale[0, i, i] = 2 * self.scale * self.Sigma[i, i]
            gradK = concatenate((gradK,gradscale))
        self.gradK = gradK

    # calculates alpha = (K + sigma)^{-1}Y_mu 
    # and the Cholesky decomposition L
    def alpha_L(self):
        A = array(self.K)
        self.A = A
        if (self.scale == None):
            A[:, :] = self.K[:, :] + self.Sigma[:, :]
        else:
            A[:, :] = self.K[:, :] + self.scale**2 * self.Sigma[:, :]
        # calculate alpha = A^{-1}Y_mu using a Cholesky decomposition
        try:
            L = linalg.cholesky(A)
            b = linalg.solve(L, self.Y_mu)
            self.alpha = linalg.solve(transpose(L), b)
            self.L = L
        except np.linalg.linalg.LinAlgError: 
            # A not positive definit
            self.alpha = None
            self.L = None

    # calculate vector of covariances kstar between one test point
    # and the n training points
    def covariance_vector(self, xstar):
        kstar = zeros(self.n)
        self.covf.x2 = xstar
        for i in range(self.n):
            self.covf.x1 = self.X[i, :]
            kstar[i] = self.covf.covfunc()
        return kstar


    def log_likelihood(self, theta=None, mu='False', muargs=(), prior='False',
                       priorargs=(), scale='False'):
        # set new attributes
        if (theta != None): 
            self.set_theta(theta)
        if (mu != 'False'): 
            self.set_mu(mu,muargs)
        if (prior != 'False'): 
            self.set_prior(prior, self.gradprior, priorargs)
        if (scale != 'False'): 
            self.set_scale(scale)
        return (-self.nlog_likelihood())

    # calculate the negative log marginal likelihood -log p(y|X,theta) 
    def nlog_likelihood(self):
        if (self.uptodate == 'False'):
            self.input_covariance()
            self.alpha_L()
            if (self.grad == 'True'):
                self.grad_covariance()
            self.uptodate = 'True'
        # log likelihood of prior
        if (self.prior != None):
            priorp = self.prior(self.covf.theta, *self.priorargs)
            if (priorp < 0.0):
                warnings.warn("Invalid prior. Negative prior will " +
                              "be treated as prior=0.")
                return 1.0e+20
            if (priorp == 0.0):
                return 1.0e+20
            priorlogp = log(priorp)
        else:
            priorlogp = 0.0
        # calculate the negative log marginal likelihood
        if (self.alpha == None):
            logp = 1.0e+20 - priorlogp
        else:
            logp = -(-0.5 * dot(transpose(self.Y_mu), self.alpha) - 
                      np.sum(log(diagonal(self.L))) - self.n/2 * log(2*pi) + 
                      priorlogp)
        return logp

    # calculate the negative log marginal likelihood -log p(y|X,theta) 
    # and its gradient with respect to theta
    def grad_nlog_likelihood(self):
        if (self.uptodate == 'False'):
            self.input_covariance()
            self.alpha_L()
            self.grad_covariance()
            self.uptodate = 'True'
        logp = self.nlog_likelihood()
        if (self.alpha == None):
            try:
                self.gradlogp = 0.9 * self.gradlogp
                return (logp, array(-self.gradlogp))
            except:
                raise RuntimeError('invalid hyperparameters; ' + 
                                   'covariance matrix not positive definit')
        # number of hyperparameters (plus 1 if scale is trained)
        nh = len(self.gradK)
        # calculate trace of alpha*alpha^T gradK
        traaTgradK = zeros(nh)
        for t in range(nh):
            aTgradK = zeros(self.n)
            for i in range(self.n):
                aTgradK[i] = np.sum(self.alpha[:] * self.gradK[t, :, i])
            traaTgradK[t] = np.sum(self.alpha[:] * aTgradK[:])
        # calculate trace of A^{-1} gradK
        invL = linalg.inv(self.L)
        invA = dot(transpose(invL), invL)
        trinvAgradK = zeros(nh)
        for t in range(nh):
            for i in range(self.n):
                trinvAgradK[t] = trinvAgradK[t] + np.sum(invA[i, :] * 
                                                         self.gradK[t, :, i])
        # gradient of the prior log likelihood
        gradpriorlogp = zeros(nh)
        if (self.gradprior != None):
            gradpriorp = self.gradprior(self.covf.theta, *self.priorargs)
            if (self.prior == None):
                warnings.warn('no prior given in gp.grad_nlog_likelihood;'
                              + ' gradprior will be ignored')
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
        gradlogp = array(-0.5 * (traaTgradK[:] - trinvAgradK[:]) - gradpriorlogp)
        self.gradlogp = gradlogp
        return(logp, gradlogp)

    # calculate the predictive mean and standard deviation of (f-mu) at
    # test point xstar
    def prediction(self, xstar):
        if (self.uptodate == 'False'):
            self.input_covariance()
            self.alpha_L()
            if (self.grad == 'True'):
                self.grad_covariance()
            self.uptodate = 'True'
        if (self.alpha == None):
            raise RuntimeError('invalid hyperparameters; ' + 
                               'covariance matrix not positive definit')
        # calculate covariance vector kstar
        kstar = self.covariance_vector(xstar)
        # predictive mean
        mean = dot(transpose(kstar), self.alpha)
        # calculate predictive standard deviation
        v = linalg.solve(self.L, kstar)
        self.covf.x1 = xstar
        self.covf.x2 = xstar
        covstar = self.covf.covfunc()
        stdev = sqrt(covstar - dot(transpose(v), v))
        return(mean, stdev)

    # train the hyperparameters
    def hypertrain(self, covfunction=None, theta=None, mu='False', muargs=(),
                   prior='False', gradprior=None, priorargs=(),
                   thetatrain=None, scale='False', scaletrain=None, grad=None):
        # set new attributes
        if (theta != None): 
            self.set_theta(theta)
        if (covfunction != None): 
            self.set_covfunction(covfunction)
        if (mu != 'False'): 
            self.set_mu(mu,muargs)
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
                               "i.e. no hyperparameters are to be trained.")
        return(self.fhypertrain())

    def fhypertrain(self):
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
            bounds = self.covf.bounds()
            if (self.covf.scaletrain == 'False'):
                # all hyperparameters will be trained
                if (self.thetatrain == 'True'):
                    def logpfunc(theta):
                        self.set_theta(theta)
                        return self.grad_nlog_likelihood()
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
                        (logp,gradlogp) = self.grad_nlog_likelihood()
                        gradlogp = take(gradlogp,indices)
                        return (logp, gradlogp)
                    th = opt.fmin_tnc(logpfunc, inith, bounds=bound, 
                                      messages=8)[0]
                    for i in range(len(indices)):
                        theta[indices[i]] = th[i]
                print ("")
                print ("Optimized hyperparameters:")
                print ("theta = " + str(theta))
                return theta
            else:
                # scale and all hyperparameters will be trained
                if (self.thetatrain == 'True'):
                    # initheta_s contains initheta and s
                    initheta_s = append(array(initheta), self.scale)
                    def logpfunc(theta_s): # theta_s contains theta and s
                        sds = (theta_s[len(theta_s)-1], self.covf.dscale)
                        self.set_scale(sds)
                        theta = resize(theta_s, ((len(theta_s) - 1), ))
                        self.set_theta(theta)
                        return self.grad_nlog_likelihood()
                    # determine theta containing s
                    theta_s = opt.fmin_tnc(logpfunc, initheta_s, 
                                           bounds=bounds, messages=8)[0]
                    theta = resize(theta_s, ((len(theta_s) - 1), ))
                    sds = (theta_s[len(theta_s) - 1], self.covf.dscale)
                    self.set_scale(sds)
                    self.set_theta(theta)
                # only scale will be trained. hyperparameters are fixed
                elif (self.thetatrain == 'False'): 
                    bound = ((self.covf.iniscale/1.0e15, None), )
                    iniscale = (self.scale, )
                    theta = array(initheta)
                    def logpfunc(scale):
                        sds = (scale,self.covf.dscale)
                        self.set_scale(sds)
                        (logp, gradlogp) = self.grad_nlog_likelihood()
                        gradlogp = gradlogp[len(initheta):]
                        return (logp, gradlogp)
                    scale = float(opt.fmin_tnc(logpfunc, iniscale, 
                                               bounds=bound, messages=8)[0])
                    sds = (scale, self.covf.dscale)
                    self.set_scale(sds)
                 # scale and some hyperparameters will be trained
                else:
                    # indices of the hyperparameters that are to be trained
                    indices = flatnonzero(self.thetatrain)
                    # add index for scale
                    indices_s = append(indices, len(initheta))
                    # array of the initial values of these hyperparameters
                    inith_s = take(append(initheta, self.scale), indices_s)
                    bound = []
                    for i in range(len(indices_s)):
                        bound.append(bounds[indices_s[i]])
                    theta = array(initheta)
                    def logpfunc(th_s):
                        for i in range(len(indices)):
                            theta[indices[i]] = th_s[i]
                        self.set_theta(theta)
                        sds = (th_s[len(th_s) - 1], self.covf.dscale)
                        self.set_scale(sds)
                        (logp,gradlogp) = self.grad_nlog_likelihood()
                        gradlogp = take(gradlogp, indices_s)
                        return (logp,gradlogp)
                    th_s = opt.fmin_tnc(logpfunc, inith_s, bounds=bound, 
                                        messages=8)[0]
                    for i in range(len(indices)):
                        theta[indices[i]] = th_s[i]
                    self.set_theta(theta)
                    sds = (th_s[len(th_s) - 1], self.covf.dscale)
                    self.set_scale(sds)
                print ("")
                print ("Optimized hyperparameters:")
                print ("theta = " + str(theta))
                print ("scale = " + str(self.scale))
                return (self.covf.theta, self.scale)
        else:
            if (self.covf.scaletrain == 'False'):
                # all hyperparameters will be trained
                if (self.thetatrain == 'True'):
                    constraints = self.covf.constraints()
                    def logpfunc(theta):
                        self.set_theta(theta)
                        return(self.nlog_likelihood())
                    theta = opt.fmin_cobyla(logpfunc, initheta, constraints)
                # some hyperparameters will be trained
                else:
                    constraints = self.covf.constraints(self.thetatrain)
                    # indices of the hyperparameters that are to be trained
                    indices = flatnonzero(self.thetatrain)
                    # array of the initial values of these hyperparameters
                    inith = take(initheta, indices)
                    theta = initheta   # initialize theta
                    def logpfunc(th):
                        for i in range(len(indices)):
                            theta[indices[i]] = th[i]
                        self.set_theta(theta)
                        return self.nlog_likelihood()
                    th = opt.fmin_cobyla(logpfunc, inith, constraints)
                    for i in range(len(indices)):
                        theta[indices[i]] = th[i]
                self.set_theta(theta)
                print ("")
                print ("Optimized hyperparameters:")
                print ("theta = " + str(theta))
                return(theta)
            else:
                # all hyperparameters and the scale will be trained
                if (self.thetatrain == 'True'):
                    constraints = self.covf.constraints()
                    # initheta_s contains initheta and s
                    initheta_s = append(array(initheta), self.scale)
                    def logpfunc(theta_s): # theta_s contains theta and s
                        sds = (theta_s[len(theta_s) - 1], self.covf.dscale)
                        self.set_scale(sds)
                        self.set_theta(resize(theta_s, len(theta_s) - 1))
                        return self.nlog_likelihood()
                    # determine theta containing s
                    theta_s = opt.fmin_cobyla(logpfunc, initheta_s, 
                                              constraints, args=(), consargs=())
                    sds = (theta_s[len(theta_s) - 1], self.covf.dscale)
                    theta = resize(theta_s, len(theta_s) - 1)
                    self.set_scale(sds)
                    self.set_theta(theta)
                # only scale will be trained. hyperparameters are fixed
                elif (self.thetatrain == 'False'): 
                    theta = array(self.covf.theta)
                    def constr(scale):
                        return float(scale - self.covf.iniscale/1.0e15)
                    constraints = (constr, )
                    def logpfunc(scale):
                        sds = (scale, self.covf.dscale)
                        self.set_scale(sds)
                        return self.nlog_likelihood()
                    scale = opt.fmin_cobyla(logpfunc, self.covf.scale,
                                            constraints, args=(), consargs=())
                    sds = (scale, self.covf.dscale)
                    self.set_scale(sds)
                # some hyperparameters and the scale will be trained
                else:
                    constraints = self.covf.constraints(self.thetatrain)
                    # indices of the hyperparameters that are to be trained
                    indices = flatnonzero(self.thetatrain)
                    # add index for scale
                    indices_s = append(indices, len(initheta))
                    # array of the initial values of these hyperparameters
                    inith_s = take(append(initheta, self.scale), indices_s)
                    theta = array(self.covf.theta)
                    def logpfunc(th_s):
                        for i in range(len(indices)):
                            theta[indices[i]] = th_s[i]
                        self.set_theta(theta)
                        sds = (th_s[len(th_s)-1], self.covf.dscale)
                        self.set_scale(sds)
                        return self.nlog_likelihood()
                    th_s = opt.fmin_cobyla(logpfunc, inith_s, constraints)
                    for i in range(len(indices)):
                        theta[indices[i]] = th_s[i]
                    self.set_theta(theta)
                    sds = (th_s[len(th_s) - 1], self.covf.dscale)
                    self.set_scale(sds)
                print ("")
                print ("Optimized hyperparameters:")
                print ("theta = " + str(theta))
                print ("scale = " + str(self.scale))
                return (self.covf.theta, self.scale)


    def gp(self, theta=None, Xstar=None, cXstar=None, mu='False', muargs=(),
           prior='False', gradprior=None, priorargs=(), thetatrain=None,
           scale='False', scaletrain=None, grad=None, unpack='False'):
        # set new attributes
        if (theta != None): 
            self.set_theta(theta)
        if (Xstar != None): 
            self.set_Xstar(Xstar)
            if (cXstar != None):
                warnings.warn("Xstar and cXstar given in gp. " + 
                              "cXstar will be ignored.")
        elif (cXstar != None): 
            self.create_Xstar(cXstar[0], cXstar[1], cXstar[2])
        if (mu != 'False'): 
            self.set_mu(mu,muargs)
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
        return(self.fgp(unpack=unpack))

    # full Gaussian process run
    def fgp(self, unpack):
        # train the hyperparameters
        if (self.thetatrain != 'False' or self.covf.scaletrain == 'True'):
            self.hypertrain()
        # reconstruct f(x)
        fmean_mu = zeros(self.nstar)
        fmean = zeros(self.nstar)
        fstd = zeros(self.nstar)
        for i in range(self.nstar):
            (fmean_mu[i], fstd[i]) = self.prediction(self.Xstar[i, :])
        if (self.mu != None):
            for i in range(self.nstar):
                fmean[i] = fmean_mu[i] + self.mu(self.Xstar[i, :], 
                                                 *self.muargs)
        else:
            fmean = fmean_mu[:]
        self.fmean_mu = fmean_mu
        self.fstd_mu = fstd
        self.fmean = fmean
        self.fstd = fstd
        self.reconstruction = concatenate((self.Xstar, 
                                           reshape(fmean, (self.nstar, 1)),
                                           reshape(fstd, (self.nstar, 1))), 
                                          axis=1)
        if (self.scale == None):
            if (unpack == 'False'):
                return(self.reconstruction, self.covf.theta)
            else:
                return(self.Xstar, self.fmean, self.fstd, self.covf.theta)
        else:
            if (unpack == 'False'):
                return(self.reconstruction, self.covf.theta, self.scale)
            else:
                return(self.Xstar, self.fmean, self.fstd, self.covf.theta, 
                       self.scale)

