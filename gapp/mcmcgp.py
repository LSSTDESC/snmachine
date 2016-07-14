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




import gp, covariance
import numpy as np
from numpy import array, concatenate, ones, random, reshape, shape, zeros
import multiprocessing


def mcmc_log_likelihood(th, sc0,  X, Y_mu, Sigma, covfunction, prior, 
                        priorargs):
    try:
        if (np.min(th) < 0.0):
            return np.NINF
        if (sc0 == False):
            theta = th
            scale = 1
        else:
            theta = th[:-1]
            scale = th[-1]
        g = gp.GaussianProcess(X, Y_mu, Sigma, covfunction, theta, prior=prior, 
                               priorargs=priorargs, scale=scale)
        logp = g.log_likelihood()
        return logp
    except KeyboardInterrupt:
        return


def recthread(i, th, sc0, X, Y, Sigma, covfunction, Xstar, mu, muargs):
    try:
        if (sc0 == False):
            theta = th
            scale = 1
        else:
            theta = th[:-1]
            scale = th[-1]
        g = gp.GaussianProcess(X, Y, Sigma, covfunction, theta, Xstar, mu=mu, 
                               muargs=muargs, thetatrain='False', scale=scale, 
                               scaletrain='False')
        nstar = len(Xstar)
        (fmean, fstd) = g.gp(unpack='True')[1:3]
        pred = concatenate((reshape(fmean, (nstar, 1)), 
                            reshape(fstd, (nstar, 1))), axis=1)
        return (i, pred)
    except KeyboardInterrupt:
        return

def recarray(j, recj, k, nsample):
    try:
        rarr = array([])
        for i in range(len(recj)):
            if (recj[i, 1] > 0):
                rarr = concatenate((rarr, random.normal(recj[i, 0], 
                                                        recj[i, 1], 
                                                        k[i] * nsample)))
            else:
                rarr = concatenate((rarr, recj[i, 0] * ones(k[i] * nsample)))
        return (j, rarr)
    except KeyboardInterrupt:
        return


class MCMCGaussianProcess(gp.GaussianProcess):
    def __init__(self, X, Y, Sigma, theta0, Niter=100,
                 covfunction=covariance.SquaredExponential,
                 Xstar=None, cXstar=None, mu=None, muargs=(), prior=None, 
                 priorargs=(), scale0=None, a=2.0, threads=1, nacor=10,
                 nsample=50, sampling='True'):


        if (scale0 != None):
            assert (len(theta0) == len(scale0)) ,\
                "Lengths of theta0 and scale0 must be identical."
            self.pos = concatenate((theta0, reshape(scale0, (len(scale0), 1))), 
                                   axis=1)
            self.sc0 = True
            scale = scale0[0]
        else:
            self.pos = theta0
            self.sc0 = False
            scale = None

        gp.GaussianProcess.__init__(self, X, Y, Sigma, covfunction, 
                                    theta0[0,:], Xstar, cXstar, mu, muargs,
                                    prior, gradprior=None, priorargs=priorargs,
                                    thetatrain='False', scale=scale, 
                                    scaletrain='False')
        self.theta0 = theta0
        self.scale0 = scale0
        self.covfunction = covfunction
        self.Niter = Niter
        self.a = a
        self.threads = threads
        self.nacor = nacor
        self.nsample = nsample
        self.sampling = sampling
        (self.nwalkers, self.ndim) = shape(self.pos)

        if (sampling == 'True'):
            try:
                import emcee
            except ImportError:
                print("Error: MCMCGaussianProcess requires the python package emcee.")
                print("emcee can be installed from http://github.com/dfm/emcee")
                raise SystemExit
            try:
                import acor
            except ImportError:
                print("Error: MCMCGaussianProcess requires the python package acor.")
                print("acor can be installed from http://github.com/dfm/acor")
                raise SystemExit

            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, 
                                                 mcmc_log_likelihood, 
                                                 args=(self.sc0, self.X, self.Y_mu, 
                                                       self.Sigma, covfunction,
                                                       prior, priorargs),
                                                 a=a, threads=threads)



    def mcmc_sampling(self):
        print("start burn-in")
        (pos, prob, state) = self.sampler.run_mcmc(self.pos, 10)
        try:
            maxa = max(self.sampler.acor)
            if (shape(self.sampler.chain)[1] < self.nacor * maxa):
                c = True
            else:
                c = False
        except RuntimeError:
            c = True
        while (c):
            (pos, prob, state) = self.sampler.run_mcmc(pos, 1, rstate0=state)
            try:
                maxa = max(self.sampler.acor)
                if (shape(self.sampler.chain)[1] >= self.nacor * maxa):
                    c = False
            except RuntimeError:
                pass
        if (self.sc0 == False):
            self.theta0 = pos
        else:
            self.scale0 = pos[:, -1]
            self.theta0 = pos[:, :-1]
        print("burn-in finished")
        print("number of burn-in steps: " + str(shape(self.sampler.chain)[1]))
        print("autocorrelation time: " + str(maxa))
        print("acceptance fraction: " + 
              str(np.mean(self.sampler.acceptance_fraction)))
        self.sampler.reset()
        (pos, prob, state) = self.sampler.run_mcmc(pos, self.Niter, 
                                                   rstate0=state)
        self.possample = self.sampler.flatchain
        if (self.sc0 == False):
            self.thetasample = self.possample
            self.scalesample = None
        else:
            self.thetasample = self.possample[:, :-1]
            self.scalesample = self.possample[:, -1]


    def mcmcgp(self):
        if (self.sampling == 'True'):
            self.mcmc_sampling()
        else:
            self.possample = self.pos
            self.thetasample = self.theta0
            self.scalesample = self.scale0
        redpossample = []
        k = []
        for i in range(len(self.possample)):
            if (i > 0 and all(self.possample[i, :] == self.possample[i-1, :])):
                k[-1] += 1
            else:
                redpossample.append(self.possample[i, :])
                k.append(1)
        redpossample = array(redpossample)
        if (self.threads == 1):
            self.serialrec(redpossample, k)
        else:
            self.parallelrec(redpossample, k)
        return (self.Xstar, self.reconstruction)



    def serialrec(self, redpossample, k):
        rec = zeros((len(redpossample), self.nstar, 2))
        for i in range(len(redpossample)):
            if (self.sc0 == False):
                self.set_theta(redpossample[i, :])
            else:
                self.set_theta(redpossample[i, :-1])
                self.set_scale(redpossample[i, -1])
            (fmean, fstd) = self.gp(unpack='True')[1:3]
            rec[i, :, 0] = fmean[:]
            rec[i, :, 1] = fstd[:]
        reconstruction = zeros((self.nstar, len(self.possample) * self.nsample))
        for j in range(self.nstar):
            rarr = array([])
            for i in range(len(rec)):
                if (rec[i, j, 1] > 0):
                    rarr = concatenate((rarr, random.normal(rec[i, j, 0], 
                                                            rec[i, j, 1], 
                                                            k[i] * self.nsample)))
                else:
                    rarr = concatenate((rarr, rec[i, j, 0] * 
                                        ones(k[i] * self.nsample)))
            reconstruction[j, :] = rarr[:]
        self.reconstruction = reconstruction



    def parallelrec(self, redpossample, k):
        pool = multiprocessing.Pool(processes=self.threads)
        recres = [pool.apply_async(recthread, (i, redpossample[i, :], self.sc0, 
                                               self.X, self.Y, self.Sigma, 
                                               self.covfunction, self.Xstar, 
                                               self.mu, self.muargs)) 
                  for i in range(len(redpossample))]
        rec = zeros((len(redpossample), self.nstar, 2))
        for r in recres:
            a = r.get()
            rec[a[0], :, :] = a[1]
        reconstruction = zeros((self.nstar, len(self.possample) * self.nsample))
        recon = [pool.apply_async(recarray, (j, rec[:, j, :], k, self.nsample)) 
                 for j in range(self.nstar)]
        for r in recon:
            a = r.get()
            reconstruction[a[0], :] = a[1]
        pool.close()
        pool.join()
        self.reconstruction = reconstruction

