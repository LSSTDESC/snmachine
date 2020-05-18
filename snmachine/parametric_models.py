"""
Module for parametric models for use in snfeatures module.
"""

from __future__ import print_function

__all__ = []

import numpy as np


class NewlingModel:
    """Parametric model as implemented in Newling et al.
    (http://arxiv.org/abs/1010.1005).
    """
    def __init__(self, **kwargs):
        """Initialisation.

        Parameters
        ----------
        limits : dict, optional
            Change the bounds for the parameters
        initial : dict, optional
            Starting points for the parameters
        """
        self.model_name = 'newling'
        self.param_names = ['logA', 'phi', 'logsigma', 'logk', 'logpsi']
        # self.param_names=['logA', 'phi', 'logsigma', 'logk', 'psi']
        if 'limits' in kwargs:
            self.limits = kwargs['limits']
        else:
            self.limits = {'logA': [0, 10], 'phi': [-60, 100],
                           'logsigma': [-3, 4], 'logk': [-4, 4],
                           'logpsi': [-6, 10]}
            # self.limits={'logA':[0, 10], 'phi':[0, 100],
            #              'logsigma':[1e-5, 4], 'logk':[1e-5, 4],
            #              'psi':[0, 500]}
        if 'initial' in kwargs:
            self.initial = kwargs['initial']  # init params for minuit search
        else:
            self.initial = {'logA': 5, 'phi': 30, 'logsigma': 1, 'logk': 1,
                            'logpsi': 1}

        self.lower_limit = []
        self.upper_limit = []
        for p in self.param_names:
            self.lower_limit.append(self.limits[p][0])
            self.upper_limit.append(self.limits[p][1])
        self.lower_limit = np.array(self.lower_limit)
        self.upper_limit = np.array(self.upper_limit)

    def __fit_spline(self, x, y, d, x_eval):
        """Utility function to fit a spline.
        """

        # Fits a cubic spine between only two points given x and y input and
        # their derivatives and evaluates it on x_eval
        if len(x) > 2:
            print('this is only appropriate for 2 datapoints')
            return 0

        x1, x2 = x
        y1, y2 = y
        k1, k2 = d

        a = k1*(x2-x1)-(y2-y1)
        b = -k2*(x2-x1)+(y2-y1)

        t = (x_eval-x1)/(x2-x1)

        return (1-t)*y1+t*y2+t*(1-t)*(a*(1-t)+b*t)

    def evaluate(self, t, params):
        """Evaluate the function at given values of t.

        Parameters
        ----------
        t : ~np.ndarray
            The time steps over which to evaluate (starting at 0).
        params : list-like
            The parameters of the model.

        Returns
        -------
        ~np.ndarray
            Function values evaluated at t
        """
        logA, phi, logsigma, logk, logpsi = params
        A = np.exp(logA)
        k = np.exp(logk)
        s = np.exp(logsigma)
        psi = np.exp(logpsi)
        Ft = np.zeros(len(t))
        delta = (t-phi)/s
        delta = delta[t > phi]

        tau = k*s+phi  # peak

        # Calculate big psi
        Psi = np.zeros(len(t))

        y_int = self.__fit_spline([phi, tau], [0, psi], [0, 0],
                                  t[(t >= phi) & (t <= tau)])
        Psi[t < phi] = 0
        Psi[(t >= phi) & (t <= tau)] = y_int
        Psi[t > tau] = psi
        Ft[t > phi] = (A * (delta**k) * np.exp(-delta) * (k**(-k)) * np.exp(k)
                       + Psi[t > phi])
        return Ft


class KarpenkaModel:
    """Parametric model as implemented in Karpenka et al.
    (http://arxiv.org/abs/1208.1264).
    """
    def __init__(self, **kwargs):
        """Initialisation.

        Parameters
        ----------
        limits : dict, optional
            Change the bounds for the parameters.
        initial : dict, optional
            Starting points for the parameters.
        """
        self.model_name = 'karpenka'
        self.param_names = ['logA', 'logB', 't0', 't1', 'T_rise', 'T_fall']
        if 'limits' in kwargs:
            self.limits = kwargs['limits']
        else:
            self.limits = {'logA': [np.log(1e-5), np.log(1000)],
                           'logB': [np.log(1e-5), np.log(100)], 't0': [0, 100],
                           't1': [0, 100], 'T_rise': [0, 100],
                           'T_fall': [0, 100]}
            # self.limits={'logA':[1e-5, 1000], 'logB':[1e-5, 100],
            #              't0':[0, 100], 't1':[0, 100], 'T_rise':[0, 100],
            #              'T_fall':[0, 100]}
        if 'initial' in kwargs:
            self.initial = kwargs['initial']  # init params for minuit search
        else:
            self.initial = {'logA': np.log(100), 'logB': np.log(10), 't0': 20,
                            't1': 30, 'T_rise': 40, 'T_fall': 40}

        self.lower_limit = []
        self.upper_limit = []
        for p in self.param_names:
            self.lower_limit.append(self.limits[p][0])
            self.upper_limit.append(self.limits[p][1])

    def evaluate(self, t, params):
        """Evaluate the function at given values of t.

        Parameters
        ----------
        t : ~np.ndarray
            The time steps over which to evaluate (starting at 0).
        params : list-like
            The parameters of the model.

        Returns
        -------
        ~np.ndarray
            Function values evaluated at t.
        """
        logA, logB, t0, t1, T_rise, T_fall = params
        A = np.exp(logA)
        B = np.exp(logB)
        return (A*(1+B*(t-t1)*(t-t1))*np.exp(-(t-t0)/T_fall)
                / (1+np.exp(-(t-t0)/T_rise)))
