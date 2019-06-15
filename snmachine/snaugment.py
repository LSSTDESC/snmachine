import george
import scipy.optimize as op
import numpy as np
from astropy.table import Table, vstack
from snmachine import snfeatures
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek


"""
Module handling the data augmentation of supernova data sets
"""


class SNAugment:
    """
    Skeletal base class outlining the structure for the augmentation of a
    sndata instance. Classes that encapsulate a specific data augmentation
    procedure should be derived from this class.
    """

    def __init__(self, d):
        """
        class constructor.

        Parameters: (why would you call this constructor in the first place?)
        ----------
        d : sndata object
            the supernova data set we want to augment

        """
        self.dataset = d
        # This can contain any metadata that the augmentation
        # process produces, and we want to keep track of.
        self.meta = {}
        self.algorithm = None
        # This is a list of object names that were in the data set prior to augmenting.
        # DO NOT TOUCH FOR SELFISH PURPOSES -- TAREK: Do we need this line anymore?
        self.original = d.object_names.copy()

    def augment(self):
        pass

    def remove(self, obj=None):
        """
        reverts the augmentation step by fully or partially removing those
        light curves that have been added in the augmentation procedure from
        the data set.

        Parameters:
        ----------
        obj : list of strings
            These are the objects we will remove. If None is given, then we
            remove all augmented objects that have not been in the data set
            when we created the SNAugment object.
            NB: If obj contains object names that are in the original data set
            then we do not throw an error, but follow through on what you tell
            us to do.

        """
        if obj is None:
            obj = list(set(self.dataset.object_names()) - set(self.original))

        for o in obj:
            assert(o in self.dataset.object_names)
            self.dataset.data.pop(o)
            self.dataset.object_names = [x for x in self.dataset.object_names if x != o]

    def extract_proxy_features(self, peak_filter='desr', nproc=1, fit_salt2=False, salt2feats=None, return_features=False, fix_redshift=False, which_redshifts='header', sampler='leastsq'):
        """
        Extracts the 2D proxy features from raw light curves, e.g., redshift and peak logflux in a certain band.
        There are plenty of options for how to get these values, if you should be so inclined.
        For the peak flux, we take either the maximum flux amongst the observations in the specified band (quick and dirty),
        or we perform SALT2 fits to the data and extract the peak flux from there (proper and slow).
        For the redshift, we take either the redshift specified in the header or the fitted SALT2 parameter.

        Parameters:
        ----------
        peak_filter : str, optional (default: 'desr')
            name of the filter whose peak flux will be used as second column.
        nproc : int, optional (default: 1)
            number of processes for salt2 feature extraction
        fit_salt2 : boolean, optional (default: False)
            if True, we compute the peak flux from SALT2 fits; if False, we return the brightest observation
        salt2feats : astropy.table.Table, optional (default: None)
            if you already have the features precomputed and do not want to recompute, you can hand them over here.
        return_features : bool, optional (default: False)
            if you want to store fitted SALT2 features, you can set this flag to return them
        fix_redshift : bool, optional (default: False)
            if True, we fix the redshift in the SALT2 fits to the value found in the table headers; if False, we leave the parameter free for sampling
        which_redshifts : str, optional (default: 'header')
            if 'salt2', the first column of proxy features will be the SALT2-fitted redshift; if 'header', we take the redshift from the header,
            if 'headerfill', we take the header redshift where available and fill with fitted values for the objects without valid redshift.
        sampler : str, optional (default: 'leastsq')
            Which sampler do we use to perform the SALT2 fit? 'leastsq' is a simple least squares fit,
            'nested' uses nested sampling and requires MultiNest and PyMultiNest to be installed.
        Returns:
        -------
        proxy_features : np.array
            Nobj x 2 table with the extracted (z,peakflux) proxy features
        salt2feats : astropy.table.Table
            fitted SALT2 features for further applications
        """

        # Consistency check: is the specified filter actually in the dataset?
        if peak_filter not in self.dataset.filter_set:
            raise RuntimeError('Filter %s not amongst the filters in the dataset!')

        # Consistency check: if we want to return salt2-fitted redshifts, do we actually have features?
        if which_redshifts is 'headerfill' and all([(isinstance(z, float)) & (z >= 0) for z in self.dataset.get_redshift()]):
            # Corner case: if 'headerfill' this check should only complain if there are actually invalid z values to fill in
            which_redshifts = 'header'
        if not fit_salt2 and which_redshifts in ['salt2', 'headerfill']:
            print("We need SALT2 features in order to return fitted redshifts! Setting which_redshifts to 'header'.")
            which_redshifts = 'header'

        # Consistency check: to return features, we need to have features
        if return_features and not fit_salt2 and salt2feats is None:
            print("We need SALT2 features to return features - either provide some or compute them! Setting return_features to False.")
            return_features = False

        # Fitting new features
        if fit_salt2:
            # We use a fit to SALT2 model to extract the r-band peak magnitudes
            tf = snfeatures.TemplateFeatures(sampler=sampler)
            if salt2feats is None:
                salt2feats = tf.extract_features(self.dataset, number_processes=nproc, use_redshift=fix_redshift)

            # Fit models and extract r-peakmags
            peaklogflux = []
            for i in range(len(self.dataset.object_names)):
                model = tf.fit_sn(self.dataset.data[self.dataset.object_names[i]], salt2feats)
                model = model[model['filter'] == peak_filter]
                if len(model) > 0:
                    peaklogflux = np.append(peaklogflux, np.log10(np.nanmax(model['flux'])))
                else:
                    # Band is missing: do something better than this
                    peaklogflux = np.append(peaklogflux, -42)
        else:
            peaklogflux = []
            for o in self.dataset.object_names:
                model = self.dataset.data[o]
                model = model[model['filter'] == peak_filter]
                if len(model) > 0:
                    peaklogflux = np.append(peaklogflux, np.log10(np.nanmax(model['flux'])))
                else:
                    # Band is missing: do something better
                    peaklogflux = np.append(peaklogflux, -42)

        # Extracting redshifts
        if which_redshifts is 'header':
            z = self.dataset.get_redshift()
        elif which_redshifts is 'salt2':
            z = [float(salt2feats[salt2feats['Object'] == o]['[Ia]z']) for o in self.dataset.object_names]
        elif which_redshifts is 'headerfill':
            z = self.dataset.get_redshift().astype(float)
            for i in range(len(z)):
                if not isinstance(z[i], float) or z[i] < 0:
                    z[i] = float(salt2feats['[Ia]z'][salt2feats['Object'] == self.dataset.object_names[i]])
        else:
            raise RuntimeError('Unknown value %s for argument which_redshifts!' % which_redshifts)

        proxy_features = np.array([z, peaklogflux]).transpose()

        if return_features:
            return proxy_features, salt2feats
        else:
            return proxy_features

    def compute_propensity_scores(self, train_names, algo='logistic', **kwargs):
        """
        Wherein we fit a model for the propensity score (the probability of an object to be in the training set)
        in the proxy-feature parameter space. We then evaluate the model on the full dataset and return the values.

        Parameters:
        ----------
        train_names : np.array of str
             names of all objects in the training set
        algo : str, optional
             name of the model to fit. 'logistic' is logistic regression, 'network' is a simple ANN with two nodes
             in one hidden layer.
        kwargs : optional
             all of these will be handed to self.extract_proxy_features. See there for more info.

        Returns:
        -------
        propensity_scores : np.array
             array of all fitted scores
        """
        retval = self.extract_proxy_features(**kwargs)
        if len(retval) == 2 and 'return_features' in kwargs.keys() and kwargs['return_features']:
            proxy_features = retval[0]
            salt2_features = retval[1]
        else:
            proxy_features = retval
        # Logistic regression on proxy_features
        is_in_training_set = [1 if o in train_names else 0 for o in self.dataset.object_names]
        if algo is 'logistic':
            regr = LogisticRegression()
        elif algo is 'network':
            regr = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_size=(2,))
        regr.fit(proxy_features, is_in_training_set)
        propensity_scores = regr.predict_proba(proxy_features)
        if len(retval) == 2 and 'return_features' in kwargs.keys() and kwargs['return_features']:
            return propensity_scores[:, 0], salt2_features
        else:
            return propensity_scores[:, 0]

    def divide_into_propensity_percentiles(self, train_names, nclasses, **kwargs):
        """
        Wherein we fit the propensity scores and divide into equal-percentile classes from lowest to hightest

        Parameters:
        ----------
        train_names : np.array of str
            names of objects in the training set
        nclasses : int
            number of classes
        kwargs : optional
            all of these will be handed to self.extract_proxy_features. See there for more info.
        Returns:
        -------
        classes : np.array of int
            classes from 0, ..., nclasses-1, in the same order as self.dataset.object_names.
        """
        retval = self.compute_propensity_scores(train_names, **kwargs)
        if len(retval) == 2 and 'return_features' in kwargs.keys() and kwargs['return_features']:
            prop = retval[0]
            salt2_features = retval[1]
        else:
            prop = retval
        sorted_indices = np.argsort(prop)
        N = len(self.dataset.object_names)
        classes = np.array([-42] * N)
        for c in range(len(self.dataset.object_names)):
            thisclass = (c * nclasses) // N
            classes[sorted_indices[c]] = thisclass
        if len(retval) == 2 and 'return_features' in kwargs.keys() and kwargs['return_features']:
            return classes, salt2_features
        else:
            return classes


class NNAugment(SNAugment):
    """
    Derived class that encapsulates data augmentation via Nearest Neighbour
    inspired algorithms such as SMOTE, ADASYN etc.
    """

    def __init__(self, X, y, method):
        self.X = X
        self.y = y
        self.method = method
    # def __init__(self, data, method):
    #         super().__init__(data)
    #         self.method = method
    """
    Make directories that will be used for analysis

    Parameters
    ----------
    X : numpy.ndarray
        Collection of data containing features
    y : numpy.ndarray
        True labels for data
    method : string
        Augmentation method one would like to resample the given data with.
        List of possible choices include:

            ['SMOTE', 'ADASYN', 'SVMSMOTE', 'SMOTEENN', 'SMOTETomek']

    Notes
    -------


    """
    _METHODS = [
        'SMOTE',
        'ADASYN',
        'SVMSMOTE',
        'SMOTEENN',
        'SMOTETomek'
    ]

    @classmethod
    def methods(cls):
        return cls._METHODS

    @staticmethod
    def augment(X, y, method):

        print(NNAugment.methods())
        if method not in NNAugment.methods():
            raise ValueError(F"{method} not a possible augmentation method in `snmachine`")

        print(F"Before resampling: {sorted(Counter(y).items())}")

        X_resampled, y_resampled = eval(method)().fit_resample(X, y)
        print(F"After resampling: {sorted(Counter(y_resampled).items())}")

        return X_resampled, y_resampled


class GPAugment(SNAugment):
    """
    Derived class that encapsulates data augmentation via Gaussian Processes
    """

    def __init__(self, d, stencils=None, cadence_stencils=None, stencil_weights=None, cadence_stencil_weights=None):
        """
        class constructor.

        Parameters:
        ----------
        d : sndata object
            the supernova data set we want to augment
        stencils : list of strings
            If the stencils argument is given (as a list of object names
            that are in the data set), then the augmentation step will take
            these light curves to train the GPs on. If not, then every object
            in the data set is considered fair game.
        cadence_stencils : list of strings
            If given, the augmentation will sample the cadence for the new light
            curves from these objects. If not, every object is fair game.
        stencil_weights : np.array or list of float
            If given, these weights are the probabilities with which the respective
            stencils will be picked in the augmentation step. If not given, we will
            use uniform weights.
        cadence_stencil_weights : np.array or list of float
            Like stencil_weights, but for the cadence stencils.
        """

        self.dataset = d
        self.meta = {}
        self.meta['trained_gp'] = {}
        self.algorithm = 'GP augmentation'
        if stencils is None:
            self.stencils = d.object_names.copy()
        else:
            self.stencils = stencils
        if cadence_stencils is None:
            self.cadence_stencils = d.object_names.copy()
        else:
            self.cadence_stencils = cadence_stencils

        if stencil_weights is not None:
            assert np.all(stencil_weights >= 0.), 'Stencil weights need to be larger than zero!'
            stencil_weights = np.array(stencil_weights)
            self.stencil_weights = stencil_weights / sum(stencil_weights)
        else:
            self.stencil_weights = np.ones(len(self.stencils)) / len(self.stencils)

        if cadence_stencil_weights is not None:
            assert np.all(cadence_stencil_weigths >= 0.), 'Cadence stencil weights need to be larger than zero!'
            cadence_stencil_weights = np.array(cadence_stencil_weights)
            self.cadence_stencil_weights = cadence_stencil_weights / sum(cadence_stencil_weights)
        else:
            self.cadence_stencil_weights = np.ones(len(self.cadence_stencils)) / len(self.cadence_stencils)

        self.rng = np.random.RandomState()
        self.random_seed = self.rng.get_state()

        self.original = d.object_names.copy()

    def train_filter(self, x, y, yerr, initheta=[100, 20]):
        """
        Train one Gaussian process on the data from one band. We use the squared-exponential
        kernel, and we optimise its hyperparameters

        Parameters:
        -----------
        x : numpy array
            mjd values for the cadence
        y : numpy array
            flux values
        yerr : numpy array
            errors on the flux
        initheta : list, optional
            initial values for the hyperparameters. They should roughly correspond to the
            spread in y and x direction.

        Returns:
        -------
        g : george.GP
            the trained GP object
        """
        def nll(p):
            g.set_parameter_vector(p)
            ll = g.log_likelihood(y, quiet=True)
            return -ll if np.isfinite(ll) else 1.e25

        def grad_nll(p):
            g.set_parameter_vector(p)
            return -g.grad_log_likelihood(y, quiet=True)

        g = george.GP(initheta[0] ** 2 * george.kernels.ExpSquaredKernel(metric=initheta[1]**2))
        if len(x) == 0:
            return g
        g.compute(x, yerr)
        p0 = g.get_parameter_vector()
        results = op.minimize(nll, p0, jac=grad_nll, method='L-BFGS-B')
        g.set_parameter_vector(results.x)
        return g

    def sample_cadence_filter(self, g, cadence, y, yerr, add_measurement_noise=True):
        """
        Given a trained GP, and a cadence of mjd values, produce a sample from the distribution
        defined by the GP, on that cadence. The error bars are set to the spread of the GP distribution
        at the given mjd value.

        Parameters:
        -----------
        g : george.GP
            the trained Gaussian process object
        cadence : numpy.array
            the cadence of mjd values.
        y : numpy array
            the flux values of the data that the GP has been trained on.
        add_measurement_noise : bool, optional
            cf the documentation of snaugment.GPAugment.produce_new_lc

        Returns:
        --------
        flux : numpy array
            flux values for the new sample
        fluxerr : numpy array
            error bars on the flux for the new sample
        """
        if len(cadence) == 0:
            flux = np.array([])
            fluxerr = np.array([])
        else:
            mu, cov = g.predict(y, cadence)
            flux = self.rng.multivariate_normal(mu, cov)
            fluxerr = np.sqrt(np.diag(cov))
        # Adding measurement error
        if add_measurement_noise:
            flux += self.rng.randn(len(y)) * yerr
            fluxerr = np.sqrt(fluxerr**2 + yerr**2)
        return flux, fluxerr

    def produce_new_lc(self, obj, cadence=None, savegp=True, samplez=True, name='dummy', add_measurement_noise=True):
        """
        Assemble a new light curve from a stencil. If the stencil already has been used
        and the resulting GPs have been saved, then we use those. If not, we train a new GP.

        Parameters:
        -----------
        obj : str or astropy.table.Table
           the object (or name thereof) that we use as a stencil to train the GP on.
        cadence : str or dict of type {string:numpy.array}, optional.
           the cadence for the new light curve, defined either by object name or by {filter:mjds}. If none is given,
           then we pull the cadence of the stencil.
        savegp : bool, optional
           Do we save the trained GP in self.meta? This results in a speedup, but costs memory.
        samplez : bool, optional
           Do we give the new light curve a random redshift value drawn from a Gaussian of location
           and width defined by the stencil? If not, we just take the value of the stencil.
        name : str, optional
           object name of the new light curve.
        add_measurement_noise : bool, optional
           Usually, the data is modelled as y_i = f(t_i) + sigma_i*eps_i, where f is a gaussianly-distributed function, and
           where eps_i are iid Normal RVs, and sigma_i are the measurement error bars. If this flag is unset, we return a
           sample from the GP f and its stddev. If it is set, we return y*_j including the measurement noise (also in the error bar).
           If this is unclear, please consult Rasmussen/Williams chapter 2.2.

        Returns:
        --------
        new_lc: astropy.table.Table
           The new light curve
        """

        if type(obj) is Table:
            obj_table = obj
            obj_name = obj.meta['name']
        elif type(obj) in [str, np.str_]:
            obj_table = self.dataset.data[obj]
            obj_name = str(obj)
        else:
            print('obj: type %s not recognised in produce_new_lc()!' % type(obj))
            # TODO: actually throw an error

        if cadence is None:
            cadence_dict = self.extract_cadence(obj)
        else:
            if add_measurement_noise:
                print('warning: GP sampling does NOT include measurement noise, since sampling is performed on a different cadence!')
                add_measurement_noise = False
            if type(cadence) in [str, np.str_]:
                cadence_dict = self.extract_cadence(cadence)
            elif type(cadence) is dict:
                cadence_dict = cadence
            else:
                print('cadence: type %s not recognised in produce_new_lc()!' % type(cadence))
                # TODO: actually throw an error

        # Either train a new set of GP on the stencil obj, or pull from metadata
        if obj_name in self.meta['trained_gp'].keys():
            print('fetching')
            all_g = self.meta['trained_gp'][obj_name]
        else:
            print('training')
            all_g = {}
            for f in self.dataset.filter_set:
                obj_f = obj_table[obj_table['filter'] == f]
                x = np.array(obj_f['mjd'])
                y = np.array(obj_f['flux'])
                yerr = np.array(obj_f['flux_error'])
                g = self.train_filter(x, y, yerr)
                all_g[f] = g
            if savegp:
                self.meta['trained_gp'][obj_name] = all_g

        # Produce new LC based on the set of GP
        if samplez and 'z_err' in obj_table.meta.keys():
            newz = obj_table.meta['z'] + obj_table.meta['z_err'] * self.rng.randn()
        else:
            newz = obj_table.meta['z']
        new_lc_meta = obj_table.meta.copy()
        modified_meta = {'name': name, 'z': newz, 'stencil': obj_name, 'augment_algo': self.algorithm}
        new_lc_meta.update(modified_meta)
        new_lc = Table(names=['mjd', 'filter', 'flux', 'flux_error'], dtype=['f', 'U', 'f', 'f'], meta=new_lc_meta)
        for f in self.dataset.filter_set:
            obj_f = obj_table[obj_table['filter'] == f]
            y = obj_f['flux']
            yerr = obj_f['flux_error']
            flux, fluxerr = self.sample_cadence_filter(all_g[f], cadence_dict[f], y, yerr, add_measurement_noise=add_measurement_noise)
            filter_col = [str(f)] * len(cadence_dict[f])
            dummy_table = Table((cadence_dict[f], filter_col, flux, fluxerr), names=['mjd', 'filter', 'flux', 'flux_error'], dtype=['f', 'U', 'f', 'f'])
            new_lc = vstack([new_lc, dummy_table])

        # Sort by cadence, for cosmetics
        new_lc.sort('mjd')
        return new_lc

    def extract_cadence(self, obj):
        """
        Given a light curve, we extract the cadence in a format that we can insert into produce_lc and sample_cadence_filter.

        Parameters:
        -----------
        obj : str or astropy.table.Table
            (name of) the object

        Returns:
        --------
        cadence : dict of type {str:numpy.array}
            the cadence, in the format {filter1:mjd1, filter2:mjd2, ...}
        """
        if type(obj) in [str, np.str_]:
            table = self.dataset.data[obj]
            objname = obj
        elif type(obj) is Table:
            table = obj
            objname = table.meta['name']
        else:
            print('obj: type %s not recognised in extract_cadence()!' % type(obj))
        cadence = {flt: np.array(table[table['filter'] == flt]['mjd']) for flt in self.dataset.filter_set}
        return cadence

    def augment(self, numbers, return_names=False, **kwargs):
        """
        High-level wrapper of GP augmentation: The dataset will be augmented to the numbers by type.
        Parameters:
        ----------
        numbers : dict of type int:int
            The numbers to which the data set will be augmented, by type. Keys are the types,
            values are the numbers of light curves pertaining to a type after augmentation.
            Types that do not appear in this dict will not be touched.
        return_names : bool
            If True, we return a list of names of the objects that have been added into the data set
        kwargs :
            Additional arguments that will be handed verbatim to produce_new_lc. Interesting choices
            include savegp=False and samplez=True
        Returns:
        --------
        newobjects : list of str
            The new object names that have been created by augmentation.
        """
        dataset_types = self.dataset.get_types()
        # dataset_types['Type'][dataset_types['Type']>10]=dataset_types['Type'][dataset_types['Type']>10]//10#NB: this is specific to a particular remapping of indices!!

        types = np.unique(dataset_types['Type'])

        newnumbers = dict()
        newobjects = []
        for t in types:
            thistype_oldnumbers = len(dataset_types['Type'][dataset_types['Type'] == t])
            newnumbers[t] = numbers[t] - thistype_oldnumbers
            thistype_stencils = [dataset_types['Object'][i] for i in range(len(dataset_types)) if dataset_types['Object'][i] in self.stencils and dataset_types['Type'][i] == t]
            thistype_stencil_weights = [self.stencil_weights[i] for i in range(len(dataset_types)) if dataset_types['Object'][i] in self.stencils and dataset_types['Type'][i] == t]

            if newnumbers[t] < 0:
                print('There are already %d objects of type %d in the original data set, cannot augment to %d!' % (thistype_oldnumbers, t, numbers[t]))
                continue
            elif newnumbers[t] == 0:
                continue
            else:
                # print('now dealing with type: '+str(t))
                # print('stencils: '+str(thistype_stencils))
                for i in range(newnumbers[t]):
                    # Pick random stencil
                    # thisstencil=thistype_stencils[self.rng.randint(0,len(thistype_stencils))]
                    thisstencil = self.rng.choice(thistype_stencils, p=thistype_stencil_weights / np.sum(thistype_stencil_weights))
                    # Pick random cadence
                    # thiscadence_stencil=self.cadence_stencils[self.rng.randint(0,len(self.cadence_stencils))]
                    # thiscadence_stencil=thisstencil

                    # cadence=self.extract_cadence(thiscadence_stencil)
                    # new_name='augm_t%d_%d_'%(t,i) + thisstencil + '_' + thiscadence_stencil + '.DAT'
                    new_name = 'augm_t%d_%d_' % (t, i) + thisstencil  # + '.DAT'
                    new_lightcurve = self.produce_new_lc(obj=thisstencil, name=new_name, **kwargs)  # ,cadence=cadence)
                    self.dataset.insert_lightcurve(new_lightcurve)
                    # print('types: '+str(new_lightcurve.meta['type'])+' '+str(self.dataset.data[thisstencil].meta['type']))
                    newobjects = np.append(newobjects, new_name)
        return newobjects
