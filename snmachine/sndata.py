"""
Module containing Dataset classes. These read in data from various sources and
turn the light curves into astropy tables that can be read by the rest of the
code.
"""
from __future__ import division  # To be compatible with python 2

__all__ = []

import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sncosmo

from astropy.io import fits
from astropy.table import Table, Column
from past.builtins import basestring
from random import shuffle, sample
from snmachine import chisq as cs

# Colours for graphs
colours = {'sdssu': '#6614de', 'sdssg': '#007718', 'sdssr': '#b30100',
           'sdssi': '#d35c00', 'sdssz': 'k', 'desg': '#007718',
           'desr': '#b30100', 'desi': '#d35c00', 'desz': 'k',
           'lsstu': '#984ea3', 'lsstg': '#377eb8', 'lsstr': '#4daf4a',
           'lssti': '#e3c530', 'lsstz': '#ff7f00', 'lssty': '#e41a1c'}

sntypes = {1: 'Ia', 2: 'II', 21: 'IIn', 22: 'IIP', 23: 'IIL',
           3: 'Ibc', 32: 'Ib', 33: 'Ic', 66: 'other'}
markers = {'desg': '^', 'desr': 'o', 'desi': 's', 'desz': '*',
           'lsstu': 'o', 'lsstg': 'v', 'lsstr': '^',
           'lssti': '<', 'lsstz': '>', 'lssty': 's'}
labels = {'desg': 'g', 'desr': 'r', 'desi': 'i', 'desz': 'z'}


def plot_lc(lc, show_legend=True):
    """
    External function to plot light curves.

    Parameters
    ----------
    lc : astropy.table.Table
        Light curve
    show_legend : bool, optional
        Default True. If True shows the legend of the plot.

    """
    """
        @param fname The filename of the supernova (relative to data_root)
    """
    # This selects the filters from the possible set that this object has
    # measurements in and maintains the order
    filts = np.unique(lc['filter'])

    # Keep track of the min and max values on the plot for resizing axes
    min_x = np.inf
    max_x = -np.inf
    lines = []
    for j in range(len(filts)):
        inds = np.where(lc['filter'] == filts[j])[0]
        if 'flux_error' in lc.keys():
            t = lc['mjd'][inds]
            F, F_err = lc['flux'][inds], lc['flux_error'][inds]
            error = True
        else:
            t, F = lc['mjd'][inds], lc['flux'][inds]
            error = False

        tdelt = t  # tdelt=t-t.min()
        if filts[j] in markers.keys():
            mkr = markers[filts[j]]
        else:
            mkr = 'o'
        if error:
            lc_plot = plt.errorbar(tdelt, F, yerr=F_err, marker=mkr,
                                   linestyle='none', color=colours[filts[j]],
                                   markersize=4)
        else:
            lc_plot = plt.plot(tdelt, F, lw=2, marker=mkr,
                               color=colours[filts[j]])
        lines.append(lc_plot)
        if tdelt.min() < min_x:
            min_x = tdelt.min()
        if tdelt.max() > max_x:
            max_x = tdelt.max()

    ext = 0.05*(max_x-min_x)
    plt.xlim([min_x-ext, max_x+ext])
    plt.xlabel('Time (days)', fontsize=16)
    plt.ylabel('Flux', fontsize=16)

    if show_legend:
        if len(lines) >= 6:
            number_columns = 2
        else:
            number_columns = 2
        plt.legend(lines, filts, ncol=number_columns, numpoints=1, loc='best')


class EmptyDataset:
    """
    Empty data set, to fill up with light curves
    (of format astropy.table.Table) in your memory.
    """

    def __init__(self, folder=None, survey_name=None, filter_set=[]):

        """
        Initialisation.
        Parameters
        ----------
        folder : str
            Root folder containing the data
        survey_name : str
            Specifies the name of the survey; needed for output folder name
        filter_set : list, optional
            List of possible filters used

        """
        self.filter_set = filter_set
        self.rootdir = folder
        self.survey_name = survey_name
        self.data = {}
        self.object_names = []
        self.models = {}

    def mix(self):
        shuffle(self.object_names)

    def set_filters(self, filter_set):
        """
        Setting the filter set of an empty data set
        Parameters
        ----------
        filter_set : array of str
            filters to be added
        """
        self.filter_set = filter_set

    def set_rootdir(self, folder):
        """
        Setting the root directory containing light curves
        Parameters
        ----------
        folder : str
            name of the folder
        """
        self.rootdir = folder
        # self.survey_name=folder.splot(os.path.sep)[-2]

    def insert_lightcurve(self, lc, subtract_min=True):
        """
        Wraps the insertion of a new light curve into a data set. Also includes
        the new object names and possibly new filters into the data set
        metadata. The new object name needs to be in the header.

        Parameters
        ----------
        lc : astropy.table.Table
            new light curve
        """
        name = lc.meta['name']
        self.object_names = np.append(self.object_names, name)
        if subtract_min and len(lc) > 0:
            lc['mjd'] -= lc['mjd'].min()
        self.data[name] = lc
        for flt in np.unique(lc['filter']):
            if not str(flt) in self.filter_set:
                print('Adding filter '+str(flt)+' ...')
                self.filter_set.append(str(flt))

    def __plot_this(self, fname, title=True, loc='best'):
        """
        Internal function used by other functions to plot light curves.

        Parameters
        ----------
        fname : str
            The filename of the supernova (relative to data_root)
        title : str, optional
            Put a title on the plot
        loc : str, optional
            Location of legend
        """

        lc = self.data[fname]

        # This selects the filters from the possible set that this object has
        # measurements in and maintains the order
        filts = sorted(set(self.filter_set) & set(np.unique(lc['filter'])),
                       key=self.filter_set.index)

        # Keep track of the min and max values on the plot for resizing axes
        min_x = np.inf
        max_x = -np.inf
        lines = []

        labs = []
        for j in range(len(filts)):
            f = filts[j]
            inds = np.where(lc['filter'] == filts[j])[0]
            t = lc['mjd'][inds]
            F = lc['flux'][inds]
            F_err = lc['flux_error'][inds]
            if self.sep_detect:
                inds_d = np.where((lc['filter'] == filts[j])
                                  & (lc['detected'] == 1))[0]
                t_d = lc['mjd'][inds_d]
                F_d = lc['flux'][inds_d]
                F_err_d = lc['flux_error'][inds_d]
                if len(t_d) > 0:
                    detect_in_band = True
                else:
                    detect_in_band = False

            tdelt = t  # tdelt=t-t.min()
            if self.sep_detect and detect_in_band:
                tdelt_d = t_d

            if filts[j] in markers.keys():
                mkr = markers[filts[j]]
            else:
                mkr = 'o'
                mkr_d = 'x'

            # Plot the model, if it has been set
            if self.plot_model:
                if fname in self.models.keys():
                    mod = self.models[fname]
                    if mod is not None:
                        inds = np.where(mod['filter'] == filts[j])[0]
                        t_mod, F_mod = mod['mjd'][inds], mod['flux'][inds]
                        plt.plot(t_mod, F_mod, color=colours[filts[j]])

            line = plt.errorbar(tdelt, F, yerr=F_err,  marker=mkr,
                                linestyle='none', color=colours[filts[j]],
                                markersize=4)
            lines.append(line)

            if self.sep_detect and detect_in_band:
                l_d = plt.errorbar(tdelt_d, F_d, yerr=F_err_d, marker=mkr_d,
                                   linestyle='none',  color=colours[filts[j]],
                                   markersize=7)
                lines.append(l_d)

            if tdelt.min() < min_x:
                min_x = tdelt.min()
            if tdelt.max() > max_x:
                max_x = tdelt.max()

            if f in labels.keys():
                labs.append(labels[f])
                if self.sep_detect and detect_in_band:
                    labs.append(labels[f] + ' detected')
            else:
                labs.append(f)
                if self.sep_detect and detect_in_band:
                    labs.append(f + ' detected')

        ext = 0.05*(max_x-min_x)
        plt.xlim([min_x-ext, max_x+ext])
        plt.xlabel('Time (days)')
        plt.ylabel('Flux')
        # plt.gca().tick_params(labelsize=8)
        try:
            chi_obj = [fname]
            chi_2_single_object = self.compute_chisq_over_pts(
                chi_obj)[chi_obj[0]]
        except TypeError:
            print('The chisquare over datapoints can not be computed because '
                  'no model has been fitted.')
            chi_2_single_object = 0

        if title:
            title_z = lc.meta['z']
            title_type = str(lc.meta['type'])
            plt.title(f'Object: {fname}, z: {title_z:.2f},  Type: {title_type}'
                      f', R-ChiSquare: {chi_2_single_object:.3f}')

        if self.sep_detect and loc == 'outside':
            plt.legend(lines, labs, numpoints=1, bbox_to_anchor=(1.02, 1),
                       loc="upper left")
        else:
            plt.legend(lines, labs, numpoints=1, loc=loc)
        # plt.subplots_adjust(left=0.3)

    def plot_lc(self, fname, plot_model=True, title=True, loc='best'):
        """Public function to plot a single light curve.

        Parameters
        ----------
        fname : str
            The filename of the supernova (relative to data_root)
        plot_model : bool, optional
            Whether or not to overplot the model
        title : str, optional
            Put a title on the plot
        loc : str, optional
            Location of the legend
        """
        self.plot_model = plot_model
        self.__plot_this(fname, title=title, loc=loc)
        plt.show()

    def __on_press(self, event):
        """
        Allows one to cycle through the supernovae in the dataset by hitting
        the left or right arrow keys.

        Parameters
        ----------
        event : keyboard event object
            Keyboard event (i.e. the left or right arrow button has been
            pressed).
        """

        event.canvas.figure.clear()
        if event.key == 'right' and self.__ind < len(self.object_names)-1:
            self.__ind += 1
        elif event.key == 'left' and self.__ind > 0:
            self.__ind -= 1
        self.__plot_this(self.object_names[self.__ind])
        event.canvas.draw()

    def plot_all(self, plot_model=True, mix=False, sep_detect=False):
        """
        Plots all the supernovae in the dataset and allows the user to cycle
        through them with the left and right arrow keys.

        Parameters
        ----------
        plot_model : bool, optional
            Whether or not to overplot the model.
        """

        if len(self.data) == 0:
            print('Data set does not contain any light curves - exiting!')
            return
        if mix:
            self.mix()

        self.sep_detect = sep_detect
        # We use a class variable because this can't be passed directly to
        # `.__on_press`
        self.plot_model = plot_model
        fig = plt.figure()
        self.__ind = -1
        self.cid = fig.canvas.mpl_connect('key_press_event', self.__on_press)
        plt.plot([0, 0])
        # subplots_adjust(right=0.95, top=0.95)
        plt.show()

    def save_to_folder(self, outpath, overwrite=True, listname=None):
        """
        Saves the light curves in one data set to a folder, including one
        '.LIST' file with all object names (the name can be changed if
        desired). The light curve format is the sncosmo standard, including a
        header for the metadata

        Parameters
        ----------
        outpath : string
            Path to folder that the light curve files will be saved to. If not
            existent, it will be created.
        overwrite : bool, optional
            If the files already exist, do we overwrite them?
        listname : str
            Hand this argument to this routine if you want the list to be
            stored under a different name.

        """
        # In case the path does not end in separator, add one for good measure
        outpath = os.path.join(outpath, '')

        if not os.path.exists(outpath):
            os.makedirs(outpath)

        # The folder name is the second to last item in the list
        folder_name = outpath.split(os.path.sep)[-2]
        if listname is None:
            object_list_path = os.path.join(outpath, (folder_name+'.LIST'))
        else:
            object_list_path = os.path.join(outpath, listname)
        if overwrite or not os.path.exists(object_list_path):
            np.savetxt(object_list_path, self.object_names, fmt='%s')
        for obj in self.object_names:
            sncosmo.write_lc(self.data[obj], os.path.join(outpath, obj),
                             format='ascii', overwrite=overwrite)

    def get_object_names(self):
        return self.object_names

    def get_max_length(self):
        """Gets the length (in days) of the longest observation in the dataset.
        """
        max_obs = 0
        for n in self.object_names:
            times = self.data[n]['mjd']
            dif = times.max()-times.min()
            if dif > max_obs:
                max_obs = dif
        return max_obs

    def set_model(self, fit_sn, *args):
        """
        Can use any function to set the model for all objects in the data.

        Parameters
        ----------
        fit_sn : function
            A function which can take a light curve (astropy table) argument
            and a list of arguments and returns an astropy table.
        args : list, optional
            Whatever arguments `fit_sn` requires.
        """
        print('Fitting transient models...')
        for obj in self.object_names:
            self.models[obj] = fit_sn(self.data[obj], *args)
        print('Models fitted.')

    def get_types(self):
        """
        Returns a list of the types of the entire dataset.

        Returns
        -------
        `astropy.table.Table`
            Array of types
        """
        typs = []
        for o in self.object_names:
            typs.append(self.data[o].meta['type'])
        tab = Table(data=[self.object_names, typs], names=['Object', 'Type'])
        return tab

    def get_redshift(self):
        """
        Returns a list of the redshifts of the entire dataset.

        Returns
        -------
        `~numpy.ndarray`
            Array of redshifts
        """
        z = []
        for o in self.object_names:
            if 'z' in self.data[o].meta:
                z.append(self.data[o].meta['z'])
            else:
                z.append(-1)
        return np.array(z)

    def sim_stats(self, **kwargs):
        """Prints information about the survey/simulation.

        Parameters
        ----------
        indices : list-like, optional
            List of indices to indicate which objects to consider. This allows,
            for example, to see the statistics of a training subsample.
        plot_redshift : bool, optional
            Plots a histogram of the redshift distribution.
        """
        if len(self.data) == 0:
            print('Data set does not contain any light curves - exiting!')
            return

        if 'indices' in kwargs:
            indices = kwargs['indices']
        else:
            indices = np.arange(len(self.object_names))
        if 'plot_redshifts' in kwargs:
            plot_redshifts = kwargs['plot_redshifts']
        else:
            plot_redshifts = True

        N = len(indices)
        redshifts = np.zeros(N)
        types = np.zeros(N)
        for i in np.arange(N):
            ind = indices[i]
            lc = self.get_lightcurve(self.object_names[ind])
            redshifts[i] = lc.meta['z']
            types[i] = lc.meta['type']

        print(f'\nTotal number of SNe: {N}\n')
        ks = sntypes.keys()
        ks = sorted(ks)
        for k in ks:
            nk = len(np.where(types == k)[0])
            print(f'Number of {sntypes[k]}: {nk} ({nk/N*100:0.2f}%)')
        nk = len(np.where(types == -9)[0])
        print('Number of unknown: {nk} ({nk/N*100:0.2f}%)')

        if plot_redshifts is True:
            plt.hist(redshifts[redshifts != -9], 30, facecolor='#0057f6')
            plt.xlabel('Redshift', fontsize=16)
            plt.show()

    def compute_chisq_over_pts(self, subset='none'):
        """
        Returns the reduced chi squared for each object, once a model has been
        set.

        Parameters
        ----------
        subset : str or list-like, optional
            List of a subset of object names. If not supplied, the full
            dataset will be used.

        Returns
        -------
        dict
            Dictionary of reduced X^2 for each object.
        """
        if len(self.models) == 0:
            pass
            # This gets really annoying as it's being called as part of
            # `plot_all`
            # print('No models fitted so it is not possible to calculate the '
            #       'reduced X^2.')
        else:
            if subset == 'none':
                data_list = self.object_names
            else:
                data_list = subset
            chisq_over_pts_dict = {}

            for obj in data_list:
                obj_data_pd = self.data[obj].to_pandas()
                obj_model_pd = self.models[obj].to_pandas()
                # The passband was saved as bytes
                obj_data_pd_pb = obj_data_pd['filter'].values.astype(str)
                obj_model_pd_pb = obj_model_pd['filter'].values.astype(str)
                obj_data_pd['filter'] = obj_data_pd_pb
                obj_model_pd['filter'] = obj_model_pd_pb

                chisq_over_pts = cs.compute_overall_chisq_over_pts(
                    obj_data_pd, obj_model_pd)
                chisq_over_pts_dict[obj] = chisq_over_pts

            return chisq_over_pts_dict


class PlasticcData(EmptyDataset):
    """Class to read in the PLAsTiCC dataset. This is a simulated LSST catalog.

    Parameters
    ----------
    folder : str
        Folder where simulations are located.
    data_file: str
        Filename of the pandas dataframe which is has the light curve data.
    metadata_file: str
        Filename of the pandas dataframe containing the metadata for the light
        curves.
    mix : bool, optional
        Default False. If True, randomly permutes the objects when they are
        read in.
    """

    def __init__(self, folder, data_file, metadata_file, mix=False):
        super().__init__(folder, survey_name='lsst',
                         filter_set=['lsstu', 'lsstg', 'lsstr', 'lssti',
                                     'lsstz', 'lssty'])
        self.set_data(folder, data_file)
        self.set_metadata(folder, metadata_file)
        if mix is True:
            self.mix()
        # Set the central wavelength of each passband
        self.pb_wavelengths = {'lsstu': 3685, 'lsstg': 4802, 'lsstr': 6231,
                               'lssti': 7542, 'lsstz': 8690, 'lssty': 9736}

    def set_data(self, folder, data_file):
        """Reads in simulated data and saves it.

        The data is saved into the `data` method from EmptyDataset.

        Parameters
        ----------
        folder : str
            Folder where simulations are located.
        data_file : str or list-like
            .csv file of object light curves.
        """
        print('Reading data...')
        time_start_reading = time.time()
        data = pd.read_csv(folder + '/' + data_file, sep=',')
        data = self._remap_filters(df=data)

        # snmachine and PLAsTiCC use a different denomination
        data.rename({'flux_err': 'flux_error'}, axis='columns', inplace=True)
        data.rename({'detected_bool': 'detected'}, axis='columns',
                    inplace=True)
        data.rename({'ddf_bool': 'ddf'}, axis='columns', inplace=True)

        # Abstract column names from dataset
        for col in data.columns:
            if re.search('mjd', col):  # catches the column that includes `mjd`
                self.mjd_col = col
            if re.search('id', col):  # catches the column that includes `id`
                self.id_col = col

        number_invalid_objs = 0  # Some objects may have empty data
        number_objs = len(data[self.id_col].unique())
        object_names = []

        for i, id in enumerate(data[self.id_col].unique()):
            # Use +1 because the order starts at 0 in python
            self.print_progress(i+1, number_objs)

            object_names.append(str(id))
            obj_lc = data.query('{0} == {1}'.format(self.id_col, id))
            lc = self.get_obj_lc_table_starting_from_mjd_zero(pandas_lc=obj_lc)
            if len(lc[self.mjd_col] > 0):
                self.data[str(id)] = lc
            else:
                number_invalid_objs += 1
        if number_invalid_objs > 0:
            print('{} objects were invalid and not added to the dataset.'
                  ''.format(number_invalid_objs))
        self.object_names = object_names
        print('{} objects read into memory.'.format(len(self.data)))
        self.print_time_difference(time_start_reading, time.time())

    def get_obj_lc_table_starting_from_mjd_zero(self, pandas_lc):
        """Transform the pandas dataframe into an astropy table starting from mjd=0

        Takes a pandas light curve from the plasticc dataset format
        and converts it to the astropy table.table format starting from mjd=0.

        Parameters
        ----------
        pandas_lc : pandas.core.frame.DataFrame
            Single object multi-band lightcurve.

        Returns
        -------
        lc : astropy.table.table
            New single object light curve.
        """
        lc = Table.from_pandas(pandas_lc)
        lc[self.mjd_col] -= lc[self.mjd_col].min()
        return lc

    def set_metadata(self, folder, meta_file):
        """Reads in simulated metadata and saves it.

        The data is saved into the `metadata` method from EmptyDataset and
        into a dictonary associated with each `data` method
        (`.data[obj].meta`).

        Parameters
        ----------
        folder : str
            Folder where simulations are located.
        data_file : str or list-like
            .csv file of objects metadata.
        """
        print('Reading metadata...')
        time_start_reading = time.time()
        metadata_pd = pd.read_csv(folder + '/' + meta_file, sep=',',
                                  index_col=self.id_col)
        metadata_pd.index = metadata_pd.index.astype(str)

        # add `object_id` column because it is useful to call it
        metadata_pd['object_id'] = metadata_pd.index
        # snmachine and PLAsTiCC use a different denomination
        metadata_pd.rename({'ddf_bool': 'ddf'}, axis='columns', inplace=True)
        self.metadata = metadata_pd

        # Everything bellow is to conform with `snmachine` version < 2.0
        number_objs = len(self.object_names)
        for i, obj in enumerate(self.object_names):
            # Use +1 because the order starts at 0 in python
            self.print_progress(i+1, number_objs)

            self.set_inner_metadata(obj)
        print(f'Finished getting the metadata for {number_objs} objects.')
        self.print_time_difference(time_start_reading, time.time())

    def set_inner_metadata(self, obj):
        """Set the metadata inside the astropy observation data.

        This inner metadata is only used by `snmachine` version < 2.0 but
        to have backwards compatibility, we keep it.

        Parameters
        ----------
        obj : str
            Name of the object we are working with.
        """
        # remove duplicated entry
        metadata = self.metadata.drop(columns=['object_id'])

        metadata_entry = metadata.loc[obj]
        columns = metadata_entry.keys()
        self.data[obj].meta['name'] = obj  # the name is the object id
        self.data[obj].meta['z'] = None
        for col in columns:
            if re.search('target', col):
                self.data[obj].meta['type'] = str(metadata_entry[col])
            else:
                self.data[obj].meta[str(col)] = metadata_entry[col]
        self._set_metadata_z(obj)

    def _set_metadata_z(self, obj):
        """Set the redshift as spectroscopic or if not available, photometric.

        This is only used by the old code of `snmachine` but to keep backwards
        compatibility, we keep it.

        Parameters
        ----------
        obj : str
            Name of the object we are working with.
        """
        metadata = self.data[obj].meta
        columns = metadata.keys()
        for col in columns:
            if re.search('specz', col) and not np.isnan(metadata[col]):
                self.data[obj].meta['z'] = metadata[col]
            if re.search('photoz', col) and re.search('err', col) is None:
                photoz = metadata[col]
        if self.data[obj].meta['z'] is None:  # if no spec z -> z = photo z
            self.data[obj].meta['z'] = photoz

    @property
    def labels(self):
        """Returns the labels of the objects, if they are known."""
        try:
            labels = self.metadata.target
        except AttributeError:  # We don't know the objects' labels
            labels = None
        return labels

    @property
    def object_names(self):
        """Returns the name of the objects to work with.

        Not always this corresponds to the whole dataset.
        """
        return self._object_names

    @object_names.setter
    def object_names(self, value):
        """Set the name of the objects to work with.

        Parameters
        ----------
        value: list-like
            Name of the objects to work with.
        """
        self._object_names = np.array(value, dtype='str')

    @staticmethod
    def print_time_difference(initial_time, final_time):
        """Print the a time interval.

        Parameters
        ----------
        initial_time : float
            Time at which the time interval starts.
        final_time : float
            Time at which the time interval ends.
        """
        time_spent_on_this_task = pd.to_timedelta(int(final_time-initial_time),
                                                  unit='s')
        print('This has taken {}\n'.format(time_spent_on_this_task))

    @staticmethod
    def print_progress(obj_ordinal, number_objs):
        """Print the percentage of objects already saved.

        This funtion uses a weird formula to know at which percentages to
        print. This formula was choosen because it looks good.

        Parameters
        ----------
        obj_ordinal : int
            Ordinal number of the object we are currently on.
        number_objs : int
            Total number of objects to perform the action on.
        """
        percent_to_print = pow(10, -int(np.log10(number_objs)/2))
        if int(math.fmod(obj_ordinal, number_objs*percent_to_print)) == 0:
            print('{}%'.format(int(obj_ordinal/(number_objs*0.01))))

    def update_dataset(self, new_objs):
        """Update the datset so it only contains a subset of objects.

        Parameters
        ----------
        new_objs : list-like
            The id of the objects we want to have in our dataset.

        Raises
        ------
        ValueError
            All the objects in `new_objs` need to already exist in the dataset.
        """
        if np.sum(~np.in1d(new_objs, self.object_names)) != 0:
            raise ValueError('All the objects in `new_objs` need to exist in '
                             'the original dataset.')

        self.object_names = new_objs
        self.data = {objects: self.data[objects] for objects in
                     self.object_names}

        current_objs = self.metadata.object_id.astype(str)
        is_new_obj = np.in1d(current_objs, new_objs)
        self.metadata = self.metadata[is_new_obj]

        # Reorder the object names to match the metadata
        self.object_names = self.metadata['object_id']

    def _remap_filters(self, df):
        """Remaps the dataset filters to human-understandable values.

        Function to remap integer filters to the corresponding LSST filters.
        For internal `snmachine` consistency, this function also changes the
        column name `passand` to `filter`.

        Parameters
        ----------
        df : pandas.core.frame.DataFrame
            Light curve observations with numeric filters/passbands between 0
            and 5.

        Returns
        -------
        df : pandas.core.frame.DataFrame
            Light curve observations with the LSST filters: `lsstu`, `lsstg`,
            `lsstr`, `lssti`, `lsstz`, `lssty`.
        """
        df.rename({'passband': 'filter'}, axis='columns', inplace=True)
        filter_replace = {0: 'lsstu', 1: 'lsstg', 2: 'lsstr', 3: 'lssti',
                          4: 'lsstz', 5: 'lssty'}
        df['filter'].replace(to_replace=filter_replace, inplace=True)
        return df

    @staticmethod
    def plot_obj_and_model(obj_data, obj_model=None, **kwargs):
        """Plot an object observations and the model fitted to them.

        If `obj_model` is not provided, the function only plots the light
        curve observations.

        Parameters
        ----------
        obj_data : pandas.core.frame.DataFrame or astropy.table.Table
            Time, flux, flux error and passbands of the object.
        obj_model : {None, astropy.table.Table, pandas.core.frame.DataFrame},
                    optional
            If `None`, only plots `obj_data`. Otherwise, `obj_model` has the
            time, flux, flux error (optional) and passbands of the model
            fitted to the object.
        **kwargs : dict
            Additional keyword arguments that can replace default parameters in
            other funtions:
            - axes : {None, matplotlib.axes}, optional
                If the axes are provided, the figure is plotted on the axes.
                Otherwise, it is plotted directly with `matplotlib`.
            - pb_colors : dict, optional
                Mapping between the passband names and the colours with which
                they are represented. If none mapping is provided, the
                passbands are represented with the default colours.
            - show_title : Bool, default = False
                Whether to show the plot title.
            - title : str, optional
                The title for the plot.
            - show_legend : Bool, default = True
                Whether to show the plot legend.

        Raises
        ------
        AttributeError
            There is a default title if `obj_data` contains the object id and
            redshift accessible through `obj_data.meta['name']` and
            `obj_data.meta['hostgal_photoz']`, respectively.
            Otherwise, if the kwarg `show_title` is `True` a title must be
            provided in the kwarg `title`.
        """
        # Extra plotting parameters passed as kwargs
        if 'axes' in kwargs:
            axes = kwargs['axes']
        else:
            axes = None
        if 'pb_colors' in kwargs:
            pb_colors = kwargs['pb_colors']
        else:
            pb_colors = colours
        if 'show_title' in kwargs:
            show_title = kwargs['show_title']
        else:
            show_title = False
        if 'title' in kwargs:
            title = kwargs['title']
        elif show_title is True:
            try:
                title = 'Object ID: {}\nPhoto-z = {:.3f}'.format(
                    obj_data.meta['name'], obj_data.meta['hostgal_photoz'])
            except AttributeError:
                raise AttributeError('No default title available. Provide the'
                                     ' desired title in the kwarg `title`.')
        if 'show_legend' in kwargs:
            show_legend = kwargs['show_legend']
        else:
            show_legend = True

        passbands = ['lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty']
        for pb in passbands:
            # Get the light curve observations in the chosen passband `pb`
            obj_data_pb = obj_data[obj_data['filter'] == pb]

            # Skip this code block and plot only the light curve observations
            # if no `obj_model` was inputed.
            if obj_model is not None:
                # Get the model observations in the chosen passband `pb`
                obj_model_pb = obj_model[obj_model['filter'] == pb]
                model_flux = obj_model_pb['flux']

                # Plot the model values
                if axes is None:
                    plt.plot(obj_model_pb['mjd'], model_flux,
                             color=pb_colors[pb], alpha=.7, label='')
                else:
                    axes.plot(obj_model_pb['mjd'], model_flux,
                              color=pb_colors[pb], alpha=.7, label='')
                try:  # if the model has error information, plot it
                    model_flux_error = obj_model_pb['flux_error']
                    if axes is None:
                        plt.fill_between(x=obj_model_pb['mjd'],
                                         y1=model_flux-model_flux_error,
                                         y2=model_flux+model_flux_error,
                                         color=pb_colors[pb], alpha=.15,
                                         label=None)
                    else:
                        axes.fill_between(x=obj_model_pb['mjd'],
                                          y1=model_flux-model_flux_error,
                                          y2=model_flux+model_flux_error,
                                          color=pb_colors[pb], alpha=.15,
                                          label=None)
                except KeyError:  # the model has no error information
                    pass

            # Plot the object observations.
            # They are ploted after the models to be on top of these.

            # Get the right marker for the plot
            try:
                marker_pb = markers[pb]
            except KeyError:  # no marker for this passband
                marker_pb = 'o'
            if axes is None:
                plt.errorbar(obj_data_pb['mjd'], obj_data_pb['flux'],
                             obj_data_pb['flux_error'], fmt=marker_pb,
                             color=pb_colors[pb], label=pb[-1])
            else:
                axes.errorbar(obj_data_pb['mjd'], obj_data_pb['flux'],
                              obj_data_pb['flux_error'], fmt=marker_pb,
                              color=pb_colors[pb], label=pb[-1])
        if axes is None:
            plt.xlabel('Time (days)')
            plt.ylabel('Flux units')
        else:
            axes.set_xlabel('Time (days)')
            axes.set_ylabel('Flux units')

        if show_title:
            if axes is None:
                plt.title(title)
            else:
                axes.set_title(title)

        if show_legend:
            if axes is None:
                plt.legend(ncol=2, handletextpad=.3, borderaxespad=.3,
                           labelspacing=.2, borderpad=.3, columnspacing=.4)
            else:
                axes.legend(ncol=2, handletextpad=.3, borderaxespad=.3,
                            labelspacing=.2, borderpad=.3, columnspacing=.4)

    def remove_gaps(self, max_gap_length, verbose=False):
        """Remove the first gap longer than the given threshold.

        To remove all the gaps longer than `max_gap_length`, this function
        must be called a few times.

        Parameters
        ----------
        max_gap_length: float
            Maximum duration of the gap to allowed in the light curves.
        verbose: bool, optional
            Default False. If True prints the ID of the longest event and its
            length.
        """
        obj_names = self.object_names
        time_transient = np.zeros(len(obj_names))
        for i in range(len(obj_names)):
            obj_data = self.data[obj_names[i]]
            obs_time = obj_data['mjd']

            # time gaps between consecutive observations
            time_diff = obs_time[1:] - obs_time[:-1]

            if np.max(time_diff) > max_gap_length:
                index_gap = np.nonzero(time_diff >= max_gap_length)[0][0]
                time_last_obs_before = obs_time[index_gap]
                obs_time_detected = obs_time[obj_data['detected'] == 1]

                # number of detections before and after the gap
                number_detections_before = np.sum(
                    obs_time_detected <= time_last_obs_before)
                number_detections_after = np.sum(
                    obs_time_detected > time_last_obs_before)

                # more detections before the gap
                if number_detections_before > number_detections_after:
                    is_obs_transient = obs_time <= time_last_obs_before

                # more detections after the gap
                elif number_detections_before < number_detections_after:
                    is_obs_transient = obs_time > time_last_obs_before

                # same number of detections on before and after the gap
                else:
                    number_obs_before = np.sum(
                        obs_time <= time_last_obs_before)
                    number_obs_after = np.sum(
                        obs_time > time_last_obs_before)
                    # more observation before the gap
                    if number_obs_before >= number_obs_after:
                        is_obs_transient = obs_time <= time_last_obs_before
                    # more observation after the gap
                    else:
                        is_obs_transient = obs_time > time_last_obs_before
                obs_transient = obj_data[is_obs_transient]

                # introduce uniformity: all transients start at time 0
                obs_transient['mjd'] -= min(obs_transient['mjd'])

                self.data[obj_names[i]] = obs_transient
            time_transient[i] = obj_data['mjd'][-1] - obj_data['mjd'][0]
        if verbose:
            print(f'The longest event is '
                  f'{obj_names[np.argmax(time_transient)]} '
                  f'and its length is {np.max(time_transient):.2f} days.')


class Dataset(EmptyDataset):
    """
    Class to manage the files from a single dataset. The base class works with
    data from the SPCC. This class can be inherited and overridden to work a
    completely different kind of dataset.
    All this class really needs is a list of object names and a method,
    `get_lightcurve`, which takes an individual object name and returns a
    light curve. Other functions provided here are for plotting and
    convenience.
    """

    def __init__(self, folder, subset='none',
                 filter_set=['desg', 'desr', 'desi', 'desz']):
        """Initialisation.

        Parameters
        ----------
        folder : str
            Root folder containing the data
        subset : str or list-like, optional
            Subset of events to be passed to `get_object_names` and use in the
            rest of analysis.
        filter_set : list, optional
            List of possible filters used

        """
        self.filter_set = filter_set
        self.rootdir = folder
        self.survey_name = folder.split('/')[-2]

        self.object_names = self.get_object_names(subset=subset)
        # Get all the data as a list of astropy tables (this should not be
        # memory intensive, even for large numbers of light curves)
        self.data = {}
        invalid = 0  # Some objects may have empty data
        print('Reading data...')
        for i in range(len(self.object_names)):
            lc = self.get_lightcurve(self.object_names[i])
            if len(lc['mjd'] > 0):
                self.data[self.object_names[i]] = lc
            else:
                invalid += 1
        if invalid > 0:
            print(f'{invalid} objects were invalid and not added to the '
                  f'dataset.')
        number_objs = len(self.data)
        print(f'{number_objs} objects read into memory.')
        # We create an optional model set which can be set by whatever feature
        # class used
        self.models = {}

    def get_object_names(self, subset='none'):
        """
        Gets a list of the names of the files within the dataset.

        Parameters
        ----------
        subset : str or list-like, optional
            Used to specify which files you want. Current setup is
            `get_object_names` will accept a list of indices, a list of actual
            object names as a subset or the keyword 'spectro'.

        """
        if isinstance(subset, basestring):
            if subset == 'spectro':
                object_names = np.genfromtxt(self.rootdir+'DES_spectro.list',
                                             dtype='U').flatten()
            else:
                file_name = self.rootdir+self.survey_name+'.LIST'
                object_names = np.genfromtxt(file_name, dtype='U')
        elif all(isinstance(l, basestring) for l in subset):
            # We assume subset is a list of strings containing object names
            object_names = subset
        else:
            # Otherwise it must be a list of indices. Otherwise raise an error.
            names = np.genfromtxt(self.rootdir+self.survey_name+'.LIST',
                                  dtype='U')
            try:
                object_names = names[subset]
            except IndexError:
                print('Invalid subset provided')
                sys.exit()

        return np.sort(object_names)

    def get_lightcurve(self, flname):
        """
        Given a filename, returns a light curve astropy table that conforms
        with `sncosmo` requirements.

        Parameters
        ----------
        flname : str
            The filename of the supernova (relative to data_root)

        Returns
        -------
        astropy.table.Table
            Light curve
        """
        fl = open(self.rootdir+flname, 'r')
        mjd = []
        flt = []
        flux = []
        fluxerr = []
        z = -9
        z_err = -9
        type = -9
        for line in fl:
            s = line.split()
            if len(s) > 0:
                if s[0] == 'HOST_GALAXY_PHOTO-Z:':
                    z = (float)(s[1])
                    z_err = (float)(s[3])
                elif s[0] == 'OBS:':
                    mjd.append((float)(s[1]))
                    flt.append('des'+s[2])
                    flux.append((float)(s[4]))
                    fluxerr.append((float)(s[5]))
                elif s[0] == 'SIM_COMMENT:':
                    for k in sntypes.keys():
                        if sntypes[k] in s:
                            type = (int)(k)

        # Zeropoint
        zp = np.array([27.5]*len(mjd))
        zpsys = ['ab']*len(mjd)

        # Make everything arrays
        mjd = np.array(mjd)
        flt = np.array(flt, dtype='str')
        flux = np.array(flux)
        fluxerr = np.array(fluxerr)
        start_mjd = mjd.min()
        # We shift the times of the observations to all start at zero. If
        # required, the mjd of the initial observation is stored in the
        # metadata.
        mjd = mjd-start_mjd
        # Note: obviously this will only zero the observations in one filter
        # passband, the others have to be zeroed if fitting functions.
        tab = Table([mjd, flt, flux, fluxerr, zp, zpsys],
                    names=('mjd', 'filter', 'flux',
                           'flux_error', 'zp', 'zpsys'),
                    meta={'name': flname, 'z': z, 'z_err': z_err, 'type': type,
                          'initial_observation_time': start_mjd})

        return tab

    def get_types(self, show_subtypes=True):
        """
        Returns a list of the types and subtypes (optional) of the entire
        dataset.

        Parameters
        ----------
        show_subtypes : bool, optional
            If True (default), the supernovae subtypes are shown. If False,
            only the main types (Ia, Ibc, II) are returned.

        Returns
        -------
        `astropy.table.Table`
            Array of types
        """  # It adds a new case to the general `get_types`
        typs = []
        for o in self.object_names:
            typs.append(self.data[o].meta['type'])
        tab = Table(data=[self.object_names, typs], names=['Object', 'Type'])

        # For SPCC we might just want the types and not subtypes
        if show_subtypes is False:
            tab['Type'][np.floor(tab['Type']/10) == 2] = 2
            tab['Type'][np.floor(tab['Type']/10) == 3] = 3
        return tab


class OpsimDataset(EmptyDataset):
    """
    Class to read in an LSST simulated dataset, based on OpSim runs and SNANA
    simulations.
    """
    def __init__(self, folder, subset='none', mix=False,
                 filter_set=['lsstu', 'lsstg', 'lsstr',
                             'lssti', 'lsstz', 'lssty']):
        """Initialisation.

        Parameters
        ----------
        folder : str
            Folder where simulations are located
        subset : str or list-like, optional
            List of a subset of object names. If not supplied, the full
            dataset will be used.
        mix : bool, optional
            The output of the simulations is often highly ordered, this
            randomly permutes the objects when they're read in.
        filter_set : list-like, optional
            Set of possible filters.
        """
        self.survey_name = 'LSST'
        self.filter_set = filter_set
        self.get_data(folder, subset=subset)
        self.models = {}

        if mix is True:
            shuffle(self.object_names)

    def get_data(self, folder, subset='none'):
        """
        Reads in the simulated data

        Parameters
        ----------
        folder : str
            Folder where simulations are located
        subset : str or list-like, optional
            List of a subset of object names. If not supplied, the full
            dataset will be used.

        """
        is_basestring = isinstance(subset, basestring)
        is_all_in_subset = (all(isinstance(l, basestring) for l in subset))
        if (~is_basestring and is_all_in_subset):
            # We have to deal with separate Ia and nIa fits files
            Ia_head = os.path.join(folder, 'LSST_Ia_HEAD.FITS')
            nIa_head = os.path.join(folder, 'LSST_NONIa_HEAD.FITS')

            df = fits.open(Ia_head)[1].data
            # If these are read from the header they have to be stripped of
            # white space
            subset = np.char.strip(subset)

            Ia_ids = subset[np.in1d(subset, np.char.strip(df['SNID']))]

            df = fits.open(nIa_head)[1].data
            nIa_ids = subset[np.in1d(subset, np.char.strip(df['SNID']))]

            data_Ia = sncosmo.read_snana_fits(
                Ia_head, os.path.join(folder, 'LSST_Ia_PHOT.FITS'),
                snids=Ia_ids)
            data_nIa = sncosmo.read_snana_fits(
                nIa_head, os.path.join(folder, 'LSST_NONIa_PHOT.FITS'),
                snids=nIa_ids)
        else:
            data_Ia = sncosmo.read_snana_fits(
                os.path.join(folder, 'LSST_Ia_HEAD.FITS'),
                os.path.join(folder, 'LSST_Ia_PHOT.FITS'))
            data_nIa = sncosmo.read_snana_fits(
                os.path.join(folder, 'LSST_NONIa_HEAD.FITS'),
                os.path.join(folder, 'LSST_NONIa_PHOT.FITS'))

        if isinstance(subset, basestring) and subset == 'Ia':
            all_data = data_Ia
        elif isinstance(subset, basestring) and subset == 'nIa':
            all_data = data_nIa
        else:
            all_data = data_Ia+data_nIa
        self.data = {}
        self.object_names = []

        invalid = 0  # Some objects may have empty data
        print('Reading data...')

        for i in range(len(all_data)):
            snid = all_data[i].meta['SNID'].decode('UTF-8')
            is_basestring = isinstance(subset, basestring)
            if is_basestring or ((snid in subset) or (i in subset)):
                self.object_names.append((str)(snid))
                lc = self.get_lightcurve(all_data[i])
                if len(lc['mjd'] > 0):
                    self.data[snid] = lc
                else:
                    invalid += 1
        if invalid > 0:
            print(f'{invalid} objects were invalid and not added to the '
                  f'dataset.')
        self.object_names = np.array(self.object_names, dtype='str')
        number_objs = len(self.data)
        print(f'{number_objs} objects read into memory.')

    def get_lightcurve(self, tab):
        """
        Converts the sncosmo convention for the astropy tables to snmachine's.

        Parameters
        ----------
        tab : astropy.table.Table
            Light curve

        """
        tab_new = tab['MJD', 'FLUXCAL', 'FLUXCALERR', 'FLT']
        tab_new.rename_column('MJD', 'mjd')
        start_mjd = (tab_new['mjd']).min()
        tab_new['mjd'] = tab_new['mjd']-start_mjd
        tab_new.rename_column('FLUXCAL', 'flux')
        tab_new.rename_column('FLUXCALERR', 'flux_error')
        tab_new.rename_column('FLT', 'filter')
        tab_new = Table(tab_new, dtype=['f', 'f', 'f', 'U5'])
        old_filts = ['u', 'g', 'r', 'i', 'z', 'Y']
        for f in range(len(old_filts)):
            new_is_old = tab_new['filter'] == old_filts[f]
            tab_new['filter'][new_is_old] = self.filter_set[f]
        zp = Column(name='zp', data=np.array([27.5]*len(tab_new['mjd'])))
        zpsys = Column(name='zpsys', data=['ab']*len(tab_new['mjd']))
        tab_new.add_column(zp)
        tab_new.add_column(zpsys)

        tab_new.meta = {'name': tab.meta['SNID'].decode('UTF-8'),
                        'z': tab.meta['REDSHIFT_FINAL'],
                        'z_err': tab.meta['REDSHIFT_FINAL_ERR'],
                        'type': tab.meta['SNTYPE'],
                        'initial_observation_time': start_mjd}
        return tab_new


class LSSTCadenceSimulations(OpsimDataset):
    """
    Class to read in the SNANA cadence simulations, which are divided into
    chunks.
    """

    def __init__(self, folder, subset='none', mix=False,
                 filter_set=['lsstu', 'lsstg', 'lsstr',
                             'lssti', 'lsstz', 'lssty'],
                 indices=range(1, 21)):
        """Initialisation.

        Parameters
        ----------
        folder : str
            Folder where simulations are located
        subset : str or list-like, optional
            List of a subset of object names. If not supplied, the full
            dataset will be used.
        mix : bool, optional
            The output of the simulations is often highly ordered, this
            randomly permutes the objects when they're read in.
        filter_set : list-like, optional
            Set of possible filters
        indices : list of ints
            List of indices that index the fits files which each store one
            chunk of the simulated light curves.
        """
        self.indices = indices
        super().__init__(folder, subset, mix, filter_set)

    def get_data(self, folder, subset='none'):
        """
        Reads in the simulated data

        Parameters
        ----------
        folder : str
            Folder where simulations are located.
        subset : str or list-like, optional
            List of a subset of object names. If not supplied, the full
            dataset will be used.
        """
        data_Ia = []
        data_nIa = []

        print('Reading data...')
        for i in self.indices:
            print(f'chunk {i:02d}')
            is_basestring = isinstance(subset, basestring)
            is_all_in_subset = (all(isinstance(l, basestring) for l in subset))
            if (not is_basestring) and is_all_in_subset:
                # We have to deal with separate Ia and nIa fits files
                Ia_head = os.path.join(folder,
                                       f'RH_LSST_WFD_Ia-{i:02d}_HEAD.FITS')
                nIa_head = os.path.join(folder,
                                        f'RH_LSST_WFD_NONIa-{i:02d}_HEAD.FITS')

                df = fits.open(Ia_head)[1].data
                # If these are read from the header they have to be stripped
                # of white space
                subset = np.char.strip(subset)
                Ia_ids = subset[np.in1d(subset, np.char.strip(df['SNID']))]

                df = fits.open(nIa_head)[1].data
                nIa_ids = subset[np.in1d(subset, np.char.strip(df['SNID']))]

                thischunk_Ia = sncosmo.read_snana_fits(
                    Ia_head,
                    os.path.join(folder, f'RH_LSST_WFD_Ia-{i:02d}_PHOT.FITS'),
                    snids=Ia_ids)
                thischunk_nIa = sncosmo.read_snana_fits(
                    nIa_head,
                    os.path.join(folder,
                                 f'RH_LSST_WFD_NONIa-{i:02d}_PHOT.FITS'),
                    snids=nIa_ids)
            else:
                thischunk_Ia = sncosmo.read_snana_fits(
                    os.path.join(folder, f'RH_LSST_WFD_Ia-{i:02d}_HEAD.FITS'),
                    os.path.join(folder, f'RH_LSST_WFD_Ia-{i:02d}_PHOT.FITS'))
                thischunk_nIa = sncosmo.read_snana_fits(
                    os.path.join(folder,
                                 f'RH_LSST_WFD_NONIa-{i:02d}_HEAD.FITS'),
                    os.path.join(folder,
                                 f'RH_LSST_WFD_NONIa-{i:02d}_PHOT.FITS'))

            data_Ia += thischunk_Ia
            data_nIa += thischunk_nIa

        if isinstance(subset, basestring) and subset == 'Ia':
            all_data = data_Ia
        elif isinstance(subset, basestring) and subset == 'nIa':
            all_data = data_nIa
        else:
            all_data = data_Ia+data_nIa
        self.data = {}
        self.object_names = []

        invalid = 0  # Some objects may have empty data

        for i in range(len(all_data)):
            if i % 1e4 == 0:
                number_to_print = i//1e3
                print(f'{number_to_print}k')
            snid = all_data[i].meta['SNID'].decode('UTF-8')
            is_basestring = isinstance(subset, basestring)
            if is_basestring or ((snid in subset) or (i in subset)):
                self.object_names.append((str)(snid))
                lc = self.get_lightcurve(all_data[i])
                if len(lc['mjd'] > 0):
                    self.data[snid] = lc
                else:
                    invalid += 1
        if invalid > 0:
            print(f'{invalid} objects were invalid and not added to the '
                  f'dataset.')
        self.object_names = np.array(self.object_names, dtype='str')
        number_objs = len(self.data)
        print(f'{number_objs} objects read into memory.')


class SDSS_Data(EmptyDataset):
    """
    Class to read in the SDSS supernovae dataset
    """

    def __init__(self, folder, subset='none', training_only=False,
                 filter_set=['sdssu', 'sdssg', 'sdssr', 'sdssi', 'sdssz'],
                 subset_length=False, classification='none'):
        """Initialisation

        Parameters
        ----------
        folder : str
            Folder where simulations are located.
        subset : str or list-like, optional
            List of a subset of object names. If not supplied, the full
            dataset will be used.
        filter_set : list-like, optional
            Set of possible filters
        subset_length : bool or int, optional
            If supplied, will return this many random objects (can be used in
            conjunction with subset="spectro").
        classification : str, optional
            Can return one specific type of supernova.
        """
        self.filter_set = filter_set
        self.rootdir = folder
        # second to last / / /..
        self.survey_name = folder.split(os.path.sep)[-2]
        self.object_names = self.get_object_names(
            subset=subset, subset_length=subset_length,
            classification=classification)
        # Get all the data as a list of astropy tables (this should not be
        # memory intensive, even for large numbers of light curves)
        self.data = {}
        invalid = 0  # Some objects may have empty data
        print('Reading data...')
        for i in range(len(self.object_names)):
            lc = self.get_lightcurve(self.object_names[i])
            if len(lc['mjd'] > 0):
                self.data[self.object_names[i]] = lc
            else:
                invalid += 1
        if invalid > 0:
            print(f'{invalid} objects were invalid and not added to the '
                  f'dataset.')
        number_objs = len(self.data)
        print(f'{number_objs} objects read into memory.')
        # We create an optional model set which can be set by whatever feature
        # class used
        self.models = {}

    def get_SNe(self, subset_length):
        """
        Function to take all supernovae from Master SDSS data file  and return
        a random sample of SNe of user-specified length if requested

        Parameters
        ----------
        subset_length : int
            Number of objects to return

        Returns
        -------
        list-like
            List of object names
        """
        fl = open(self.rootdir+self.survey_name+'.LIST')
        SN = []
        for line in fl:
            s = line.split()
            is_sn_name = (s[5] == "SNIa" or s[5] == "SNIb" or s[5] == "SNIc"
                          or s[5] == "SNII" or s[5] == "SNIa?"
                          or s[5] == "pSNIa" or s[5] == "pSNIbc"
                          or s[5] == "pSNII" or s[5] == "zSNIa"
                          or s[5] == "zSNIbc" or s[5] == "zSNII")
            if is_sn_name:
                if len(str(s[0])) == 3:
                    SN.append("SMP_000%s.dat" % s[0])
                elif len(str(s[0])) == 4:
                    SN.append("SMP_00%s.dat" % s[0])
                elif len(str(s[0])) == 5:
                    SN.append("SMP_0%s.dat" % s[0])
    # SN now contains all file names for supernovae
        if subset_length is not False:
            SN = [SN[i] for i in sorted(sample(range(len(SN)), subset_length))]
        return SN

    def get_spectro(self, subset_length, classification):
        """
        Function to take all spectroscopically confirmed supernovae from
        Master file and return a random sample of SNe of user-specified length
        if requested.

        Parameters
        ----------
        subset_length : bool or int
            Number of objects to return (False to return all).
        classification : str
            Can specify a particular type of supernova to return ('none' for
            all types).

        Returns
        -------
        list-like
            List of object names
        """
        fl = open(self.rootdir+self.survey_name+'.LIST')
        SN = []
        classes = []
        for line in fl:
            s = line.split()
            is_sn_name = (s[5] == "SNIa" or s[5] == "SNIb" or s[5] == "SNIc"
                          or s[5] == "SNII")
            if is_sn_name:
                classes.append(s[5])
                if len(str(s[0])) == 3:
                    SN.append("SMP_000%s.dat" % s[0])
                elif len(str(s[0])) == 4:
                    SN.append("SMP_00%s.dat" % s[0])
                elif len(str(s[0])) == 5:
                    SN.append("SMP_0%s.dat" % s[0])
        # SN now contains all file names for spectroscopically confirmed
        # supernovae

        if subset_length is not False:
            x = sorted(sample(range(len(SN)), subset_length))
            SN = [SN[i] for i in x]
            classes = [classes[i] for i in x]

        # Can specify classification of supernova if user-requested
        if classification != 'none':
            if classification == 'Ia' or classification == 'SNIa':
                SN = [SN[i] for i in range(len(SN))
                      if classes[i] == 'SNIa']
            elif classification == 'Ib' or classification == 'SNIb':
                SN = [SN[i] for i in range(len(SN))
                      if classes[i] == 'SNIb']
            elif classification == 'Ic' or classification == 'SNIc':
                SN = [SN[i] for i in range(len(SN))
                      if classes[i] == 'SNIc']
            elif classification == 'Ibc' or classification == 'SNIbc':
                SN = [SN[i] for i in range(len(SN))
                      if classes[i] == 'SNIb' or classes[i] == 'SNIc']
            elif classification == 'II' or classification == 'SNII':
                SN = [SN[i] for i in range(len(SN)) if classes[i] == 'SNII']
            else:
                print('Invalid classification requested.')
                sys.exit()

        return SN

    def get_photo(self, subset_length, classification):
        """
        Function to take all purely photometric supernovae from Master file
        and return a random sample of SNe of user-specified length if requested

        Parameters
        ----------
        subset_length : bool or int
            Number of objects to return (False to return all).
        classification : str
            Can specify a particular type of supernova to return ('none' for
            all types).

        Returns
        -------
        list-like
            List of object names
        """
        fl = open(self.rootdir+self.survey_name+'.LIST')
        SN = []
        classes = []
        for line in fl:
            s = line.split()
            is_sn_name = (s[5] == "pSNIa" or s[5] == "pSNIbc"
                          or s[5] == "pSNII" or s[5] == "zSNIa"
                          or s[5] == 'zSNIbc' or s[5] == 'zSNII'
                          or s[5] == 'SNIa?')
            if is_sn_name:
                classes.append(s[5])
                if len(str(s[0])) == 3:
                    SN.append("SMP_000%s.dat" % s[0])
                elif len(str(s[0])) == 4:
                    SN.append("SMP_00%s.dat" % s[0])
                elif len(str(s[0])) == 5:
                    SN.append("SMP_0%s.dat" % s[0])
        # SN now contains all file names for spectroscopically confirmed
        # supernovae

        if subset_length is not False:
            x = sorted(sample(range(len(SN)), subset_length))
            SN = [SN[i] for i in x]
            classes = [classes[i] for i in x]

        # Can specify classification of supernova if user-requested
        if classification != 'none':
            if classification == 'Ia' or classification == 'SNIa':
                SN = [SN[i] for i in range(len(SN))
                      if classes[i] == 'SNIa?' or classes[i] == 'pSNIa'
                      or classes[i] == 'zSNIa']
            elif classification == 'Ibc' or classification == 'SNIbc':
                SN = [SN[i] for i in range(len(SN))
                      if classes[i] == 'pSNIbc' or classes[i] == 'zSNIbc']
            elif classification == 'II' or classification == 'SNII':
                SN = [SN[i] for i in range(len(SN))
                      if classes[i] == 'pSNII' or classes[i] == 'zSNII']
            else:
                print('Invalid classification requested.')
                sys.exit()

        return SN

    def get_object_names(self, subset='none', subset_length=False,
                         classification='none'):
        """Gets a list of the names of the files within the dataset.

        Parameters
        ----------
        subset : str or list-like, optional
            List of a subset of object names. If not supplied, the full
            dataset will be used.
        subset_length : bool or int, optional
            Number of objects to return (False to return all)
        classification : str, optional
            Can specify a particular type of supernova to return ('none' for
            all types).
        Returns
        -------
        list-like
            Object names
        """
        if isinstance(subset, basestring):
            if subset == 'spectro':
                # have to convert to numpy array for wavelet features to work
                return np.array(self.get_spectro(subset_length,
                                                 classification))
                # loads random sample of the spec confirmed data defaulted to
                # whole
            elif subset == 'photo':  # only photometric data
                return np.array(self.get_photo(subset_length, classification))
            elif subset == 'none':
                return np.array(self.get_SNe(subset_length))
                # loads random sample of SNe from whole master file - sample
                # defaulted to whole
        elif all(isinstance(l, basestring) for l in subset):
            # We assume subset is a list of strings containing object names
            return subset
        else:
            # Otherwise it must be a list of indices. Otherwise raise an error.
            names = np.genfromtxt(self.rootdir+self.survey_name+'.LIST',
                                  dtype='str')
            try:
                return names[subset]
            except IndexError:
                print('Invalid subset provided')

    def get_info(self, flname):
        """
        Function which takes file name of supernova and returns dictionary of
        spectroscopic and photometric redshifts and their errors when
        available as well as the type of the supernova.

        Parameters
        ----------
        flname : str
            Name of object
        Returns
        -------
        list-like
            Redshift, redshift error, type
        """
        fl = open(self.rootdir+self.survey_name+'.LIST')
        # z_hel is spectroscopic heliocentric redshift and z_psnid uses zspec
        # as prior but has many more??
        z = {'z_hel': float('nan'), 'z_phot': float('nan')}
        z_err = {'z_hel_err': float('nan'), 'z_phot_err': float('nan')}
        t = -9
        for line in fl:
            s = line.split()
            is_good_name = ("SMP_000%s.dat" % s[0] == flname
                            or "SMP_00%s.dat" % s[0] == flname
                            or "SMP_0%s.dat" % s[0] == flname)
            if is_good_name:  # is this a bit slow?
                if s[103] != "\\N":
                    z['z_phot'] = float(s[103])
                else:
                    z['z_phot'] = -9
                if s[11] != "\\N":
                    z['z_hel'] = float(s[11])
                else:
                    z['z_hel'] = -9
                if s[104] != "\\N":
                    z_err['z_phot_err'] = float(s[104])
                else:
                    z_err['z_phot_err'] = -9
                if s[12] != "\\N":
                    z_err['z_hel_err'] = float(s[12])
                else:
                    z_err['z_hel_err'] = -9
                # All classifications of type Ia SNe - includes probable SNeIa
                is_snia = (s[5] == 'SNIa' or s[5] == 'pSNIa' or s[5] == 'zSNIa'
                           or s[5] == 'SNIa?')
                is_snibc = (s[5] == 'SNIb' or s[5] == 'SNIc'
                            or s[5] == 'pSNIbc' or s[5] == 'zSNIbc')
                if is_snia:
                    t = 1
                elif is_snibc:
                    t = 3
                elif s[5] == 'SNII' or s[5] == 'pSNII' or s[5] == 'zSNII':
                    t = 2
        return z, z_err, t

    def get_lightcurve(self, flname):
        """
        Given a filename, returns a light curve astropy table that conforms
        with sncosmo requirements

        Parameters
        ----------
        flname : str
            The filename of the supernova (relative to data_root)
        Returns
        -------
        astropy.table.Table
            Light curve
        """
        print(flname)
        fl = open(self.rootdir+flname, 'r')
        mjd = []
        flt = []  # band
        flux = []
        fluxerr = []  # error in flux
        mag = []
        magerr = []
        line_count = 0
        for line in fl:
            s = line.split()
            line_count = line_count + 1
            # s[0] is flag - a flag of 1024 or greater indicates bad data
            # ( http://arxiv.org/pdf/0908.4277v1.pdf page 18)
            if len(s) > 0 and line_count > 4 and float(s[0]) < 1024:
                mjd.append(s[1])
                flux.append(s[7])
                fluxerr.append(s[8])
                mag.append(s[3])
                magerr.append(s[4])
                if s[2] == '0':
                    flt.append('sdssu')
                elif s[2] == '1':
                    flt.append('sdssg')
                elif s[2] == '2':
                    flt.append('sdssr')
                elif s[2] == '3':
                    flt.append('sdssi')
                elif s[2] == '4':
                    flt.append('sdssz')
        mjd = [float(x) for x in mjd]  # contains strings - convert to floats
        flux = [float(x) for x in flux]
        fluxerr = [float(x) for x in fluxerr]
        mag = [float(x) for x in mag]
        magerr = [float(x) for x in magerr]
        (Z, Z_err, type) = self.get_info(flname)
        if Z['z_hel'] != -9:
            z = Z['z_hel']  # use spectroscopic redshifts if available
            z_err = Z_err['z_hel_err']
        elif Z['z_phot'] != -9:
            z = Z['z_phot']  # photometric redshift
            z_err = Z_err['z_phot_err']
        else:
            z = -9
            z_err = -9  # returns values of -9 if no redshift is available

        # Zeropoint
        zp = np.array([27.5]*len(mjd))
        zpsys = ['ab']*len(mjd)

        # Make everything arrays
        mjd = np.array(mjd)
        flt = np.array(flt, dtype='str')
        flux = np.array(flux)
        fluxerr = np.array(fluxerr)
        mag = np.array(mag)
        magerr = np.array(magerr)
        start_mjd = mjd.min()
        r_flux = np.array([flux[i] for i in range(len(flux))
                           if flt[i] == 'sdssr'])
        if len(r_flux) > 0:
            peak_flux = r_flux.max()
        else:
            peak_flux = -9
        # We shift the times of the observations to all start at zero. If
        # required, the mjd of the initial observation is stored in the
        # metadata.
        mjd = mjd-start_mjd
        # Note: obviously this will only zero the observations in one filter
        # passband, the others have to be zeroed if fitting functions.
        tab = Table([mjd, flt, flux, fluxerr, zp, zpsys, mag, magerr],
                    names=('mjd', 'filter', 'flux', 'flux_error',
                           'zp', 'zpsys', 'mag', 'mag_error'),
                    meta={'name': flname, 'z': z, 'z_err': z_err,
                          'type': type, 'initial_observation_time': start_mjd,
                          'peak flux': peak_flux})
        return tab


class SDSS_Simulations(EmptyDataset):
    """
    Class to read in the SDSS simulations dataset
    """

    def __init__(self, folder, subset='none', training_only=False,
                 filter_set=['sdssu', 'sdssg', 'sdssr', 'sdssi', 'sdssz'],
                 subset_length=False, classification='none'):
        """Initialisation

        Parameters
        ----------
        folder : str
            Folder where simulations are located
        subset : str or list-like, optional
            List of a subset of object names. If not supplied, the full
            dataset will be used.
        filter_set : list-like, optional
            Set of possible filters
        subset_length : bool or int, optional
            If supplied, will return this many random objects (can be used in
            conjunction with subset="spectro").
        classification : str, optional
            Can return one specific type of supernova.
        """
        self.filter_set = filter_set
        self.rootdir = folder
        # second to last / / /..
        self.survey_name = folder.split(os.path.sep)[-2]
        # Get all the data as a list of astropy tables (this should not be
        # memory intensive, even for large numbers of light curves)
        self.data = {}
        invalid = 0  # Some objects may have empty data
        print('Reading data..')
        (self.data, invalid) = self.get_data(subset, subset_length,
                                             classification)
        if invalid > 0:
            print(f'{invalid} objects were invalid and not added to the '
                  f'dataset.')
        number_objs = len(self.data)
        print(f'{number_objs} objects read into memory.')
        self.object_names = self.data.keys()
        # We create an optional model set which can be set by whatever feature
        # class used
        self.models = {}

    def get_data(self, subset='none', subset_length=False,
                 classification='none'):
        """
        Function to get all data in same form as SDSS Data
        """
        # read in data as snana files
        d_Ia = sncosmo.read_snana_fits(self.rootdir + "SDSS_Ia_HEAD.FITS",
                                       self.rootdir + "SDSS_Ia_PHOT.FITS")
        d_nIa = sncosmo.read_snana_fits(self.rootdir + "SDSS_NONIa_HEAD.FITS",
                                        self.rootdir + "SDSS_NONIa_PHOT.FITS")
        invalid = 0  # number of invalid LCs
        data = {}

        # allow user to specify dataset of entirely Ia, Ibc, non-Ia or II
        if classification == 'none':
            for i in range(len(d_Ia)):
                SN = self.get_lightcurve(d_Ia[i])
                if len(SN['mjd']) > 0:
                    sn_id = SN.meta['snid']
                    data[f'{sn_id:s}'] = SN
                else:
                    invalid += 1
            for i in range(len(d_nIa)):
                SN = self.get_lightcurve(d_nIa[i])
                if len(SN['mjd']) > 0:
                    sn_id = SN.meta['snid']
                    data[f'{sn_id:s}'] = SN
                else:
                    invalid += 1
        elif classification == 'Ia' or classification == 'SNIa':
            for i in range(len(d_Ia)):
                SN = self.get_lightcurve(d_Ia[i])
                if len(SN['mjd']) > 0:
                    sn_id = SN.meta['snid']
                    data[f'{sn_id:s}'] = SN
                else:
                    invalid += 1
        elif classification == 'nIa' or classification == 'non-SNIa':
            for i in range(len(d_nIa)):
                SN = self.get_lightcurve(d_nIa[i])
                if len(SN['mjd']) > 0:
                    sn_id = SN.meta['snid']
                    data[f'{sn_id:s}'] = SN
                else:
                    invalid += 1
        elif classification == 'Ibc' or classification == 'SNIbc':
            for i in range(len(d_nIa)):
                SN = self.get_lightcurve(d_nIa[i])
                if len(SN['mjd']) > 0:
                    if SN.meta['type'] == 3:
                        sn_id = SN.meta['snid']
                        data[f'{sn_id:s}'] = SN
                else:
                    invalid += 1
        elif classification == 'II' or classification == 'SNII':
            for i in range(len(d_nIa)):
                SN = self.get_lightcurve(d_Ia[i])
                if len(SN['mjd']) > 0:
                    if SN.meta['type'] == 2:
                        sn_id = SN.meta['snid']
                        data[f'{sn_id:s}'] = SN
                else:
                    invalid += 1
        else:
            print('Invalid classification requested')
            sys.exit()

        # allow user to request entirely spectroscopic data or photometric data
        if subset == 'spectro':
            data = dict((key, value) for (key, value) in data.items()
                        if value.meta['data type'] == 'spec')
        if subset == 'photo':
            data = dict((key, value) for (key, value) in data.items()
                        if value.meta['data type'] == 'phot')

        # allow user to request a shortened subset of the dataset
        if subset_length is not False:
            data = dict(sample(data.items(), subset_length))

        return data, invalid

    def get_lightcurve(self, lc):
        """
        """
        mjd = np.array([lc['MJD'][i] for i in range(len(lc['MJD']))
                        if lc['FLUXCAL'][i] > 0])
        flt = np.array(['sdss' + lc['FLT'][i] for i in range(len(lc['FLT']))
                        if lc['FLUXCAL'][i] > 0])
        # Ignore negative flux values
        flux = np.array([(lc['FLUXCAL'][i]*math.pow(10, -1.44))
                         for i in range(len(lc['FLUXCAL']))
                         if lc['FLUXCAL'][i] > 0])
        # FLUX: MULTIPLY BY 10^-1.44 (FLUX IN SDSS IS CALCULATED AS
        # 10^(-0.4MAG +9.56) WHEREAS SIMULATED FLUXES ARE CALCULATED AS
        # 10^(-0.4MAG + 11)- WE USE THE SDSS CONVENTION).
        fluxerr = np.array([(lc['FLUXCALERR'][i]*math.pow(10, -1.44))
                            for i in range(len(lc['FLUXCALERR']))
                            if lc['FLUXCAL'][i] > 0])
        mag = np.array([lc['MAG'][i] for i in range(len(lc['MAGERR']))
                        if lc['FLUXCAL'][i] > 0])
        magerr = np.array([lc['MAGERR'][i] for i in range(len(lc['MAGERR']))
                           if lc['FLUXCAL'][i] > 0])
        start_mjd = mjd.min()
        r_flux = np.array([flux[i] for i in range(len(flux))
                           if flt[i] == 'sdssr'])
        if len(r_flux) > 0:
            peak_flux = r_flux.max()
        else:
            peak_flux = -9
        # We shift the times of the observations to all start at zero. If
        # required, the mjd of the initial observation is stored in the
        # metadata.
        mjd = mjd-start_mjd
        # Note: obviously this will only zero the observations in one filter
        # passband, the others have to be zeroed if fitting functions.

        # find supernova classification and whether it is spectroscopically
        # classified or photometrically
        sntype = -9
        dtype = -9
        if lc.meta['SNTYPE'] == 120:
            sntype = 1
            dtype = 'spec'
        elif lc.meta['SNTYPE'] == 106:
            sntype = 1
            dtype = 'phot'
        elif lc.meta['SNTYPE'] == 32 or lc.meta['SNTYPE'] == 33:
            sntype = 3
            dtype = 'spec'
        elif lc.meta['SNTYPE'] == 132 or lc.meta['SNTYPE'] == 133:
            sntype = 3
            dtype = 'phot'
        elif lc.meta['SNTYPE'] == 22:
            sntype = 2
            dtype = 'spec'
        elif lc.meta['SNTYPE'] == 122:
            sntype = 2
            dtype = 'phot'
        # get redshift - heliocentric is used where possible, otherwise a
        # simulated heliocentric redshift is used
        z = -9
        z_err = -9
        exists_z = (lc.meta['REDSHIFT_HELIO'] != -9
                    and lc.meta['REDSHIFT_HELIO_ERR'] != -9)
        if exists_z:
            z = lc.meta['REDSHIFT_HELIO']
            z_err = lc.meta['REDSHIFT_HELIO_ERR']
        else:
            z = lc.meta['SIM_REDSHIFT_HELIO']
            # no simulated error in redshift available

        # supernova identifier (used as object name)
        snid = lc.meta['SNID']

        # Zeropoint
        zp = np.array([27.5]*len(mjd))
        zpsys = ['ab']*len(mjd)

        # form astropy table
        tab = Table([mjd, flt, flux, fluxerr, zp, zpsys, mag, magerr],
                    names=('mjd', 'filter', 'flux', 'flux_error', 'zp',
                           'zpsys', 'mag', 'mag_error'),
                    meta={'snid': snid, 'z': z, 'z_err': z_err, 'type': sntype,
                          'initial_observation_time': start_mjd,
                          'peak flux': peak_flux, 'data type': dtype})
        return tab


class SNANA_Data(EmptyDataset):
    """
    Generic class to read in any SNANA simulation set.
    """
    def __init__(self, folder='', list_of_files=[], subset='none',
                 filter_set=['sdssu', 'sdssg', 'sdssr', 'sdssi', 'sdssz'],
                 subset_length=False):
        """Initialisation

        Parameters
        ----------
        folder : str
            Folder where simulations are located
        subset : str or list-like, optional
            List of a subset of object names. If not supplied, the full
            dataset will be used
        filter_set : list-like, optional
            Set of possible filters
        subset_length : bool or int, optional
            If supplied, will return this many random objects (can be used in
            conjunction with subset="spectro")
        classification : str, optional
            Can return one specific type of supernova.
        """
        self.filter_set = filter_set
        if len(list_of_files) == 0:
            if len(folder) == 0:
                print('WARNING: No files or folder provided')
            else:
                fls = os.listdir(folder)
                list_of_files = []
                for f in fls:
                    if 'HEAD' in f and ('FITS' in f or 'fits' in f):
                        list_of_files.append(os.path.join(folder, f))
        self.list_of_files = list_of_files
        # Get all the data as a list of astropy tables (this should not be
        # memory intensive, even for large numbers of light curves)
        self.snana_types_dict = {
            101: 1,
            20: 2,
            21: 2,
            22: 2,
            23: 2,
            120: 2,
            121: 2,
            122: 2,
            123: 2,
            32: 3,
            33: 3,
            132: 3,
            133: 3,
            42: 4,
            142: 4}

        self.data = {}
        invalid = 0  # Some objects may have empty data
        print('Reading data..')
        (self.data, invalid) = self.get_data(subset, subset_length)
        if invalid > 0:
            print(f'{invalid} objects were invalid and not added to the '
                  f'dataset.')
        number_objs = len(self.data)
        print(f'{number_objs} objects read into memory.')
        self.object_names = list(self.data.keys())
        # We create an optional model set which can be set by whatever feature
        # class used
        self.models = {}

        # SNANA conventions (we use 1 for Ia, 2 for II, 3 for Ibc, 4 for
        # Ia_pec). Generally a 1 at the beginning denotes a "photometric" SNe.
        # Since these are simulations this doesn't matter as much

    def get_data(self, subset='none', subset_length=False):
        """
        Function to get all data in same form as SDSS Data
        """
        # read in data as snana files
        # This will automatically look for matching HEAD and PHOT files
        invalid = 0  # number of invalid LCs
        data = {}

        for f in self.list_of_files:
            if 'HEAD' in f:
                ind = f.index('HEAD')
                fl_prefix = f[:ind]
                print('Reading data from the', fl_prefix, 'files')
                phot_file = fl_prefix+'PHOT'+f[ind+4:]
                try:
                    sne = sncosmo.read_snana_fits(f, phot_file)
                except FileNotFoundError:
                    print(f'WARNING Either the HEAD or PHOT file is missing '
                          f'for {fl_prefix}')

            for i in range(len(sne)):
                snid = sne[i].meta['SNID']
                if (subset == 'none') or (snid in subset):
                    SN = self.get_lightcurve(sne[i])
                    if len(SN['mjd']) > 0:
                        sn_id = SN.meta['snid']
                        data[f'{sn_id:s}'] = SN
                    else:
                        invalid += 1
        return data, invalid

    def get_lightcurve(self, lc):
        """
        """
        mjd = np.array([lc['MJD'][i] for i in range(len(lc['MJD']))
                        if lc['FLUXCAL'][i] > 0])
        flt = np.array(['sdss' + lc['FLT'][i] for i in range(len(lc['FLT']))
                        if lc['FLUXCAL'][i] > 0])
        # Ignore negative flux values
        flux = np.array([lc['FLUXCAL'][i] for i in range(len(lc['FLUXCAL']))
                         if lc['FLUXCAL'][i] > 0])
        fluxerr = np.array([lc['FLUXCALERR'][i]
                            for i in range(len(lc['FLUXCALERR']))
                            if lc['FLUXCAL'][i] > 0])

        start_mjd = mjd.min()
        r_flux = np.array([flux[i] for i in range(len(flux))
                           if flt[i] == 'sdssr'])
        if len(r_flux) > 0:
            peak_flux = r_flux.max()
        else:
            peak_flux = -9
        # We shift the times of the observations to all start at zero. If
        # required, the mjd of the initial observation is stored in the
        # metadata.
        mjd = mjd-start_mjd
        # Note: obviously this will only zero the observations in one filter
        # passband, the others have to be zeroed if fitting functions.

        # Find supernova classification and whether it is spectroscopically
        # classified or photometrically
        sntype = lc.meta['SNTYPE']
        if sntype in list(self.snana_types_dict.keys()):
            sntype = self.snana_types_dict[sntype]

        # get redshift - heliocentric is used where possible, otherwise a
        # simulated heliocentric redshift is used
        z = -9
        z_err = -9
        exists_z = (lc.meta['REDSHIFT_HELIO'] != -9
                    and lc.meta['REDSHIFT_HELIO_ERR'] != -9)
        if exists_z:
            z = lc.meta['REDSHIFT_HELIO']
            z_err = lc.meta['REDSHIFT_HELIO_ERR']
        else:
            z = lc.meta['SIM_REDSHIFT_HELIO']
            # no simulated error in redshift available

        # supernova identifier (used as object name)
        snid = lc.meta['SNID'].decode()

        # Zeropoint
        zp = np.array([27.5]*len(mjd))
        zpsys = ['ab']*len(mjd)

        # form astropy table
        tab = Table([mjd, flt, flux, fluxerr, zp, zpsys],
                    names=('mjd', 'filter', 'flux',
                           'flux_error', 'zp', 'zpsys'),
                    meta={'snid': snid, 'z': z, 'z_err': z_err, 'type': sntype,
                          'initial_observation_time': start_mjd,
                          'peak flux': peak_flux})
        return tab
