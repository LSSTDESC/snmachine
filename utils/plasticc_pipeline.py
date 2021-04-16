"""
Useful functions for the Machine learning pipeline for classifying the
PLAsTiCC dataset using snmachine.
"""

import multiprocessing
import os
import re
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import yaml
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
try:
    from snmachine import gps, snaugment, sndata, snfeatures
except ImportError:
    print("Unable to import snmachine. Check environment set correctly")

from argparse import ArgumentParser
from astropy.table import Table
#from imblearn.metrics import classification_report_imbalanced  # not used at the moment
#from imblearn.over_sampling import SMOTE  # not used at the moment
#from imblearn.pipeline import make_pipeline  # not used at the moment
from utils.plasticc_utils import plasticc_log_loss, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def get_git_revision_short_hash():
    """Helper function to obtain current version control hash value

    Returns
    -------
    _hash : str
        Short representation of current version control hash value

    Examples
    --------
    >>> ...
    >>> sha = get_git_revision_short_hash()
    >>> print(sha)
    'ede068e'
    """
    _hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    return _hash.decode("utf-8").rstrip()


def get_timestamp():
    """Helper function to obtain latest modified time of the configuration file

    Returns
    -------
    timestamp : str
        Short representation of last modified time for the configuration file
        used: 'YYYY-MM-DD-HOURMINUTE'

    Examples
    --------
    >>> ...
    >>> timestamp = get_timestamp()
    >>> print(timestamp)
    '2019-05-18-2100'
    """
    _timestamp = subprocess.check_output(['date', '+%Y-%m-%d-%H%M'])
    return _timestamp.decode("utf-8").rstrip()


def create_folder_structure(analyses_directory, analysis_name):
    """Make directories that will be used for analysis.

    Parameters
    ----------
    analysis_directory : str
        System path to where the user would like to contain
        a run of the analysis
    analysis_name : str
        Given name of analysis run. This is appended with the current git hash
        the code has been run with.

    Examples
    --------
    Each folder name can then be accessed with dictionary methods:
    >>> ...
    >>> analysis_directory = params.get("analysis_directory", None)
    >>> analysis_name = params.get("analysis_name", None)
    >>> directories = create_folder_structure(analysis_directory,
                                              analysis_name)
    >>> print(directories.get("analysis_directory", None))
    """
    analysis_directory = os.path.join(analyses_directory, analysis_name)
    features_directory = os.path.join(analysis_directory, 'wavelet_features')
    classifications_directory = os.path.join(analysis_directory,
                                             'classifications')
    intermediate_files_directory = os.path.join(analysis_directory,
                                                'intermediate_files')
    plots_directory = os.path.join(analysis_directory, 'plots')

    directories = {
            "analysis_directory": analysis_directory,
            "features_directory": features_directory,
            "classifications_directory": classifications_directory,
            "intermediate_files_directory": intermediate_files_directory,
            "plots_directory": plots_directory}

    if os.path.isdir(analysis_directory):
        errmsg = """
                Folders already exist with this analysis name.

                Are you sure you would like to proceed, this will overwrite the
                {} folder [Y/n]
                """.format(analysis_name)
        print(errmsg)

        _yes = ["yes", "y", "ye"]
        _no = ["no", "n"]

        choice = input().lower()

        if choice in _yes:
            print("Overwriting existing folder..")
            for key, value in directories.items():
                subprocess.call(['mkdir', value], stderr=subprocess.DEVNULL)
        elif choice in _no:
            print("I am NOT sure")
            sys.exit()
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'")
    else:
        for key, value in directories.items():
            subprocess.call(['mkdir', value])


def get_directories(analyses_directory, analysis_name):
    """Return the folder directories inside of a given analysis.

    # TODO [Add a link to the place where we have an explanation of the folder
    # structure]

    Parameters
    ----------
    analyses_directory : str
        System path to where the user stores all analysis.
    analysis_name : str
        Name of the analysis we want.

    Returns
    -------
    directories : dict
        Dictionary containing the mapping of folders inside of `analysis_name`.

    Raises
    ------
    ValueError
        If the folders of the required analysis do not exist.
    """
    analysis_directory = os.path.join(analyses_directory, analysis_name)
    exists_path = os.path.exists(analysis_directory)
    if not exists_path:
        raise ValueError('There are no folders created in {}. You can create '
                         'new folders for this analysis using '
                         '`plasticc_pipeline.create_folder_structure`.'
                         ''.format(analysis_directory))

    features_directory = os.path.join(analysis_directory, 'wavelet_features')
    classifications_directory = os.path.join(analysis_directory,
                                             'classifications')
    intermediate_files_directory = os.path.join(analysis_directory,
                                                'intermediate_files')
    plots_directory = os.path.join(analysis_directory, 'plots')

    directories = {"analysis_directory": analysis_directory,
                   "features_directory": features_directory,
                   "classifications_directory": classifications_directory,
                   "intermediate_files_directory":
                   intermediate_files_directory,
                   "plots_directory": plots_directory}

    return directories


def load_configuration_file(path_to_configuration_file):
    """Load from disk the configuration file that is to be used

    Parameters
    ----------
    path_to_configuration_file : str
        System path to where the configuration file is located

    Returns
    -------
    params : dict
        Dictionary of parameters contained inside the configuration file

    Examples
    --------
    Each item inside the configuration file can be accessed like so:
    >>> ...
    >>> params = load_configuration_file(path_to_configuration_file)
    >>> kernel_param = params.get("kernel_param", None)
    >>> print(kernel_param)
    [500.0, 20.0]
    >>> number_gp = params.get("number_gp", None)
    >>> print(number_gp)
    '1100'
    """
    try:
        with open(path_to_configuration_file) as f:
            params = yaml.load(f)
    except IOError:
        print("Invalid yaml file provided")
        sys.exit()
    print("The parameters are:\n {}".format(params))
    return params


def save_configuration_file(params, analysis_directory):
    """Make a copy of the configuration file that has been used inside the
    analysis directory

    Parameters
    ----------
    params : dict
        Dictionary containing the parameters used for this analysis
    analysis_directory : string
        Folder where this analysis is taking place

    Returns
    -------
    None

    Examples
    --------
    >>> ...
    >>> save_configuration_file(params, analysis_directory)
    >>> subprocess.call(['cat', os.path.join(analysis_directory, "logs.yml")])
    analysis_directory: /share/hypatia/snmachine_resources/data/plasticc/analysis/
    analysis_name: pipeline-test
    data_path: /share/hypatia/snmachine_resources/data/plasticc/data/raw_data/training_set_snia.pickle
    git_hash: 916eaec
    kernel_param:
    - 500.0
    - 20.0
    number_gp: 1100
    number_of_principal_components: 10
    timestamp: 2019-05-21-1204
    """
    git_hash = {"git_hash": get_git_revision_short_hash()}
    timestamp = {"timestamp": get_timestamp()}

    params.update(git_hash)
    params.update(timestamp)

    with open(os.path.join(analysis_directory, "logs.yml"), 'a+') as config:
        yaml.dump(params, config, default_flow_style=False)


def load_dataset(data_path):
    """Load from disk the dataset to use.

    This dataset was previously saved as a snmachine.PlasticcData instance.

    Parameters
    ----------
    params : dict
        Dictionary containing the parameters that reside in the configuration
        file. This will be used to obtain the path to the training data.

    Returns
    -------
    dataset : snmachine.PlasticcData
        snmachine.PlasticcData instance of a dataset

    Examples
    --------
    >>> ...
    >>> dataset = load_dataset(params)
    >>> print(dataset)
    <snmachine.sndata.PlasticcData object at 0x7f8dc9dd4e10>
    """
    try:
        if data_path.lower().endswith((".pickle", ".pkl", ".p", ".pckl")):
            with open(data_path, 'rb') as input:
                print("Opening from binary pickle")
                dataset = pickle.load(input)
                print("Dataset loaded from pickle file as: {}".format(
                    dataset))
        else:
            folder_path, data_file_name = os.path.split(data_path)
            metadata_file_name = "_metadata.".join(data_file_name.split(
                "."))

            print("Opening from CSV")
            dataset = sndata.PlasticcData(folder=folder_path,
                                          data_file=data_file_name,
                                          metadata_file=metadata_file_name)
            print("Dataset loaded from csv file as: {}".format(dataset))
            print("Saving {} object to pickle binary".format(dataset))

            dat_binary = os.path.splitext(data_file_name)[0] + ".pckl"
            print(os.path.join(folder_path, dat_binary))
            with open(os.path.join(folder_path, dat_binary), 'wb') as f:
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    except FileNotFoundError:
        print("No file found to load")
        exit()

    return dataset


def reduce_size_of_training_data(training_data, directories, subset_size, seed=1234, save_subset_list=False):
    # TODO: Incorpate further doctrings and finish examples. Tarek: Catarina and I need to
    # discuss this further. There is some overlap between this and
    # sndata.PlasticcData.update_data() and it would be good to comebine this.
    """Load from disk the training data one will use for this analysis

    Parameters
    ----------
    training_data : snmachine.PlasticcData
        Dictionary containing the parameters that reside in the configuration
        file. This will be used to obtain the path to the training data.
    directories : dict
        Dictionary containing
    subset_size : int
        Number of objects the user would like to reduce the training data to
    seed : int
        Default set to 1234. This can be overridden by the user to check for
        consistancy of results

    Returns
    -------
    None

    Examples
    --------
    >>> ...
    >>> new_training_data = reduce_size_of_training_data(training_data, directories, 1000)
    """
    analysis_directory = directories.get("analysis_directory", None)
    print(F"ANALYSIS DIR: {analysis_directory}")
    subset_file = os.path.join(str(analysis_directory), "subset.list")
    if os.path.exists(subset_file):
        rand_objs = np.genfromtxt(subset_file, dtype='U')
    else:
        np.random.seed(seed)
        rand_objs = np.random.choice(training_data.object_names, replace=False, size=subset_size)
        rand_objs_sorted_int = sorted(rand_objs, key=lambda x: int(re.sub('\D', '', x)))
        rand_objs = np.asarray(rand_objs_sorted_int, dtype='U')
        if save_subset_list:
            np.savetxt(subset_file, rand_objs, fmt='%s')

    training_data.object_names = rand_objs

    # Erase the data we are not using
    training_data.data = {objects: training_data.data[objects] for objects in training_data.object_names}
    print("Dataset reduced to {} objects".format(
        training_data.object_names.shape[0]))


def wavelet_decomposition(training_data, number_gp, **kwargs):
    """Wrapper function for `snmachine.snfeatures.WaveletFeatures`. This
    performs a wavelet decomposition on training data evaluated at 'number_gp'
    points on a light curve

    Parameters
    ----------
    training_data : snmachine.PlasticcData
        Dictionary containing the parameters that reside in the configuration
        file. This will be used to obtain the path to the training data.
    number_gp : int
        Number of points on the light curve to do wavelet analysis. Note, this
        should be an even number for the wavelet decomposition to be able to be
        performed.
    number_processes : int
        Number CPU cores avaiable to the user, this is how many cores the
        decomposition will take place over
    save_output : string
        String defining what should be saved. See docs in
        `snmachine.snfeatures.extract_wavelets` for more details on options.
    output_root : string
        Path to where one would like the uncompressed wavelet files to be stored

    Returns
    -------
    waveout: numpy.ndarray
        Numpy array of the wavelet coefficients where each row is an object and
        each column a different coefficient
    waveout_err: numpy.ndarray
        Numpy array storing the (assuming Gaussian) error on each coefficient.
    wavelet_object: snmachine.snfeatures.WaveletFeatures object

    Examples
    --------
    >>> ...
    >>> waveout, waveout_err, wavelet_object = wavelet_decomposition(
        training_data, number_gp=number_gp,
        number_processes=number_processes, save_output='all',
        output_root=directories.get("intermediate_files_directory"))
    >>> print()
    """
    wavelet_object = snfeatures.WaveletFeatures(number_gp=number_gp)
    print("WAV = {}\n".format(wavelet_object.wav))
    print("MLEV = {}\n".format(wavelet_object.mlev))
    print("number_gp = {}\n".format(number_gp))
    waveout, waveout_err = wavelet_object.extract_wavelets(training_data,
                                                           wavelet_object.wav,
                                                           wavelet_object.mlev,
                                                           **kwargs)
    return waveout, waveout_err, wavelet_object


def combine_all_features(reduced_wavelet_features, dataframe):
    # TODO: Improve docstrings.
    """Combine snmachine wavelet features with PLASTICC features. The
    user should define a dataframe they would like to merge.

    Parameters
    ----------
    reduced_wavelet_features :  astropy.table.table.Table
        These are the N principal components from the uncompressed wavelets
    dataframe : pandas.DataFrame
        Dataframe

    Returns
    -------
    combined_features : pandas.DataFrame

    Examples
    --------
    >>> ...
    >>> print(shape.reduced_wavelet_features)

    >>> print(shape.dataframe)

    >>> combined_features = combine_all_features(reduced_wavelet_features, dataframe)
    >>> print(shape.combined_features)

    """

# def merge_features(some_features, other_features):
#     # TODO: Move this to a data processing file
#     if type(some_features) != pd.core.frame.DataFrame:
#         some_features = some_features.to_pandas()
#     if type(other_features) != pd.core.frame.DataFrame:
#         other_features = other_features.to_pandas()
#     merged_df = pd.merge(some_features, other_features)
#     merged_df.set_index("Object", inplace=True)
#     return merged_df

#     meta_df = dat.metadata
#     combined_features = merge_features(wavelet_features, meta_df)
    return combined_features


def _to_pandas(features):
    # TODO: Improve docstrings.
    """Helper function to take either an astropy Table
    or numpy ndarray and convert to a pandas DataFrame representation

    Parameters
    ----------
    features: astropy.table.table.Table OR numpy.ndarray
        This parameter can be either an astropy Table or numpy ndarray
        representation of the wavelet features

    Returns
    -------
    features : pandas.DataFrame

    Examples
    --------
    >>> ...
    >>> print(type(features))
    <class 'astropy.table.table.Table'>
    >>> features = _to_pandas(features)
    >>> print(type(features))
    <class 'pandas.core.frame.DataFrame'>
    """
    if isinstance(features, np.ndarray):
        features = pd.DataFrame(features, index=training_data.object_names)
    else:
        features = features.to_pandas()

    return features


def create_classifier(combined_features, training_data, directories,
                      augmentation_method=None, random_state=42,
                      number_comps=''):
    # TODO: Improve docstrings.
    """Creation of an optimised Random Forest classifier.

    Parameters
    ----------
    combined_features : pandas.DataFrame
        This contains. Index on objects
    random_state : int
        To allow for reproducible...

    Returns
    -------
    classifer : sklearn.RandomForestClassifier object

    Examples
    --------
    >>> ...
    >>> classifier, confusion_matrix = create_classifier(combined_features)
    >>> print(classifier)
    (RandomForestClassifier(bootstrap=True, class_weight=None,
        criterion='entropy',
        max_depth=None, max_features='auto', max_leaf_nodes=None,
        min_impurity_split=1e-07, min_samples_leaf=1,
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        n_estimators=700, n_jobs=-1, oob_score=True, random_state=42,
        verbose=0, warm_start=False), array([[ 1.]]))
    """
    labels = training_data.labels.values

    X = combined_features
    y = labels

    target_names = np.unique(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    classifer = RandomForestClassifier(n_estimators=700, criterion='entropy',
                                       oob_score=True, n_jobs=-1,
                                       random_state=random_state)

    if augmentation_method in snaugment.NNAugment.methods():
        classifer = make_pipeline(eval(augmentation_method)(sampling_strategy='not majority'), classifer)
    else:
        print("No augmentation selected, proceeding without resampling of classes")

    classifer.fit(X_train, y_train)

    # Classify and report the results
    print(classification_report_imbalanced(y_test, classifer.predict(X_test)))

    y_preds = classifer.predict(X_test)
    confusion_matrix, figure = plot_confusion_matrix(y_test, y_preds,
                                                     'Validation data',
                                                     target_names,
                                                     normalize=True)

    timestamp = get_timestamp()
    with open(os.path.join(directories.get("classifications_directory"),
                           F'classifer_{number_comps}_{augmentation_method}.pkl'), 'wb') as clf:
        pickle.dump(classifer, clf)

    figure.savefig(os.path.join(directories.get("plots_directory"),
                                F'confusion_matrix_{number_comps}_{augmentation_method}.pdf'))

    return classifer, confusion_matrix


def make_predictions(location_of_test_data, classifier):
    # TODO: Move to a seperate make_predictions file
    pass


def restart_from_saved_gps(directories):
    pass


def restart_from_saved_wavelets(directories):
    pass


def restart_from_saved_pca(directories, number_of_principal_components):
    # TODO: Write docstrings
    wavelet_features = pd.read_pickle(os.path.join(directories.get("features_directory"),
        "reduced_wavelet_components_{}.pickle".format(number_of_principal_components)))
    combined_features = wavelet_features  # For running tests for now
    classifier, confusion_matrix = create_classifier(combined_features,
                                                     training_data)
    print(F"classifier = {classifier}")


if __name__ == "__main__":

    # Set the number of processes you want to use throughout the notebook
    number_processes = multiprocessing.cpu_count()
    print("Running with {} cores".format(number_processes))

    parser = ArgumentParser(description="Run pipeline end to end")
    parser.add_argument('--configuration', '-c')
    parser.add_argument('--restart-from', '-r', help='Either restart from saved "GPs" or from saved "Wavelets"', default="full")
    arguments = parser.parse_args()
    arguments = vars(arguments)

    path_to_configuration_file = arguments['configuration']
    params = load_configuration_file(path_to_configuration_file)

    data_path = params.get("data_path", None)
    analysis_directory = params.get("analysis_directory", None)
    analysis_name = params.get("analysis_name", None)

    # snmachine parameters
    number_gp = params.get("number_gp", None)
    kernel_param = params.get("kernel_param", None)
    number_of_principal_components = params.get("number_of_principal_components", None)

    # Step 1. Creat folders that contain analysis
    directories = create_folder_structure(analysis_directory, analysis_name)
    # Step 2. Save configuration file used for this analysis
    save_configuration_file(params, directories.get("analysis_directory"))
    # Step 3. Check at which point the user would like to run the analysis from.
    # If elements already saved, these will be used but this can be overriden
    # with command line argument
    if (arguments['restart_from'].lower() == "gps"):
        # Restart from saved GPs.
        pass
    elif (arguments['restart_from'].lower() == "wavelets"):
        # Restart from saved uncompressed wavelets.
        pass
    elif (arguments['restart_from'].lower() == "pca"):
        # Restart from saved PCA components
        restart_from_saved_pca(directories, number_of_principal_components)
    else:
        # Run full pipeline but still do checks to see if elements from GPs or
        # wavelets already exist on disk; the first check should be for:
        #   a. Saved PCA files
            # path_saved_reduced_wavelets = directories.get("intermediate_files_directory")
            # eigenvectors_saved_file = np.load(os.path.join(path_saved_reduced_wavelets, 'eigenvectors_' + str(number_of_principal_components) + '.npy'))
            # means_saved_file = np.load(os.path.join(path_saved_reduced_wavelets, 'means_' + str(number_of_principal_components) + '.npy'))
        #   b. Saved uncompressed wavelets
        #   c. Saved GPs

        # Step 4. Load in training data
        training_data = load_dataset(data_path)
        print("training_data = {}".format(training_data))

        # Step 5. Compute GPs
        gps.compute_gps(training_data, number_gp=number_gp, t_min=0,
                        t_max=1100, kernel_param=kernel_param,
                        output_root=directories['intermediate_files_directory'],
                        number_processes=number_processes)

        # Step 6. Extract wavelet coeffiencts
        waveout, waveout_err, wavelet_object = wavelet_decomposition(
            training_data, number_gp=number_gp,
            number_processes=number_processes, save_output='all',
            output_root=directories.get("intermediate_files_directory"))
        print("waveout = {}".format(waveout))
        print("waveout, type = {}".format(type(waveout)))

        print("waveout_err = {}".format(waveout_err))
        print("waveout_err, type = {}".format(type(waveout_err)))

        print("wavelet_object = {}".format(wavelet_object))
        print("wavelet_object, type = {}".format(type(wavelet_object)))

        # Step 7. Reduce dimensionality of wavelets by using only N principal components
        wavelet_features, eigenvals, eigenvecs, means, num_feats = wavelet_object.extract_pca(
            object_names=training_data.object_names, wavout=waveout,
            recompute_pca=True, method='svd',
            number_comp=number_of_principal_components, tol=None,
            pca_path=None, save_output=True,
            output_root=directories.get("features_directory"))
        print("wavelet_features = {}".format(wavelet_features))
        print("wavelet_features, type = {}".format(type(wavelet_features)))

        # Step 8. TODO Combine snmachine features with user defined features

        # Step 9. TODO Create a Random Forest classifier; need to fit model and save it.
        combined_features = wavelet_features  # For running tests for now
        classifier = create_classifier(combined_features, training_data)
        print(F"classifier = {classifier}")

        # Step 10. TODO Use saved classifier to make predictions. This can occur using a seperate file
