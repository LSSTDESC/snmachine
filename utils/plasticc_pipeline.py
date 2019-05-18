"""
Machine learning pipeline for the PLAsTiCC competition using snmachine codebase.
"""
from plasticc_utils import plasticc_log_loss, plot_confusion_matrix
from astropy.table import Table
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import os
import subprocess
import multiprocessing
import yaml
import warnings
warnings.filterwarnings("ignore")
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
try:
    from snmachine import snfeatures, sndata, snaugment, gps
except ImportError:
    print("Unable to import snmachine. Check environment set correctly")


def get_git_revision_short_hash():
    """ Helper function to obtain current version control hash value

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


def get_timestamp(path_to_configuration_file):
    """ Helper function to obtain latest modified time of the configuration file

    Parameters
    ----------
    path_to_configuration_file : str
        System path to where the configuration file is located

    Returns
    -------
    timestamp : str
        Short representation of last modified time for the configuration file used.
        'YYYY-MM-DD-HOURMINUTE'

    Examples
    --------
    >>> ...
    >>> timestamp = get_timestamp(path_to_configuration_file)
    >>> print(timestamp)
    '2019-05-18-2100'
    """
    _timestamp = subprocess.check_output(['date', '+%Y-%m-%d-%H%M', '-r', path_to_configuration_file])
    return _timestamp.decode("utf-8").rstrip()


def create_folder_structure(analysis_directory, analysis_name, path_to_configuration_file):
    """ Make directories that will be used for analysis

    Parameters
    ----------
    analysis_directory : str
        System path to where the user would like to contain
        a run of the analysis
    analysis_name : str
        Given name of analysis run. This is appended with the current git hash
        the code has been run with.

    Returns
    -------
    dirs: dict
        Dictionary containing the mapping of folders that have been created.

    Examples
    --------
    Each folder name can then be accessed with dictionary methods:
    >>> ...
    >>> analysis_directory = params.get("analysis_directory", None)
    >>> analysis_name = params.get("analysis_name", None)
    >>> directories = create_folder_structure(analysis_directory, analysis_name, path_to_configuration_file)
    >>> print(directories.get("method_directory", None))

    """
    # Prepend last modified time of configuration file and git SHA to analysis name
    analysis_name = get_timestamp(path_to_configuration_file) + "-" + get_git_revision_short_hash() + "-" + analysis_name

    method_directory = os.path.join(analysis_directory, analysis_name)
    features_directory = os.path.join(method_directory, 'wavelet_features')
    classifications_directory = os.path.join(method_directory, 'classifications')
    intermediate_files_directory = os.path.join(method_directory, 'intermediate_files')
    plots_directory = os.path.join(method_directory, 'plots')

    dirs = {"method_directory": method_directory, "features_directory": features_directory,
            "classifications_directory": classifications_directory, "intermediate_files_directory": intermediate_files_directory,
            "plots_directory": plots_directory}

    for key, value in dirs.items():
        subprocess.call(['mkdir', value])

    return dirs


def load_configuration_file(path_to_configuration_file):
    # TODO: Finish doctring examples
    """ Load from disk the configuration file that is to be used

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
    >>> data_path = params.get("data_path", None)
    >>> print(data_path)

    >>> number_gp = params.get("number_gp", None)
    >>> print(number_gp)

    """
    try:
        with open(path_to_configuration_file) as f:
            params = yaml.load(f)
    except IOError:
        print("Invalid yaml file provided")
        exit()
    print("The parameters are:\n {}".format(params))
    return params


def save_configuration_file(method_directory):
    # TODO: Provide a doctring example
    """ Make a copy of the configuration file that has been used inside the
    analysis directory

    Parameters
    ----------
    method_directory : string
        The folder path used for this analysis

    Returns
    -------
    None

    Examples
    --------
    >>> ...
    >>> save_configuration_file(method_directory)
    >>> print()

    """
    with open(os.path.join(method_directory, "config.yml"), 'w') as config:
            yaml.dump(params, config, default_flow_style=False)


def load_training_data(data_path):
    # TODO: Finish doctring examples
    """ Load from disk the training data one will use for this analysis

    Parameters
    ----------
    params : dict
        Dictionary containing the parameters that reside in the configuration
        file. This will be used to obtain the path to the training data.

    Returns
    -------
    training_data : snmachine.PlasticcData
        snmachine.PlasticcData instance of the training data

    Examples
    --------
    >>> ...
    >>> training_data = load_training_data(params)
    >>> print(training_data)

    """
    try:
        if data_path.lower().endswith((".pickle", ".pkl", ".p", ".pckl")):
            with open(data_path, 'rb') as input:
                print("Opening from binary pickle")
                training_data = pickle.load(input)
                print("Dataset loaded from pickle file as: {}".format(training_data))
        else:
            folder_path, train_data_file_name = os.path.split(data_path)
            meta_data_file_name = "_metadata.".join(train_data_file_name.split("."))

            print("Opening from CSV")
            training_data = sndata.PlasticcData(folder=folder_path, data_file=train_data_file_name,
                                                metadata_file=meta_data_file_name, cut_non_detections=False)
            print("Dataset loaded from csv file as: {}".format(training_data))
            print("Saving {} object to pickle binary".format(training_data))

            dat_binary = os.path.splitext(train_data_file_name)[0] + ".pckl"
            print(os.path.join(folder_path, dat_binary))
            with open(os.path.join(folder_path, dat_binary), 'wb') as f:
                pickle.dump(training_data, f, pickle.HIGHEST_PROTOCOL)
    except FileNotFoundError:
        print("No file found to load")
        exit()

    return training_data


def reduce_size_of_training_data(training_data, dirs, subset_size, seed=1234):
    # TODO: Incorpate further doctrings and finish examples. Tarek: Catarina and I need to
    # discuss this further. There is some overlap between this and
    # sndata.PlasticcData.update_data() and it would be good to comebine this.
    """ Load from disk the training data one will use for this analysis

    Parameters
    ----------
    training_data : snmachine.PlasticcData
        Dictionary containing the parameters that reside in the configuration
        file. This will be used to obtain the path to the training data.
    dirs : dict
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
    >>> print(shape.training_data)

    >>> new_training_data = reduce_size_of_training_data(training_data, dirs, 1000))
    >>> print(shape.new_training_data)

    """

    method_directory = dirs.get("method_directory", None)
    subset_file = os.path.join(method_directory, "subset.list")
    if os.path.exists(subset_file):
        rand_objs = np.genfromtxt(subset_file, dtype='U')
    else:
        np.random.seed(seed)
        rand_objs = np.random.choice(training_data.object_names, replace=False, size=subset_size)
        rand_objs_sorted_int = np.sort(rand_objs.astype(np.int))
        rand_objs = rand_objs_sorted_int.astype('<U9')
        np.savetxt(subset_file, rand_objs, fmt='%s')

    training_data.object_names = rand_objs

    # Erase the data we are not using
    training_data.data = {objects: training_data.data[objects] for objects in training_data.object_names}
    print("Dataset reduced to {} objects".format(training_data.object_names.shape[0]))


def wavelet_decomposition(training_data, number_gp, **kwargs):
    """ Load from disk the training data one will use for this analysis

    Parameters
    ----------
    training_data : snmachine.PlasticcData
        Dictionary containing the parameters that reside in the configuration
        file. This will be used to obtain the path to the training data.
    dirs : dict
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
    >>> waveout, waveout_err, wavelet_object =
    wavelet_decomposition(training_data, number_gp=number_gp, number_processes=number_processes,
                                                                     save_output='all', output_root=dirs.get("intermediate_files_directory"))
    >>> print()

    """

    wavelet_object = snfeatures.WaveletFeatures(number_gp=number_gp)
    print("WAV = {}\n".format(wavelet_object.wav))
    print("MLEV = {}\n".format(wavelet_object.mlev))
    print("number_gp = {}\n".format(number_gp))
    waveout, waveout_err = wavelet_object.extract_wavelets(training_data, wavelet_object.wav, wavelet_object.mlev, **kwargs)
    return waveout, waveout_err, wavelet_object

# def merge_features(some_features, other_features):
#     # TODO: Move this to a data processing file
#     if type(some_features) != pd.core.frame.DataFrame:
#         some_features = some_features.to_pandas()
#     if type(other_features) != pd.core.frame.DataFrame:
#         other_features = other_features.to_pandas()
#     merged_df = pd.merge(some_features, other_features)
#     merged_df.set_index("Object", inplace=True)
#     return merged_df


def combine_all_features(reduced_wavelet_features, dataframe):
    # TODO: Improve docstrings. Discuss whether the user should pass in a CSV
    # instead?
    """ Combine snmachine wavelet features with PLASTICC features. The
    user should define a dataframe they would like to merge.

    Parameters
    ----------
    reduced_wavelet_features : numpy.ndarray
        These are the N principle components from the uncompressed wavelets
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
    meta_df = dat.metadata
    combined_features = merge_features(wavelet_features, meta_df)
    return combined_features


def create_classififer(combined_features, random_state=42):
    # TODO: Improve docstrings.
    """ Creation of an optimised Random Forest classifier.

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
    >>> classifier, confusion_matrix = create_classififer(combined_features)
    >>> print(classifier)

    >>> plot_confusion_matrix(confusion_matrix)

    """

    X = combined_features.drop('target', axis=1)
    y = combined_features['target'].values

    print("X SHAPE = {}\n".format(X.shape))
    print("y SHAPE = {}\n".format(y.shape))

    target_names = combined_features['target'].unique()

    print("X = \n{}".format(X))
    print("y = \n{}".format(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    classifer = RandomForestClassifier(n_estimators=700, criterion='entropy',
                                       oob_score=True, n_jobs=-1,
                                       random_state=random_state)

    classifer.fit(X_train, y_train)

    y_preds = classifer.predict(X_test)

    confusion_matrix = plot_confusion_matrix(y_test, y_preds, 'Validation data', target_names)

    y_probs = classifer.predict_proba(X_test)

    nlines = len(target_names)
    # we also need to express the truth table as a matrix
    sklearn_truth = np.zeros((len(y_test), nlines))
    label_index_map = dict(zip(classifer.classes_, np.arange(nlines)))
    for i, x in enumerate(y_test):
            sklearn_truth[i][label_index_map[y_test[i]]] = 1

    weights = np.array([1/18, 1/9, 1/18, 1/18, 1/18, 1/18, 1/18, 1/9, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/19])

    # weights[:-1] to ignore last class, the anomaly class
    log_loss = plasticc_log_loss(sklearn_truth, y_probs, relative_class_weights=weights[:-1])
    print("LogLoss: {:.3f}\nBest Params: {}".format(log_loss, classifer.get_params))

    return classifer, confusion_matrix


def make_predictions(location_of_test_data, classifier):
    # TODO: Move to a seperate make_predictions file
    pass


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
    dirs = create_folder_structure(analysis_directory, analysis_name, path_to_configuration_file)
    # Step 2. Save configuration file used for this analysis
    save_configuration_file(dirs.get("method_directory"))
    # Step 3. Check at which point the user would like to run the analysis from.
    # If elements already saved, these will be used but this can be overriden
    # with command line argument
    if (arguments['restart_from'].lower() == "wavelets"):
        # Restart from saved uncompressed wavelets.
        wavelet_features = Table.read(dirs.get("features_dir") + "/wavelet_features.fits")
        combined_features = combine_all_features(wavelet_features, data_path)
        classifer = create_classififer(combined_features)
    elif (arguments['restart_from'].lower() == "gps"):
        # Restart from saved GPs.
        pass
    else:
        # Run full pipeline but still do checks to see if elements from GPs or
        # wavelets already exist on disk; the first check should be for:
        #   a. Saved PCA files
            # path_saved_reduced_wavelets = dirs.get("intermediate_files_directory")
            # eigenvectors_saved_file = np.load(os.path.join(path_saved_reduced_wavelets, 'eigenvectors_' + str(number_of_principal_components) + '.npy'))
            # means_saved_file = np.load(os.path.join(path_saved_reduced_wavelets, 'means_' + str(number_of_principal_components) + '.npy'))
        #   b. Saved uncompressed wavelets
        #   c. Saved GPs

        # Step 4. Load in training data
        training_data = load_training_data(data_path)

        # Step 5. Compute GPs
        gps.compute_gps(training_data, number_gp=number_gp, t_min=0, t_max=1100,
                        kernel_param=kernel_param,
                        output_root=dirs['intermediate_files_directory'],
                        number_processes=number_processes)

        # Step 6. Extract wavelet coeffiencts
        waveout, waveout_err, wavelet_object = wavelet_decomposition(training_data, number_gp=number_gp, number_processes=number_processes,
                                                                     save_output='all', output_root=dirs.get("intermediate_files_directory"))

        # Step 7. Reduce dimensionality of wavelets by using only N principle components
        wavelet_features, eigenvals, eigenvecs, means, num_feats = wavelet_object.extract_pca(object_names=training_data.object_names, wavout=waveout, recompute_pca=True, method='svd', ncomp=number_of_principal_components,
                                                                                              tol=None, pca_path=None, save_output=True, output_root=dirs.get("intermediate_files_directory"))

        # Step 8. TODO Combine snmachine features with user defined features
        # Step 9. TODO Create a Random Forest classifier; need to fit model and
        # save it.

        # Step 10. TODO Use saved classifier to make predictions. This can occur using a seperate file
