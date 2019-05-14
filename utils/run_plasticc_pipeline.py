"""
Machine learning pipeline for the PLAsTiCC competition using snmachine codebase
"""

import numpy as np
import pandas as pd
import sys
import os
import subprocess
import multiprocessing
from astropy.table import Table
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from argparse import ArgumentParser
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

util_module_path = os.path.abspath(os.path.join('snmachine', 'utils'))
if util_module_path not in sys.path:
    sys.path.append(util_module_path)
from plasticc_utils import plasticcLogLoss, plotConfusionMatrix


def create_folder_structure(ANALYSIS_DIR, ANALYSIS_NAME):

    method_dir = os.path.join(ANALYSIS_DIR, ANALYSIS_NAME)
    features_dir = os.path.join(method_dir, 'wavelet_features')
    classif_dir = os.path.join(method_dir, 'classifications')
    interm_dir = os.path.join(method_dir, 'intermediate')
    plots_dir = os.path.join(method_dir, 'plots')

    dirs = {"method_dir": method_dir, "features_dir": features_dir,
            "classif_dir": classif_dir, "interm_dir": interm_dir,
            "plots_dir": plots_dir}

    for key, value in dirs.items():
        subprocess.call(['mkdir', value])

    return dirs


def save_configuration_file(dirs):

    METHOD_DIR = dirs.get("method_dir", None)
    with open('/{}/config.yaml'.format(METHOD_DIR), 'w') as config:
            yaml.dump(params, config, default_flow_style=False)


def load_dataset(DATA_PATH):

    try:
        if DATA_PATH.lower().endswith((".pickle", ".pkl", ".p", ".pckl")):
            with open(DATA_PATH, 'rb') as input:
                print("Opening from binary pickle")
                dat = pickle.load(input)
                print("Dataset loaded from pickle file as: {}".format(dat))
        else:

            folder, data_file = os.path.split(DATA_PATH)
            print(folder, data_file)
            meta_file = "_metadata.".join(data_file.split("."))

            print("Opening from CSV")
            dat = sndata.PlasticcData(folder=folder, data_file=data_file, meta_file=meta_file, from_pickle=False)
            print("Dataset loaded from csv file as: {}".format(dat))
            print("Saving {} object to pickle binary".format(dat))

            dat_binary = os.path.splitext(data_file)[0]+".pckl"
            print(os.path.join(folder, dat_binary))
            with open(os.path.join(folder, dat_binary), 'wb') as f:
                pickle.dump(dat, f, pickle.HIGHEST_PROTOCOL)
    except FileNotFoundError:
        print("Oii, load something !!")

    return dat


def reduce_dataset(dat, dirs, subset_size, SEED):

    METHOD_DIR = dirs.get("method_dir", None)
    subset_file = '/{}/subset.list'.format(METHOD_DIR)
    if os.path.exists(subset_file):
        rand_objs = np.genfromtxt(subset_file, dtype='U')
    else:
        np.random.seed(SEED)
        rand_objs = np.random.choice(dat.object_names, replace=False, size=subset_size)
        rand_objs_sorted_int = np.sort(rand_objs.astype(np.int))
        rand_objs = rand_objs_sorted_int.astype('<U9')
        np.savetxt(subset_file, rand_objs, fmt='%s')

    dat.object_names = rand_objs
    dat.data = {objects: dat.data[objects] for objects in dat.object_names}  # Erase the data we are not using

    print("Dataset reduced to {} objects".format(dat.object_names.shape[0]))

    return dat  # Cat: I don't think we need to return anything


def fit_gaussian_process(dat, **kwargs):  # Cat: Do we really want a mask funtion?
    # Tarek: Now that this file lives in snmachine and with the extensive
    # refactoring this is no longer necessary I believe

    # extract_GP(dat, **kwargs)
    # snfeatures.WaveletFeatures.extract_GP(dat, **kwargs)
    pass


def wavelet_decomposition(dat, ngp, **kwargs):  # Cat: we need to add ngp as input otherwise it doesn't run on the notebbok

    wavelet_object = snfeatures.WaveletFeatures(ngp=ngp)
    print("WAV = {}\n".format(wavelet_object.wav))
    print("MLEV = {}\n".format(wavelet_object.mlev))
    print("NGP = {}\n".format(ngp))
    waveout, waveout_err = wavelet_object.extract_wavelets(dat, wavelet_object.wav, wavelet_object.mlev, **kwargs)
    return waveout, waveout_err, wavelet_object


def dimentionality_reduction(wavelet_object, dirs, object_names, waveout, tolerance, **kwargs): # Cat: we need to add tolerance

    # check if reduced wavelet features already exist
    wavelet_features, eigenvalues, eigenvectors, means, num_feats = wavelet_object.extract_pca(object_names, waveout, **kwargs)

    output_root = dirs.get("features_dir")
    print("Inside dimRedux: {}\n".format(output_root))
    wavelet_features.write('{}/wavelet_features_{}.fits'.format(output_root, str(tolerance)[2:]))

    return wavelet_features, eigenvalues, eigenvectors, means


def merge_features(some_features, other_features):
    if type(some_features) != pd.core.frame.DataFrame:
        some_features = some_features.to_pandas()
    if type(other_features) != pd.core.frame.DataFrame:
        other_features = other_features.to_pandas()
    merged_df = pd.merge(some_features, other_features)
    merged_df.set_index("Object", inplace=True)
    return merged_df


def combine_additional_features(wavelet_features, dat):
    # Combine snmachine wavelet features with PLASTICC features. Allow user to
    # define the dataframe they would like to merge
    meta_df = dat.metadata
    combined_features = merge_features(wavelet_features, meta_df)
    return combined_features


def create_classififer(combined_features, RANDOM_STATE):

    X = combined_features.drop('target', axis=1)
    y = combined_features['target'].values

    print("X SHAPE = {}\n".format(X.shape))
    print("y SHAPE = {}\n".format(y.shape))

    target_names = combined_features['target'].unique()

    print("X = \n{}".format(X))
    print("y = \n{}".format(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

    clf = RandomForestClassifier(n_estimators=700, criterion='entropy',
                                 oob_score=True, n_jobs=-1,
                                 random_state=RANDOM_STATE)

    clf.fit(X_train, y_train)

    y_preds = clf.predict(X_test)

    # confm = plotConfusionMatrix(y_test, y_preds, 'Test data', target_names)

    y_probs = clf.predict_proba(X_test)

    nlines = len(target_names)
    # we also need to express the truth table as a matrix
    sklearn_truth = np.zeros((len(y_test), nlines))
    label_index_map = dict(zip(clf.classes_, np.arange(nlines)))
    for i, x in enumerate(y_test):
            sklearn_truth[i][label_index_map[y_test[i]]] = 1

    weights = np.array([1/18, 1/9, 1/18, 1/18, 1/18, 1/18, 1/18, 1/9, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/19])

    logloss = plasticcLogLoss(sklearn_truth, y_probs, relative_class_weights=weights[:-1])
    print("LogLoss: {:.3f}\nBest Params: {}".format(logloss, clf.get_params))

    # PASS IN TRAINING DATA IN FORM OF SNMACHINE OBJECT
    # CREATE rf OBJECT.
    # CROSS-VAIDATION HERE
    # RETURN CLASSIFIFER OBJECT
    return clf


def make_predictions(LOCATION_OF_TEST_DATA, CLASSIFIER):
    # LOAD TEST SET AT THIS POINT
    # USE CLASFFIFER FROM createClassififer, BY USING THAT WE THEN
    # clf.predict(test_set)
    # RETURN SUBMISSION_FILE_WITHOUT_99
    pass


def run_full_pipeline():
    pass


def restart_from_saved_gps():
    pass


def restart_from_save_wavelets():
    pass


def load_configuration_file(path_to_configuration_file):
    # LOAD CONFIGURATION FILE --->>>> COULD BE ITS OWN LOAD CONFIGURATION FUNCTION?
    try:
        with open(arguments.configuration) as f:
            params = yaml.load(f)
    except IOError:
        print("Invalid yaml file provided")
        exit()

    print("The PARAMS are:\n {}".format(params))

    return params


if __name__ == "__main__":

    parser = ArgumentParser(description="Run pipeline end to end")
    parser.add_argument('--configuration', '-c')
    parser.add_argument('--restart-from', '-r', help='Either restart from saved "GPs" or from saved "Wavelets"', default="full")
    arguments = parser.parse_args()

    params = load_configuration_file(arguments.configuration)

    # global settings
    RANDOM_STATE = params.get("RANDOM_STATE", None)
    # Tarek: maybe remove this completely and
    # set inside a function call itself, i.e. have a default which can be
    # overridden
    SEED = params.get("SEED", None)
    DATA_PATH = params.get("DATA_PATH", None)
    ANALYSIS_DIR = params.get("ANALYSIS_DIR", None)
    ANALYSIS_NAME = params.get("ANALYSIS_NAME", None)

    # snmachine parameters
    ngp = params.get("ngp", None)
    initheta = params.get("initheta", None)

    # Set the number of processes you want to use throughout the notebook
    nprocesses = multiprocessing.cpu_count()
    print("Running with {} cores".format(nprocesses))

    dirs = create_folder_structure(ANALYSIS_DIR, ANALYSIS_NAME)
    save_configuration_file(dirs)

    # RUN PIPELINE
    if (arguments.restart.lower() == "wavelets"):

        wavelet_features = Table.read(dirs.get("features_dir")+"/wavelet_features.fits")
        combined_features = combine_additional_features(wavelet_features, DATA_PATH)
        classifer = create_classififer(combined_features)

    elif (arguments.restart.lower() == "gps"):
        print("Hello")
    else:
        print("Running full pipeline .. ")

        dat = load_dataset(DATA_PATH)
        # dat = reduceDataset(dat, dirs, subset_size=10, SEED=SEED)
        fit_gaussian_process(dat, ngp=ngp, t_min=0, initheta=initheta,
                             nprocesses=nprocesses, output_root=dirs.get("interm_dir"), t_max=1100)

        waveout, waveout_err, wavelet_object = wavelet_decomposition(dat, ngp=ngp, nprocesses=nprocesses, save_output='all', output_root=dirs.get("interm_dir"))

        wavelet_features, eigenvalues, eigenvectors, means = dimentionality_reduction(wavelet_object, dirs, dat.object_names.copy(), waveout, tolerance=0.99, save_output=True, recompute_pca=True, output_root=dirs.get("features_dir"))

        combined_features = combine_additional_features(wavelet_features, DATA_PATH)
        classifer = create_classififer(combined_features)
        # snmachine.utils.fit_gaussian_process.extract_GP()
        # check for wavelets, if so restartFromWavelets()
        # else, check for gp's, if so restartFromGPs()
        # otherwise runFullPipeline()
