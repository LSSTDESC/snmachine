"""
Utility module mostly wrapping sklearn functionality and providing utility functions.
"""

from __future__ import division
from past.builtins import basestring
import numpy as np
import os
if not 'DISPLAY' in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import neighbors
from sklearn import grid_search
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,  AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.integrate import trapz
from astropy.table import Table,join,unique
import sys, collections,time
from functools import partial
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler

#This allows the user to easily loop through all possible classifiers
choice_of_classifiers=['svm', 'knn', 'random_forest', 'decision_tree','boost_dt','boost_rf', 'nb']
#boost_rf is a set of boosted random forests which Max came up with.

try:
    from sklearn.neural_network import MLPClassifier
    choice_of_classifiers.append('neural_network')
except ImportError:
    print ('Neural networks not available in this version of scikit-learn. Neural networks are available from development version 0.18.')

def roc(pr, Yt, num_classes, true_class=0):
    """
    Produce the false positive rate and true positive rate required to plot
    a ROC curve, and the area under that curve.

    Parameters
    ----------
    pr : array
        An array of probability scores, either a 1d array of size N_samples or an nd array,
	in which case the column corresponding to the true class will be used.
    Yt : array
        An array of class labels, of size (N_samples,)
    num_classes : int
        How many classes are we considering to compare against
    true_class : int
        which class is taken to be the "true class" (e.g. Ia vs everything else)

    Returns
    -------
    fpr : array
        An array containing the false positive rate at each probability threshold
    tpr : array
        An array containing the true positive rate at each probability threshold
    auc : float
        The area under the ROC curve

    """
    probs = pr.copy()
    Y_test = Yt.copy()
    min_class = (int)(Y_test.min())  # This is to deal with starting class assignment at 1.
    Y_test = Y_test.squeeze()

    if len(pr.shape)>1:
        # probs_1 = probs[:, true_class-min_class]
        # probs_1 = probs[:, 7]
        probs_1 = probs[:, num_classes-1] # -1 since 0 based indexing of numpy
    else:
        probs_1 = probs

    threshold = np.linspace(0., 1., 50)  # 50 evenly spaced numbers between 0,1

    # This creates an array where each column is the prediction for each threshold
    preds = np.tile(probs_1, (len(threshold), 1)).T >= np.tile(threshold, (len(probs_1), 1))
    Y_bool = (Y_test == true_class)
    Y_bool = np.tile(Y_bool, (len(threshold), 1)).T

    TP = (preds & Y_bool).sum(axis=0)
    FP = (preds & ~Y_bool).sum(axis=0)
    TN = (~preds & ~Y_bool).sum(axis=0)
    FN = (~preds & Y_bool).sum(axis=0)

    tpr = np.zeros(len(TP))
    tpr[TP != 0] = TP[TP != 0]/(TP[TP != 0] + FN[TP != 0])
    fpr = FP/(FP+TN)

    fpr = np.array(fpr)[::-1]
    tpr = np.array(tpr)[::-1]

    auc = trapz(tpr, fpr)

    return fpr, tpr, auc

def plot_roc(fpr, tpr, auc, labels=[], cols=[],  label_size=26, tick_size=18, line_width=3, figsize=(8,6)):
    """
    Plots a ROC curve or multiple curves. Can plot the results from multiple classifiers if fpr and tpr are arrays
    where each column corresponds to a different classifier.

    Parameters
    ----------
    fpr : array
        An array containing the false positive rate at each probability threshold
    tpr : array
        An array containing the true positive rate at each probability threshold
    auc : float
        The area under the ROC curve
    labels : list, optional
        Labels of each curve (e.g. ML algorithm names)
    cols : list, optional
        Colors of the line(s)
    label_size : float, optional
        Size of x and y axis labels.
    tick_size: float, optional
        Size of tick labels.
    line_width : float, optional
        Line width
    """

    #Automatically fill in the colors if not supplied
    if not isinstance(cols, basestring) and len(cols)==0:
        cols=['#185aa9','#008c48','#ee2e2f','#f47d23','#662c91','#a21d21','#b43894','#010202']


    #This should work regardless of whether it's one or many roc curves
    #fig=plt.figure(figsize=figsize)
    fig=plt.gcf()
    ax=fig.add_subplot(111)
    ax.set_color_cycle(cols)
    ax.plot(fpr, tpr, lw=line_width)
    # ax.plot(fpr, tpr)

    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    plt.xlabel('False positive rate (contamination)', fontsize=label_size)
    plt.ylabel('True positive rate (completeness)', fontsize=label_size)
#     plt.xlabel('False positive rate (contamination)')
#     plt.ylabel('True positive rate (completeness)')

    #Robust against the possibility of AUC being a single number instead of list
    if not isinstance(auc, collections.Sequence):
        auc=[auc]

    if len(labels)>0:
        labs=[]
        for i in range(len(labels)):
            labs.append(labels[i]+' (%.3f)' %(auc[i]))
    else:
        labs=np.array(range(len(ax.lines)), dtype='str')
        for i in range(len(labs)):
            labs[i]=(labs[i]+' (%.3f)' %(auc[i]))
    #plt.legend(labs, loc='lower right',  fancybox=True, bbox_to_anchor=(0.95, 0.05), fontsize=label_size)
    plt.legend(labs, loc='lower right',  bbox_to_anchor=(0.95, 0.05))
    plt.tight_layout()
    #plt.show()

def F1(pr,  Yt, true_class, full_output=False):
    """
    Calculate an F1 score for many probability threshold increments
    and select the best one.

    Parameters
    ----------
    pr : array
        An array of probability scores, either a 1d array of size N_samples or an nd array,
        in which case the column corresponding to the true class will be used.
    Yt : array
        An array of class labels, of size (N_samples,)
    true_class : int
        which class is taken to be the "true class" (e.g. Ia vs everything else)
    full_output : bool, optional
        If true returns two vectors corresponding to F1 as a function of threshold, instead of the best value.

    Returns
    -------
    best_F1 : float
        (If full_output=False) The largest F1 value
    best_threshold : array
        (If full_output=False) The probability threshold corresponding to best_F1
    f1  : array
        (If full_output=True) F1 as a function of threshold.
    threshold  : array
        (If full_output=True) Vector of thresholds (from 0 to 1)

    """

    probs=pr.copy()
    Y_test=Yt.copy()
    min_class=Y_test.min() #This is to deal with starting class assignment at 1.
    Y_test = Y_test.squeeze()

    if len(pr.shape)>1:
        # probs_1=probs[:, true_class-min_class]
        probs_1 = probs[:, 7]
    else:
        probs_1=probs

    threshold = np.arange(0, 1, 0.01)

    #This creates an array where each column is the prediction for each threshold
    preds=np.tile(probs_1, (len(threshold), 1)).T>np.tile(threshold, (len(probs_1), 1))
    Y_bool=(Y_test==true_class)
    Y_bool=np.tile(Y_bool, (len(threshold), 1)).T

    TP=(preds & Y_bool).sum(axis=0)
    FP=(preds & ~Y_bool).sum(axis=0)
    TN=(~preds & ~Y_bool).sum(axis=0)
    FN=(~preds & Y_bool).sum(axis=0)

    f1=np.zeros(len(TP))
    f1[TP!=0]=2*TP[TP!=0]/(2*TP[TP!=0]+FN[TP!=0]+FP[TP!=0])

    if full_output:
        return f1, threshold
    else:
        best_F1 = f1.max()
        best_threshold_index = np.argmax(f1)
        best_threshold = threshold[best_threshold_index]

        return best_F1, best_threshold

def FoM(pr,  Yt, num_classes, true_class=1, full_output=False):
    """
    Calculate a Kessler FoM for many probability threshold increments
    and select the best one.

    FoM is defined as:
    FoM = TP^2/((TP+FN)(TP+3*FP))

    Parameters
    ----------
    pr : array
        An array of probability scores, either a 1d array of size N_samples or an nd array,
        in which case the column corresponding to the true class will be used.
    Yt : array
        An array of class labels, of size (N_samples,)
    num_classes : int
        How many classes are we considering to compare against
    true_class : int
        which class is taken to be the "true class" (e.g. Ia vs everything else)
    full_output : bool, optional
        If true returns two vectors corresponding to F1 as a function of threshold, instead of the best value.

    Returns
    -------
    best_FoM : float
        (If full_output=False) The largest FoM value
    best_threshold : array
        (If full_output=False) The probability threshold corresponding to best_FoM
    fom  : array
        (If full_output=True) FoM as a function of threshold.
    threshold  : array
        (If full_output=True) Vector of thresholds (from 0 to 1)

    """
    weight = 3.0

    probs=pr.copy()
    Y_test=Yt.copy()
    min_class=Y_test.min() #This is to deal with starting class assignment at 1.
    Y_test = Y_test.squeeze()

    if len(pr.shape)>1:
        # probs_1=probs[:, true_class-min_class]
        # probs_1 = probs[:, 7]
        probs_1 = probs[:, num_classes-1] # -1 since 0 based indexing of numpy
    else:
        probs_1=probs

    threshold = np.arange(0, 1, 0.01)

    #This creates an array where each column is the prediction for each threshold
    preds=np.tile(probs_1, (len(threshold), 1)).T>np.tile(threshold, (len(probs_1), 1))
    Y_bool=(Y_test==true_class)
    Y_bool=np.tile(Y_bool, (len(threshold), 1)).T

    TP=(preds & Y_bool).sum(axis=0)
    FP=(preds & ~Y_bool).sum(axis=0)
    TN=(~preds & ~Y_bool).sum(axis=0)
    FN=(~preds & Y_bool).sum(axis=0)

    fom=np.zeros(len(TP))
    fom[TP!=0]=TP[TP!=0]**2/(TP[TP!=0]+FN[TP!=0])/(TP[TP!=0]+weight*FP[TP!=0])

    if full_output:
        return fom, threshold

    else:
        best_FoM = fom.max()
        best_threshold_index = np.argmax(fom)
        best_threshold = threshold[best_threshold_index]

        return best_FoM, best_threshold


class OptimisedClassifier():
    """Implements an optimised classifier (although it can be run without optimisation). Equipped with interfaces to several
    sklearn classes and functions.
    """

    NB_param_dict = {}
    KNN_param_dict = {'n_neighbors':list(range(1, 180, 5)), 'weights':['distance']}
    SVM_param_dict = {'C':np.logspace(-2, 5, 5), 'gamma':np.logspace(-8, 3, 5)}
    NN_param_dict={'hidden_layer_sizes':[(l,) for l in range(80, 120, 5)]}

    DT_param_dict={'criterion':['gini','entropy'],'min_samples_leaf':list(range(1,400,25))}
    RF_param_dict = {'n_estimators':list(range(200, 900, 100)), 'criterion':['gini', 'entropy']}
    ests=[DecisionTreeClassifier(criterion='entropy',min_samples_leaf=l) for l in range(5, 55, 10)]
    Boost_param_dict={'base_estimator':ests,'n_estimators':list(range(5, 85, 10))}
    #This is a strange boosted random forest classifier that Max came up that works quite well, but is likely biased
    #in general
    Boost_RF_param_dict = {'base_estimator':[RandomForestClassifier(400, 'entropy'),
                                             RandomForestClassifier(600, 'entropy')],'n_estimators':list([2, 3, 5, 10])}


    #Dictionary to hold good default ranges for parameters for each classifier.
    params={'svm':SVM_param_dict, 'knn':KNN_param_dict, 'decision_tree':DT_param_dict,'random_forest':RF_param_dict,
            'boost_dt':Boost_param_dict, 'boost_rf':Boost_RF_param_dict, 'nb':NB_param_dict, 'neural_network':NN_param_dict}

    def __init__(self, classifier, num_classes, optimise=True,  **kwargs):
        """
        Wrapper around sklearn classifiers

        Parameters
        ----------
        classifier : str or sklearn.Classifier
            Either a string (one of choice_of_classifiers) or sklearn classifier object
        optimise : bool, optional
            Whether or not to optimise the parameters
        kwargs : optional
            Keyword arguments passed directly to the classifier
        """

        self.num_classes=num_classes

        self.classifier_name=classifier
        if isinstance(classifier, basestring):
            #Choose from available classifiers, with default parameters best for SNe (unless overridden)
            if 'probability' in kwargs:
                self.prob=kwargs.pop('probability')
            else:
                self.prob=True #By default we want to always return probabilities. User can override this.

            try:
                if classifier=='svm':
                    if 'kernel' in kwargs:
                        kern=kwargs.pop('kernel') #Removes this from kwargs
                    else:
                        kern='rbf'
                    self.clf=svm.SVC(kernel=kern, probability = self.prob, **kwargs)

                elif classifier=='knn':
                    if 'n_neighbours' in kwargs:
                        n=kwargs.pop('n_neighbours')
                    else:
                        n=5
                    if 'weights' in kwargs:
                        wgts=kwargs.pop('weights')
                    else:
                        wgts='distance'
                    self.clf=neighbors.KNeighborsClassifier(n_neighbors = n,  weights=wgts, **kwargs)

                elif classifier=='random_forest':
                    self.clf = RandomForestClassifier(**kwargs)
                elif classifier=='decision_tree':
                    self.clf = DecisionTreeClassifier(**kwargs)
                elif classifier=='boost_dt' or classifier=='boost_rf':
                    self.clf = AdaBoostClassifier(**kwargs)
                elif classifier=='nb':
                    self.clf = GaussianNB(**kwargs)
                elif classifier=='neural_network':
                    if 'hidden_layer_sizes' in kwargs:
                        l=kwargs.pop('hidden_layer_sizes')
                    else:
                        l=(5, 2)
                    if 'algorithm' in kwargs:
                        algo=kwargs.pop('algorithm')
                    else:
                        algo='adam'
                    if 'activation' in kwargs:
                        activation=kwargs.pop('activation')
                    else:
                        activation='tanh'
                    self.clf = MLPClassifier(solver=algo, hidden_layer_sizes=l,activation=activation,  **kwargs)
                elif classifier=='multiple_classifier':
                    pass
                else:
                    print ('Requested classifier not recognised.' )
                    print ('Choice of built-in classifiers:')
                    print (choice_of_classifiers)
                    sys.exit(0)

            except TypeError:
                #Gracefully catch errors of sending the wrong kwargs to the classifiers
                print()
                print ('Error')
                print ('One of the kwargs below:')
                print (kwargs.keys())
                print ('Does not belong to classifier', classifier)
                print()
                sys.exit(0)


        else:
            #This is already some sklearn classifier or an object that behaves like one
            self.clf=classifier

        print ('Created classifier of type:')
        print (self.clf)
        print()




    def classify(self, X_train, y_train, X_test):
        """
        Run unoptimised classifier with initial parameters.
        Parameters
        ----------
        X_train : array
            Array of training features of shape (n_train,n_features)
        y_train : array
            Array of known classes of shape (n_train)
        X_test : array
            Array of validation features of shape (n_test,n_features)

        Returns
        -------
        Yfit : array
            Predicted classes for X_test
        probs : array
        (If self.prob=True) Probability for each object to belong to each class.
        """

        self.clf.fit(X_train, y_train)
        if self.prob: #Probabilities requested
            probs=self.clf.predict_proba(X_test)
            Yfit=probs.argmax(axis=1) #Each object is assigned the class with highest probability
            classes=np.unique(y_train)
            classes.sort()
            Yfit=classes[Yfit]
            return Yfit, probs
        else:
            Yfit=self.clf.predict(X_test)
            #scr=self.clf.score(X_test, y_test)
            return Yfit

    def __custom_auc_score(self, estimator, X, Y):
        """
        Custom scoring method for use with GridSearchCV.

        Parameters
        ----------
        estimator : sklearn.Classifier
            The current classifier (used by GridSearchCV).
        X : array
            Array of training features of shape (n_train,n_features).
        Y : array
            Array of known classes of shape (n_train)

        Returns
        -------
        auc : float
            AUC score

        """
        probs=estimator.predict_proba(X)
        # Consider 120 as Type Ia, positive class
        # fpr, tpr, auc=roc(probs, Y, self.num_classes, true_class=120)
        print("NUM OF UNIQUE CLASSES (self):\n{}".format(self.num_classes))
        fpr, tpr, auc=roc(probs, Y, self.num_classes, true_class=1)
        return auc

    def optimised_classify(self, X_train, y_train, X_test, **kwargs):
        """
        Run optimised classifier using grid search with cross validation to choose optimal classifier parameters.
        Parameters
        ----------
        X_train : array
            Array of training features of shape (n_train,n_features)
        y_train : array
            Array of known classes of shape (n_train)
        X_test : array
            Array of validation features of shape (n_test,n_features)
        params : dict, optional
            Allows the user to specify which parameters and over what ranges to optimise. If not set,
            defaults will be used.
        true_class : int, optional
            The class determined to be the desired class (e.g. Ias, which might correspond to class 1). This allows
            the user to optimise for different classes (based on ROC curve AUC).

        Returns
        -------
        Yfit : array
            Predicted classes for X_test
        probs : array
        (If self.prob=True) Probability for each object to belong to each class.
        """


        if 'params' in kwargs:
            params=kwargs['params']
        else:
            params=self.params[self.classifier_name]
#            if self.classifier_name=='svm':
#                n_features=len(X_train[0, :]) #Update now the number of features is known
#                params['gamma']=[1/(n_features**2), 1/n_features, 1/np.sqrt(n_features)]

        if 'true_class' in kwargs:
            self.true_class=kwargs['true_class']
        else:
            self.true_class=1

        self.clf=grid_search.GridSearchCV(self.clf, params, scoring=self.__custom_auc_score, cv=5)

        self.clf.fit(X_train, y_train) #This actually does the grid search
        best_params=self.clf.best_params_
        print ('Optimised parameters:', best_params)

        for k in best_params.keys():
            #This is the safest way to check if something is a number
            try:
                float(best_params[k])
                if best_params[k]<=min(params[k]):
                    print()
                    print('WARNING: Lower boundary on parameter', k, 'may be too high. Optimum may not have been reached.')
                    print()
                elif best_params[k]>=max(params[k]):
                    print()
                    print('WARNING: Upper boundary on parameter', k, 'may be too low. Optimum may not have been reached.')
                    print()

            except (ValueError, TypeError):
                pass #Ignore a parameter that isn't numerical


        if self.prob: #Probabilities requested
            probs=self.clf.predict_proba(X_test)
            Yfit=probs.argmax(axis=1).tolist() #Each object is assigned the class with highest probability
            classes=np.unique(y_train)
            classes.sort()
            Yfit=classes[Yfit]
            return Yfit, probs
        else:
            Yfit=self.clf.predict(X_test)
            return Yfit

def __call_classifier(classifier, num_classes, X_train, y_train, X_test, param_dict,return_classifier):
    """Specifically designed to run with multiprocessing"""

    c=OptimisedClassifier(classifier, num_classes)
    if classifier in param_dict.keys():
        y_fit, probs=c.optimised_classify(X_train, y_train, X_test,params=param_dict[classifier])
    else:
        y_fit, probs=c.optimised_classify(X_train, y_train, X_test)

    if return_classifier:
        return probs,c.clf #Returns the best fitting sklearn classifier object
    else:
        return probs

def run_pipeline(features,types,output_name='',columns=[],classifiers=['nb','knn','svm','neural_network','boost_dt'],
                 training_set=0.7, param_dict={}, nprocesses=1, scale=True,
                 plot_roc_curve=True,return_classifier=False):
    """
    Utility function to classify a dataset with a number of classification methods. This does assume your test
    set has known values to compare against. Returns, if requested, the classifier objects to run on future test sets.

    Parameters
    ----------
    features : astropy.table.Table or array
        Features either in the form of a table or array
    types : astropy.table.Table or array
        Classes, either in the form of a table or array
    output_name : str, optional
        Full root path and name for output (e.g. '<output_path>/salt2-run-')
    columns : list, optional
        If you want to run a subset of columns
    classifiers : list, optional
        Which available ML classifiers to use
    training_set : float or list, optional
        If a float, this is the fraction of objects that will be used as training set. If a list, it's assumed these are
        the ID's of the objects to be used
    param_dict : dict, optional
        Use to run different ranges of hyperparameters for the classifiers when optimising
    nprocesses : int, optional
        Number of processors for multiprocessing (shared memory only). Each classifier will then be run in parallel.
    scale : bool, optional
        Rescale features using sklearn's preprocessing Scalar class (highly recommended this is True)
    plot_roc_curve : bool, optional
        Whether or not to plot the ROC curve at the end
    return_classifier : bool, optional
        Whether or not to return the actual classifier objects (due to the limitations of multiprocessing, this can't
        be done in parallel at the moment).

    Returns
    -------
    dict
        (If return_classifier=True) Dictionary of fitted sklearn Classifier objects

    """
    t1= time.time()

    # Do check whether a list of objects is provided for training or we are
    # considering a training ratio
    if isinstance(training_set, list):
        training_ratio = len(training_set)
        training_ratio = str(training_ratio)
    else:
        training_ratio = training_set*100
        training_ratio = str(training_ratio)

    if isinstance(features,Table):
        #The features are in astropy table format and must be converted to a numpy array before passing to sklearn

        #We need to make sure we match the correct Y values to X values. The safest way to do this is to make types an
        #astropy table as well.

        if not isinstance(types,Table):
            types=Table(data=[features['Object'],types],names=['Object','Type'])
        feats=join(features,types,'Object')

        if len(columns)==0:
            columns=feats.columns[1:-1]

        #Split into training and validation sets
        if np.isscalar(training_set):
            objs=feats['Object']
            objs=np.random.permutation(objs)
            training_set=objs[:(int)(training_set*len(objs))]

        #Otherwise a training set has already been provided as a list of Object names and we can continue
        feats_train=feats[np.in1d(feats['Object'],training_set)]
        feats_test=feats[~np.in1d(feats['Object'],training_set)]

        X_train=np.array([feats_train[c] for c in columns]).T
        y_train=np.array(feats_train['Type'])
        X_test=np.array([feats_test[c] for c in columns]).T
        y_test=np.array(feats_test['Type'])

    else:
        #Otherwise the features are already in the form of a numpy array
        if np.isscalar(training_set):
            inds=np.random.permutation(range(len(features)))
            train_inds=inds[:(int)(len(inds)*training_set)]
            test_inds=inds[(int)(len(inds)*training_set):]

        else:
            #We assume the training set has been provided as indices
            train_inds=training_set
            test_inds=range(len(types))[~np.in1d(range(len(types)),training_set)]

        X_train=features[train_inds]
        y_train=types[train_inds]
        X_test=features[test_inds]
        y_test=types[test_inds]


    #Rescale the data (highly recommended)
    if scale:
        scaler = StandardScaler()
        scaler.fit(np.vstack((X_train, X_test)))
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    probabilities={}
    classifier_objects={}

    if nprocesses>1 and return_classifier:
        print ("Due to limitations with python's multiprocessing module, classifier objects cannot be returned if " \
              "multiple processors are used. Continuing serially...")
        print()

    if nprocesses>1 and not return_classifier:
        partial_func=partial(__call_classifier,X_train=X_train, y_train=y_train, X_test=X_test,
                             param_dict=param_dict,return_classifier=False)
        p=Pool(nprocesses, maxtasksperchild=1)
        result=p.map(partial_func,classifiers)

        for i in range(len(result)):
            cls=classifiers[i]
            probabilities[cls]=result[i]

    else:
        for cls in classifiers:

            num_classes = len(unique(types, keys="Type"))
            print("NUM OF UNIQUE CLASSES:\n{}".format(num_classes))

            retval = __call_classifier(cls, num_classes, X_train, y_train, X_test, param_dict,return_classifier)

            if return_classifier:
                probabilities[cls] = retval[0]
                classifier_objects[cls] = retval[1]
            else:
                probabilities[cls] = retval[0]

    for i in range(len(classifiers)):
        cls=classifiers[i]
        probs=probabilities[cls]


        # Consider 120 as Type Ia, positive class
        # fpr, tpr, auc=roc(probs, y_test, num_classes, true_class=120)
        fpr, tpr, auc=roc(probs, y_test, num_classes, true_class=1)
        # Consider 120 as Type Ia, positive class
        # fom, thresh_fom=FoM(probs, y_test, num_classes, true_class=120, full_output=False)
        fom, thresh_fom=FoM(probs, y_test, num_classes, true_class=1, full_output=False)

        print ('Classifier', cls+':', 'AUC =', auc, 'FoM =', fom)

        if i==0:
            FPR=fpr
            TPR=tpr
            AUC=[auc]
        else:
            FPR=np.column_stack((FPR, fpr))
            TPR=np.column_stack((TPR, tpr))
            AUC.append(auc)

        if len(output_name)!=0:#Only save if an output directory is supplied
            typs=np.unique(y_train)
            typs.sort()
            typs=np.array(typs,dtype='str').tolist()

            if isinstance(features,Table):
                index_column=feats_test['Object']
            else:
                index_column=np.array(test_inds,dtype='str')
            dat=np.column_stack((index_column,probs))
            nms=['Object']+typs
            tab=Table(dat,dtype=['S64']+['f']*probs.shape[1],names=nms)
            flname=output_name+cls+training_ratio+'.probs'
            tab.write(flname,format='ascii')

            tab=Table(np.column_stack((fpr, tpr)),names=['FPR','TPR'])
            tab.write(output_name+cls+training_ratio+'.roc',format='ascii')

            np.savetxt(output_name+cls+training_ratio+'.auc',[auc])

    print()
    print ('Time taken ', (time.time()-t1)/60., 'minutes')

    if plot_roc_curve:
        plot_roc(FPR, TPR, AUC, labels=classifiers,label_size=16,tick_size=12,line_width=1.5)
        plt.show(block=False)

    if return_classifier:
        return classifier_objects
