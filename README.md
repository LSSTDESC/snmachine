# snmachine

Welcome to the pre-release of snmachine! This is a flexible python library for reading in supernova data or simulations, extracting useful features from them and subsequently performing supervised machine learning to classify supernovae based on their light curves. Please keep this code private for now within the DESC collaboration.

## Installation

### Using conda

By far the easiest way to install snmachine is to use anaconda.

1) Install anaconda if you don't already have it (https://www.continuum.io/downloads)

2) Create a new anaconda environment by type (inside the snmachine folder):

`conda env create --name snmachine --file environment.yml`

3) Install snmachine by typing:

`python setup.py install`

or if you're developing

`python setup.py develop`

### Manual dependency installation

If you don't want to use an conda environment, snmachine requires the following packages (all of which are on either conda or pip):

dependencies:
  - astropy>=1.1.2
  - cython>=0.23.4
  - jupyter>=1.0.0
  - matplotlib>=1.5.1
  - numpy>=1.10.4
  - scikit-learn>=0.17.1
  - scipy>=0.17.0
  - emcee>=2.1.0 [pip]
  - iminuit>=1.2 [pip]
  - sncosmo>=1.2.0 [pip]
  - pywavelets>=0.4.0 [pip]

### Installation caveats

1) `snmachine` has the ability to use nested sampling for some of the parameter estimates, which we have found to give much more reliable fits than least squares. However this depends on `pymultinest` (https://github.com/JohannesBuchner/PyMultiNest) being installed separately, since it requires manual compiling of Fortran code (but not it's not a particularly difficult one).

2) If you want to use neural networks, these are currently only available in the development version of scikit-learn (version 0.18). You will also need to install that separately to enable neural networks (see http://scikit-learn.org/stable/developers/contributing.html#git-repo).

## Contributing

Please feel free to contribute to the code! Simply fork it into your own private repository and submit a pull request when ready. You can contribute by adding new dataset-reading methods, new feature extraction methods or new classification algorithms. 

## Examples

The folder `examples` contains an example jupyter notebook. Start the notebook from the `examples` directory by typing:

`jupyter notebook example_spcc.ipynb`

Execute each cell block using "shift-enter". A subset of simulated DES data from the supernova photometric classification challenge is provided to illustrate the code.
