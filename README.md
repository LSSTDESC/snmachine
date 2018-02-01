# snmachine

Welcome to version 1.0 of snmachine! This is a flexible python library for reading in supernova data or simulations, extracting useful features from them and subsequently performing supervised machine learning to classify supernovae based on their light curves. 

## Usage Policy

This code is made available within the LSST DESC Collaboration. `snmachine` was developed within the DESC, using DESC resources, and so meets the criteria given in the DESC Publication Policy for being a “DESC product” ([DESC Publication Policy](http://lsstdesc.org/sites/default/files/LSST_DESC_Publication_Policy.pdf)). We are aware that the codebase might be useful within other collaborations and welcome requests for access to the code for non-DESC use. If you wish to use the code outside DESC please submit your request [here](https://docs.google.com/forms/d/e/1FAIpQLSfHKNf-GeIGeRWODtwpVz_byXsUDBYISjlQk5lv1W9M0hgB3g/viewform?usp=sf_link).  

## Contributors

The following people have contributed to snmachine v1.0:
Michelle Lochner, Robert Schuhmann, Jason McEwen, Hiranya Peiris, Rahul Biswas, Ofer Lahav, Johnny Holland, Max Winter

## Contributing to snmachine

We welcome developers! Simply fork it into your own private repository and submit a pull request when ready. You can contribute by adding new dataset-reading methods, new feature extraction methods or new classification algorithms. Please create an issue if you have any questions or problems with the code.

## Citation

If you use snmachine in your work please cite ([BibTex])(http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2016ApJS..225...31L&data_type=BIBTEX&db_key=AST&nocookieset=1):
Lochner, M., McEwen, J., Peiris, H., Lahav, O., Winter, M. (2016) “Photometric Supernova Classification with Machine Learning”, The Astrophysical Journal Supplement Series, 225, 31

## Installation

snmachine is now compatible with Python2 and Python3.

There are two possible ways to set up snmachine:

### Using conda

By far the easiest way to install snmachine is to use anaconda.

1) Install anaconda if you don't already have it (https://www.continuum.io/downloads)

2) Create a new anaconda environment by type (inside the snmachine folder):

`conda env create --name snmachine --file environment.yml`

3) Activate the environment by typing:

`source activate snmachine`

**Note: If you have tsch instead of bash this will not work!**

A simple workaround is to manually edit your PATH environment variable to point to the new anaconda environment:

`setenv PATH <your path to anaconda>/envs/snmachine/bin/:$PATH`

4) Install snmachine by typing:

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
  - george>=0.3.0

### Installation caveats

1) `snmachine` has the ability to use nested sampling for some of the parameter estimates, which we have found to give much more reliable fits than least squares. However this depends on `pymultinest` (https://github.com/JohannesBuchner/PyMultiNest) being installed separately, since it requires manual compiling of Fortran code (but not it's not a particularly difficult one).

2) If you want to use neural networks, these are currently only available in the development version of scikit-learn (version 0.18). You will also need to install that separately to enable neural networks (see http://scikit-learn.org/stable/developers/contributing.html#git-repo).


## Documentation

snmachine has complete docstrings for all functions and classes. These can be compiled into html documentation by typing:

`make html`

in the `docs` folder.

## Examples

The folder `examples` contains an example jupyter notebook. Start the notebook from the `examples` directory by typing:

`jupyter notebook example_spcc.ipynb`

Execute each cell block using "shift-enter". A subset of simulated DES data from the supernova photometric classification challenge is provided to illustrate the code.

## Unit Tests

snmachine comes with a suite of unit tests, which allow you to check whether the software has been set up correctly and is working properly. Please navigate to the test folder to run tests.
