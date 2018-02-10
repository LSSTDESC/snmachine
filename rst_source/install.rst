**********************
Installation and Setup
**********************

snmachine is now compatible with Python2 and Python3.
There are two possible ways to set up snmachine:

Using Conda
===========

By far the easiest way to install snmachine is to use anaconda.

1. Install anaconda if you don't already have it (https://www.continuum.io/downloads)

2. Create a new anaconda environment by type (inside the snmachine folder)::

    conda env create --name snmachine --file environment.yml

3. Activate the environment by typing::

    source activate snmachine

  **Note: If you have tsch instead of bash this will not work!** A simple
  workaround is to manually edit your PATH environment variable to point to the
  new anaconda environment::

    setenv PATH <your path to anaconda>/envs/snmachine/bin/:$PATH

4. Install snmachine by typing::

    python setup.py install

   or if you're developing::

    python setup.py develop

Manual Dependency Installation
==============================

If you don't want to use a conda environment, snmachine requires the following
packages (all of which are on either conda or pip):

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

Installation Caveats
====================

snmachine has the ability to use nested sampling for some of the parameter
estimates, which we have found to give much more reliable fits than least
squares. However this depends on pymultinest
(https://github.com/JohannesBuchner/PyMultiNest) being installed separately,
since it requires manual compiling of Fortran code (but not it's not a
particularly difficult one).

If you want to use neural networks, these are currently only available in the
development version of scikit-learn (version 0.18). You will also need to
install that separately to enable neural networks
(see http://scikit-learn.org/stable/developers/contributing.html#git-repo).