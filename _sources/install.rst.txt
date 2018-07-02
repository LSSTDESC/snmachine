**********************
Installation and Setup
**********************

snmachine is now compatible with Python2 and Python3.
There are two possible ways to set up snmachine:

.. _conda-install:

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

.. literalinclude:: ../environment.yml
    :lines: 6-

Installation Caveats
====================

snmachine has the ability to use nested sampling for some of the parameter
estimates, which we have found to give much more reliable fits than least
squares. However this depends on `pymultinest <https://github.com/JohannesBuchner/PyMultiNest>`_ being installed separately,
since it requires manual compiling of Fortran code (but not it's not a
particularly difficult one).
