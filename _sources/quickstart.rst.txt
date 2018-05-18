***********
Quick Start
***********

First, if you are using `snmachine` on your own machine, please see the installation steps outlined in :doc:`install`.


Examples
========

To get a feel for running `snmachine` and to understand it's uses, the best place to start are the jupyter notebooks. These are contained in the folder `examples`, which you can also view on Github, `Getting Started Notebook <https://github.com/LSSTDESC/snmachine/blob/master/examples/example_spcc.ipynb>`_. You can also run this yourself. Start the notebook from the `examples` directory by typing,::

    jupyter notebook example_spcc.ipynb

Execute each cell block using "shift-enter". A subset of simulated DES data from the supernova photometric classification challenge is provided to illustrate the code.

Unit tests
==========

Also, great way to check your installation of snmachine is to run the suite of unit tests. These allow you to check whether the software has been set up correctly and that each step is working properly. Please navigate to the `test` folder to run tests using the syntax,::

    py.test *.py

If you would like to run a subset of tests you can run the command,::

    py.test *.py -m 'slow'

to run the tests that are the most cpu-intensive. Or conversely,::

    py.test *.py -m 'not slow'

to run the tests of the basic functionality of `snmachine`.
