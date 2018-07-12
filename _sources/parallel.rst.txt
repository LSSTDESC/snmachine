*********************************
Running on Clusters / in Parallel
*********************************

`snmachine` on NERSC
====================

To run `snmachine` on NERSC first open an ssh connection to NERSC, set your shell to bash shell, and run the following from the command line:

``export PATH=/global/projecta/projectdirs/lsst/groups/SN/miniconda3/bin:$PATH``

Then to use `snmachine` any time all that is needed is to activate the environment using, ::

    module load python
    source activate snmachine


Setting up `snmachine` on Jupyter @ NERSC
-----------------------------------------

Prior to accessing the Jupyter interface, either create a ``.condarc`` file (if you do not have already) with the following lines::

    envs_dirs:
    - /global/projecta/projectdirs/lsst/groups/SN/miniconda3/envs

or append this to your existing ``.condarc`` file. Be sure to access Jupyter via the `Jupyter-Dev hub <https://jupyter-dev.nersc.gov/>`_ , and not the Jupyter-Hub via 'MyNERSC' pages. Then, when prompted or accessing the notebook file, choose the ``Python [conda env:snmachine]`` kernel. This can be found under Kernel tab in the top menu bar. Now `snmachine` will be accessible through your Jupyter notebook.

Running jobs @ NERSC
--------------------

This will be implmented in the future.



`snmachine` on Hypatia
======================

First, login to Hypatia with your credentials via the ssh method. ::

    ssh USERNAME@hypatia-login.hpc.phys.ucl.ac.uk

Next you will need to import python into your node using ::

    module load python

Then, if you have not already setup `snmachine`, you will need to clone the `snmachine` repository, ::

    git clone https://github.com/LSSTDESC/snmachine.git

and install the package as usual, following steps 2 through 4 in the conda installation guide, :ref:`conda-install`. You should now be setup to run `snmachine` as normal on Hypatia.


Run pipeline instructions (temporary)
-------------------------------------

These instructions are a temporary (as of July 2nd 2018) guide to run the jobs pipeline on Hypatia at UCL.

1. With a working installation of `snmachine` you will first need to switch to the ``issue/38/refactor-utils-cjonly`` branch and pull the latest commits. ::

    git pull
    git checkout issue/38/refactor-utils-cjonly

2. Navigate to the ``snmachine/utils`` folder and execute, ::

    python create_jobs.py

* NOTE: If this fails it is possible that the ``create_jobs.py`` script did not find the data files specified by default. In order to try again, first, remove the ``snmachine/utils/jobs`` folder. Then, you can then pass your own list of data through the ``-op`` argument flag. In doing so you should specify the location to a ``*.LIST`` file that contains a list of all the objects that you want to run. An example of this is, ::

    python create_jobs.py -op /share/hypatia/snmachine_resources/data/DES_spcc/SIMGEN_PUBLIC_DES/SIMGEN_PUBLIC_DES.LIST

3. Next navigate to the newly created ``snmachine/utils/jobs`` folder and execute::

    sh run_all.sh

4. Your jobs will now be in the queue to execute on the nodes specified in the ``create_jobs.py`` script. You can monitor their progress through the ``showq`` and ``qstat`` shell commands.
