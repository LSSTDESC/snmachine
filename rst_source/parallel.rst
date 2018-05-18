*********************************
Running on Clusters / in Parallel
*********************************

`snmachine` @ NERSC
===================

To run `snmachine` on NERSC first open an ssh connection to NERSC, set your shell to bash shell, and run the following from the command line:

``export PATH=/global/projecta/projectdirs/lsst/groups/SN/miniconda3/bin:$PATH``

Then to use `snmachine` any time all that is needed is to activate the environment using,::

    module load python
    source activate snmachine


Setting up `snmachine` on Jupyter @ NERSC
-----------------------------------------

Prior to accessing the Jupyter interface, either create a ``.condarc`` file (if you do not have already) with the following lines::

    envs_dirs:
    - /global/projecta/projectdirs/lsst/groups/SN/miniconda3/envs

or append this to your existing ``.condarc`` file. Be sure to access Jupyter via the `Jupyter-Dev hub <https://jupyter-dev.nersc.gov/>`_ , and not the Jupyter-Hub via 'MyNERSC' pages. Then, when prompted or accessing the notebook file, choose the ``Python [conda env:snmachine]``` kernel. This can be found under Kernel tab in the top menu bar. Now `snmachine` will be accessible through your Jupyter notebook.

Running jobs @ NERSC
--------------------
