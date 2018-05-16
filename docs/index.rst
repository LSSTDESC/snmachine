*********
snmachine
*********

.. image:: https://img.shields.io/badge/GitHub-LSSTDESC%2Fsnmachine-blue.svg?style=flat
    :target: https://github.com/LSSTDESC/snmachine

Welcome to version 1.0 of snmachine! As described in Lochner et al. `(2016) <https://arxiv.org/abs/1603.00882>`_,
this is a flexible python library for reading in photometric supernova light
curves, extracting useful features from them and subsequently performing
supervised machine learning to classify supernovae based on their light curves.
The library is also flexible enough to easily extend to general transient
classification.

.. image:: _static/pipeline.png

Usage Policy
============

This code is made available within the LSST DESC Collaboration. snmachine was
developed within the DESC, using DESC resources, and so meets the criteria given
in the DESC Publication Policy for being a “DESC product” (`DESC Publication Policy <http://lsstdesc.org/sites/default/files/LSST_DESC_Publication_Policy.pdf>`_).
We are aware that the codebase might be useful within other collaborations and
welcome requests for access to the code for non-DESC use. If you wish to use the
code outside DESC please submit your request here.

Citation
========

If you use snmachine in your work please cite (`BibTex <http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2016ApJS..225...31L&data_type=BIBTEX&db_key=AST&nocookieset=1>`_.)
Lochner, M., McEwen, J., Peiris, H., Lahav, O., Winter, M. (2016)
“Photometric Supernova Classification with Machine Learning”,
The Astrophysical Journal Supplement Series, 225, 31

.. The toc directive (below) is necessary to populate the side bar in the html
   documentation. If you don't want it to show up on the html page, use
   the :hidden: directive. For a complete reference on the table of contents
   see: http://www.sphinx-doc.org/en/stable/markup/toctree.html

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   quickstart
   pmodels
   snclassifier
   sndata
   snfeatures
   tsne_plot
   parallel

Contributing to snmachine
=========================

We welcome developers! Simply fork it into your own private repository and
submit a pull request when ready. You can contribute by adding new dataset-reading
methods, new feature extraction methods or new classification algorithms.
Please create an issue if you have any questions or problems with the code.
