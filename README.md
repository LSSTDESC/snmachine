# snmachine

[![DOI](https://zenodo.org/badge/63328700.svg)](https://zenodo.org/badge/latestdoi/63328700)
<a href="https://ascl.net/2107.006"><img src="https://img.shields.io/badge/ascl-2107.006-blue.svg?colorB=262255" alt="ascl:2107.006" /></a>

| `main`  | `dev` |
| ------------- | ------------- |
| [![Build Status](https://dev.azure.com/zcicg57/snmachine/_apis/build/status/LSSTDESC.snmachine?branchName=master)](https://dev.azure.com/zcicg57/snmachine/_build/latest?definitionId=3&branchName=main) | [![Build Status](https://dev.azure.com/zcicg57/snmachine/_apis/build/status/LSSTDESC.snmachine?branchName=dev)](https://dev.azure.com/zcicg57/snmachine/_build/latest?definitionId=3&branchName=dev) |

Welcome to version 2.0.0 of `snmachine`! As described in ([Lochner et al. (2016)](https://arxiv.org/abs/1603.00882)), this is a flexible python library for reading in photometric supernova light curves, extracting useful features from them and subsequently performing supervised machine learning to classify supernovae based on their light curves. The library is also flexible enough to easily extend to general transient classification.

Up-to-date documentation of `snmachine` can be found via the following Github Pages link.

[Online Documentation](https://lsstdesc.github.io/snmachine/)

## Usage Policy

`snmachine` was developed within the DESC, using DESC resources, and so meets the criteria given in the DESC Publication Policy for being a “DESC product” ([DESC Publication Policy](http://lsstdesc.org/sites/default/files/LSST_DESC_Publication_Policy.pdf)). This software is released with a BSD 3-Clause License.

## Release

The list of released versions of this package can be found [here](https://github.com/LSSTDESC/snmachine/releases), with the `dev` branch including the most recent (non-released) development.

## Contributors

The following people have contributed to snmachine v1.0:
Michelle Lochner, Robert Schuhmann, Jason McEwen, Hiranya Peiris, Rahul Biswas, Ofer Lahav, Johnny Holland, Max Winter

The following people have contributed to snmachine v2.0.0:
Catarina Alves, Hiranya Peiris, Michelle Lochner, Jason McEwen, Tarek Allam Jr, Rahul Biswas, Christian Setzer, Robert Schuhmann

## Contributing to snmachine

We welcome developers! Simply fork it into your own private repository and submit a pull request when ready. You can contribute by adding new dataset-reading methods, new feature extraction methods or new classification algorithms. Please create an issue if you have any questions or problems with the code.

See [this page](https://github.com/LSSTDESC/snmachine/blob/dev/CONTRIBUTING.md) for a useful guide to contributing to `snmachine`.

## Citation

[![DOI](https://zenodo.org/badge/63328700.svg)](https://zenodo.org/badge/latestdoi/63328700)

If you use snmachine in your work please cite:

- [BibTex](http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2016ApJS..225...31L&data_type=BIBTEX&db_key=AST&nocookieset=1):
Lochner, M., McEwen, J., Peiris, H., Lahav, O., Winter, M. (2016) “Photometric Supernova Classification with Machine Learning”, The Astrophysical Journal Supplement Series, 225, 31

- [BibTex](https://ui.adsabs.harvard.edu/abs/2021arXiv210707531A/abstract): Alves, C. S., Peiris, H. V., Lochner, M., McEwen, J. D., Allam Jr, T., & Biswas, R. (2021) "Considerations for optimizing photometric classification of supernovae from the Rubin Observatory", arXiv preprint arXiv:2107.07531.

## Installation

`snmachine` is compatible with Python3.

The installation is detailed in the online documentation. ([Install Guide](https://lsstdesc.github.io/snmachine/install.html)):

Alternativelly, use `pip` to install `snmachine`.