from __future__ import absolute_import
import os
import tarfile

from .version import __VERSION__ as __version__
here = __file__
basedir = os.path.split(here)[0]
example_data = os.path.join(basedir, 'example_data')
spcc_data = os.path.join(example_data, 'SPCC_SUBSET')
tar = tarfile.open(spcc_data + '.tar.gz')
tar.extractall(example_data)
