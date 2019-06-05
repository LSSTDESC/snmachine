from __future__ import absolute_import
import os

from .version import version as __version__

here = __file__
basedir = os.path.split(here)[0]
example_data = os.path.join(basedir, 'example_data')
spcc_data = os.path.join(example_data, 'SPCC_SUBSET')
