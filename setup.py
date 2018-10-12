from setuptools import setup
import sys
import os
import re


PACKAGENAME = 'snmachine'
packageDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          PACKAGENAME)

# Obtain the package version
versionFile = os.path.join(packageDir, 'version.py')
with open(versionFile, 'r') as f:
          s = f.read()
# Look up the string value assigned to __version__ in version.py using regexp
versionRegExp = re.compile("__VERSION__ = \"(.*?)\"")
# Assign to __version__
__version__ =  versionRegExp.findall(s)[0]
print(__version__)

setup(
    name='snmachine',
    version=__version__,
    # packages=['snmachine', 'gapp', 'gapp.covfunctions'],
    packages=['snmachine'],
    packagedir={PACKAGENAME:'snmachine'},
    include_package_data=True,
    package_data={'snmachine': ['example_data/SPCC_SUBSET.tar.gz', 'example_data/output_spcc_no_z/features/*.dat']},
    #data_files=[('snmachine', ['example_data/SPCC_SUBSET.tar.gz'])],
    url='',
    license='MIT',
    author='Michelle Lochner',
    author_email='dr.michelle.lochner@gmail.com',
    description='Machine learning code for photometric supernova classification'
)
