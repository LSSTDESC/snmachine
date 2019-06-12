from setuptools import setup
from setuptools.command.install import install
import sys
import os
import re
import site
import tarfile


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
__version__ = versionRegExp.findall(s)[0]
print(__version__)


class ExtractExampleData(install):
    """Post-installation data extraction."""
    def run(self):
        install.run(self)
        os.system("cat testing.egg-info/PKG-INFO")

        """Extract example data in the installation directory"""
        if '--user' in sys.argv:
            paths = (site.getusersitepackages(),)
            print("Package installed outside of conda enviroment. Dependencies \
            may be missing")
        else:
            paths = (site.getsitepackages(),)
            install_directory = paths[0][0]
            install_directory = os.path.join(install_directory, PACKAGENAME)

            # Find example_data
            example_data = os.path.join(install_directory, 'example_data')
            spcc_data = os.path.join(example_data, 'SPCC_SUBSET')

            # Untar example data
            tar = tarfile.open(spcc_data + '.tar.gz')
            tar.extractall(example_data)


setup(
    name='snmachine',
    version=__version__,
    packages=['snmachine', 'gapp', 'gapp.covfunctions', 'utils'],
    packagedir={PACKAGENAME: 'snmachine'},
    include_package_data=True,
    package_data={'snmachine': ['example_data/SPCC_SUBSET.tar.gz', 'example_data/output_spcc_no_z/features/*.dat', 'example_data/example_data_for_tests.pckl']},
    exclude_package_data={'utils': ['archive/*']},
    cmdclass={'install': ExtractExampleData},
    url='',
    license='MIT',
    author='Michelle Lochner',
    author_email='dr.michelle.lochner@gmail.com',
    description='Machine learning code for photometric supernova classification'
)
