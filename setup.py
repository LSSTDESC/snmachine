from setuptools import setup
from setuptools.command.install import install
import sys
import os
import re
import site
import tarfile


PACKAGENAME = 'snmachine'
__FALLBACK_VERSION__ = '2.0.0'


class ExtractExampleData(install):
    """Post-installation data extraction."""
    def run(self):
        install.run(self)

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
            spcc_data = os.path.join(PACKAGENAME, 'example_data',
                                     'SPCC_SUBSET')

            # Untar example data
            tar = tarfile.open(spcc_data + '.tar.gz')
            tar.extractall(example_data)


setup(
    name='snmachine',
    author='Michelle Lochner',
    author_email='dr.michelle.lochner@gmail.com',
    description='Machine learning code for photometric supernova '
                'classification',
    url='https://github.com/LSSTDESC/snmachine',
    license='BSD-3-Clause License',
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
        "fallback_version": __FALLBACK_VERSION__},
    setup_requires=['setuptools_scm>=3.2.0'],
    packages=['snmachine', 'utils'],
    include_package_data=True,
    package_data={'snmachine': ['example_data/SPCC_SUBSET.tar.gz',
                                'example_data/output_spcc_no_z/features/*.dat',
                                'example_data/example_data_for_tests.pckl']},
    exclude_package_data={'utils': ['archive/*']},
    cmdclass={'install': ExtractExampleData},
    install_requires=['astropy>=1.1.2',
                      'jupyter>=1.0.0',
                      'matplotlib>=1.5.1',
                      'numpy>=1.18.4',
                      'scikit-learn=0.20.0',
                      'scipy>=0.17.0',
                      'george>=0.3.0',
                      'iminuit==1.2',
                      'pandas>=0.23.0',
                      'extinction>=0.3.0',
                      'imbalanced-learn>=0.4.3',
                      'python=3.7',
                      'pip>=20.1',
                      'emcee>=2.1.0',
                      'numpydoc>=0.6.0',
                      'pytest-remotedata>=0.3.1',
                      'pywavelets>=0.4.0',
                      'sncosmo==2.1.0',
                      'nose>=1.3.7',
                      'future>=0.16',
                      'pyyaml>=3.13',
                      'pytest-xdist>=1.26.1',
                      'seaborn',
                      'schwimmbad',
                      'cesium',
                      'tqdm',
                      'lightgbm']
)
