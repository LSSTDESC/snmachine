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
                      'matplotlib>=1.5.1',
                      'numpy>=1.18.4',
                      'scikit-learn',
                      'scipy>=0.17.0',
                      'george>=0.3.0',
                      'iminuit',
                      'pandas>=0.23.0',
                      'pywavelets>=0.4.0',
                      'sncosmo==2.1.0',
                      'nose>=1.3.7',
                      'future>=0.16',
                      'pyyaml>=3.13',
                      'seaborn',
                      'lightgbm',
                      'setuptools_scm']
)
