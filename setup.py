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
    url='https://github.com/LSSTDESC/snmachine',
    license='BSD-3-Clause License',
    author='Michelle Lochner',
    author_email='dr.michelle.lochner@gmail.com',
    description='Machine learning code for photometric supernova '
                'classification'
)
