"""
This file is has been inspired by sunpy/sunpy/version.py
It will indentify which version is install by the user
"""
from pkg_resources import get_distribution, DistributionNotFound
try:
    version = get_distribution("snmachine").version
except DistributionNotFound:
    version = 'unknown.dev'
