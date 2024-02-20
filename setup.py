## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['bb_filters'],
    package_dir={'bb_filters': 'src/bb_filters',
                 'sauvc_objects': 'src/bb_filters/sauvc_objects'})

setup(**setup_args)
