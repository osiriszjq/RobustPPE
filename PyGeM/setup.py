"""
PyGeM setup.py
"""
import os
import sys
from setuptools import setup, find_packages, Command
import pygem

# Package meta-data.
NAME = pygem.__title__
DESCRIPTION = 'Python Geometrical Morphing.'
URL = 'https://github.com/mathLab/PyGeM'
MAIL = pygem.__mail__
AUTHOR = pygem.__author__
VERSION = pygem.__version__
KEYWORDS = 'dimension_reduction mathematics ffd morphing iges stl vtk openfoam'

REQUIRED = [
    'future', 'numpy', 'scipy',	'matplotlib',
]

EXTRAS = {
    'docs': ['Sphinx==1.4', 'sphinx_rtd_theme'],
}

LDESCRIPTION = (
    "PyGeM is a python package using Free Form Deformation, Radial Basis "
    "Functions and Inverse Distance Weighting to parametrize and morph complex "
    "geometries. It is ideally suited for actual industrial problems, since it "
    "allows to handle:\n"
    "1) Computer Aided Design files (in .iges, .step, and .stl formats) Mesh "
    "files (in .unv and OpenFOAM formats)\n"
    "2) Output files (in .vtk format)\n"
    "3) LS-Dyna Keyword files (.k format).\n"
    "\n"
    "By now, it has been used with meshes with up to 14 milions of cells. Try "
    "with more and more complicated input files! See the Examples section "
    "below and the Tutorials to have an idea of the potential of this package."
)

here = os.path.abspath(os.path.dirname(__file__))
class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        """ void """
        pass

    def finalize_options(self):
        """ void """
        pass

    def run(self):
        try:
            self.status('Removing previous builds...')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution...')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine...')
        os.system('twine upload dist/*')

        self.status('Pushing git tags...')
        os.system('git tag v{0}'.format(VERSION))
        os.system('git push --tags')

        sys.exit()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LDESCRIPTION,
    author=AUTHOR,
    author_email=MAIL,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    keywords=KEYWORDS,
    url=URL,
    license='MIT',
    packages=find_packages(),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    test_suite='nose.collector',
    tests_require=['nose'],
    include_package_data=True,
    zip_safe=False,

    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },)
