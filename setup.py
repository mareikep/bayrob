import os

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy
import sys

def parse_requirements(filename):
    with open(filename) as f:
        lines = f.readlines()
    reqs = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            reqs.append(line)
    return reqs

version_file = os.path.join('src', 'bayrob', '.version')
if os.path.exists(version_file):
    __version__ = open(version_file).read().strip()
else:
    __version__ = 'v0.0.1'

install_requires = parse_requirements("requirements_.txt")

# Only define extension if numpy is available
try:
    extensions = [
        Extension(
            "bayrob.core.astar_jpt_ext",
            ["src/bayrob/core/astar_jpt_ext.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-march=native"] if sys.platform != "win32" else ["/O2"]
        )
    ]
    ext_modules = cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True
        }
    )
except ImportError:
    ext_modules = []

setup(
    name="bayrob",
    version=__version__,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    setup_requires=['numpy>=1.20.0', 'cython>=0.29.0'],
    install_requires=install_requires,
)