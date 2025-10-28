
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy
import sys

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
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    setup_requires=['numpy>=1.20.0', 'cython>=0.29.0'],
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
)