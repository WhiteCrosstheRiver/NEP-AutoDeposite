"""Build fast_xyz pybind11 extension: python setup.py build_ext --inplace"""

import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

extra_compile = ["-O3", "-fopenmp"]
extra_link = ["-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "fast_xyz",
        ["fast_xyz/fast_xyz.cpp"],
        include_dirs=[os.path.join("fast_xyz")],
        cxx_std=17,
        extra_compile_args=extra_compile,
        extra_link_args=extra_link,
    ),
]

setup(
    name="fast_xyz",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
