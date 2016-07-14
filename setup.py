# -*- coding: utf-8 -*-

# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# If some modules are not found, we use others, so no need to warn:
# pylint: disable=import-error
try:
    from setuptools import setup
    from setuptools.command.build_py import build_py
    from setuptools.command.sdist import sdist
    from setuptools.command.test import test
except ImportError:
    from distutils.core import setup
    from distutils.cmd import Command
    from distutils.command.build_py import build_py
    from distutils.command.sdist import sdist

    class test(Command):
        def __init__(self, *args, **kwargs):
            Command.__init__(self, *args, **kwargs)
        def initialize_options(self): pass
        def finalize_options(self): pass
        def run(self): self.run_tests()
        def run_tests(self): Command.run_tests(self)
        def set_undefined_options(self, opt, val):
            Command.set_undefined_options(self, opt, val)

import os.path

def readme_contents():
    readme_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'README.md')
    with open(readme_path) as readme_file:
        return unicode(readme_file.read(), 'UTF-8')

setup(
    name='cgpm',
    version='0.0.0',
    description='GPM Crosscat',
    long_description=readme_contents(),
    url='https://github.com/probcomp/cgpm',
    license='Apache-2.0',
    maintainer='Feras Saad',
    maintainer_email='fsaad@remove-this-component.mit.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    packages=[
        'cgpm',
        'cgpm.crosscat',
        'cgpm.dummy',
        'cgpm.exponentials',
        'cgpm.factor',
        'cgpm.mixtures',
        'cgpm.network',
        'cgpm.regressions',
        'cgpm.uncorrelated',
        'cgpm.utils',
        'cgpm.venturescript',
    ],
    package_dir={
        'cgpm': 'src',
    },
    install_requires=[
        'matplotlib>=1.5.0',
        'numpy',
        'pandas',
        'scipy>=0.14.0',  # For stats.multivariate_normal
    ],
    tests_require=[
        'pytest',
    ],
)
