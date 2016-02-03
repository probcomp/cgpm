# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2016 MIT Probabilistic Computing Project

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

import os.path
from setuptools import setup

def readme_contents():
    readme_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'README.md')
    with open(readme_path) as readme_file:
        return unicode(readme_file.read(), 'UTF-8')

setup(
    name='gpmcc',
    version='0.0.0',
    description='GPM Crosscat',
    long_description=readme_contents(),
    url='https://github.com/probcomp/gpmcc',
    license='MIT',
    maintainer='Feras Saad',
    maintainer_email='fsaad@remove-this-component.mit.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    packages=[
        'gpmcc',
        'gpmcc.dists',
        'gpmcc.utils'
    ],
    package_dir={
        'gpmcc': 'src',
    },
    install_requires=[
        'matplotlib>=1.5.0',
        'numpy',
        'scipy',
    ],
    tests_require=[
        'pytest',
    ],
)
