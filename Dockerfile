# Dockerfile that builds, installs, and tests cgpm. It is for development
# only; users should use the python package.

FROM ubuntu
RUN apt-get update -qq

# Non-python dependencies for cgpm.
RUN apt-get install -y -qq python-dev python-pip

# Transitive non-python dependencies for matplotlib.
RUN apt-get install -y -qq libjpeg-dev libxft-dev

# Transitive non-python dependencies for scipy: Will need to use pip to upgrade.
RUN apt-get install -y -qq liblapack-dev gfortran

# SciPy dependency tracking is confusing and slow. Install at the OS level
# instead of with pip to avoid thinking.
# See also: https://github.com/scikit-learn/scikit-learn/issues/4164
RUN apt-get install -y -qq python-numpy python-scipy python-pandas
RUN pip install scikit-learn
RUN pip install statsmodels

### Build and install cgpm.

COPY setup.py pythenv.sh check.sh README.md HACKING /cgpm/
COPY src /cgpm/src
# Notably, do not copy build, .eggs, dist, sdist, etc.
WORKDIR /cgpm
RUN python setup.py bdist_wheel
RUN pip install dist/cgpm-*-py2-none-any.whl


### Test cgpm.

RUN pip install pytest
# In case setup.py ever develops test_require that are not already satisfied:
RUN python setup.py test
# Run tests:
COPY tests /cgpm/tests
RUN find tests -name __pycache__ -o -name "*.pyc" -exec rm -fr {} \;
RUN ./check.sh tests
