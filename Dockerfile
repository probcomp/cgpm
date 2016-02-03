# Dockerfile that builds, installs, and tests gpmcc. It is for development
# only; users should use the python package.

FROM ubuntu
RUN apt-get update -qq

# Non-python dependencies for gpmcc.
RUN apt-get install -y -qq python-dev python-pip

# Transitive non-python dependencies for matplotlib.
RUN apt-get install -y -qq libjpeg-dev libxft-dev

# SciPy dependency tracking is confusing and slow. Install at the OS level
# instead of with pip to avoid thinking.
# See also: https://github.com/scikit-learn/scikit-learn/issues/4164
RUN apt-get install -y -qq python-numpy python-scipy

# Build and install gpmcc.
COPY setup.py pythenv.sh check.sh README.md HACKING /gpmcc/
COPY src /gpmcc/src
# Notably, do not copy build, .eggs, dist, sdist, etc.
WORKDIR /gpmcc
RUN python setup.py bdist_wheel
RUN pip install dist/gpmcc-*-py2-none-any.whl

RUN pip install pytest
# In case setup.py ever develops test_require that are not already satisfied:
RUN python setup.py test
# Run tests:
COPY tests /gpmcc/tests
RUN find tests -name __pycache__ -o -name "*.pyc" --exec rm -fr {} \;
RUN ./check.sh tests
