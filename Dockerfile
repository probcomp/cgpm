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
COPY HACKING README.md setup.py /src/
COPY gpmcc /src/gpmcc
WORKDIR /src
RUN python setup.py bdist_wheel
RUN pip install dist/gpmcc-*-py2-none-any.whl

# Run a simple smoke test. Delete original build directory to make sure no
# dependencies leak.
COPY gpmcc/tests /tests
WORKDIR /tests
RUN rm -rf /src
RUN python /tests/binomial.py
