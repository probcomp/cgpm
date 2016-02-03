#!/bin/sh

set -Ceu

: ${PYTHON:=python}
: ${PY_TEST:=`which py.test`}

if [ ! -x "${PY_TEST}" ]; then
    printf >&2 'unable to find pytest\n'
    exit 1
fi

root=`cd -- "$(dirname -- "$0")" && pwd`

(
    set -Ceux
    cd -- "${root}"
    rm -rf build
    "$PYTHON" setup.py build
    if [ $# -eq 0 ]; then
        # By default, when running all tests, skip tests that have
        # been marked for continuous integration by using __ci_ in
        # their names.  (git grep __ci_ to find these.)
        ./pythenv.sh "$PYTHON" "$PY_TEST" -k "not __ci_" tests
    elif [ "docker" = "$1" ]; then
        docker build -t gpmcc .  # Runs check.sh inside the docker.
    else
        # If args are specified, run all tests, including continuous
        # integration tests, for the selected components.
        ./pythenv.sh "$PYTHON" "$PY_TEST" "$@"
    fi
)
