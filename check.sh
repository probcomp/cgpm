#!/bin/sh -ex

# This script runs the automated tests for gpmcc, such as they are. It is
# intended to be run on a CI system and before pushes.

docker build -t gpmcc .
: PASS
