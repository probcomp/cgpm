# Only need to change these two variables
PKG_NAME=cgpm

OS=$TRAVIS_OS_NAME-64
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $CONDA_USER $CONDA_BLD_PATH/$OS/$PKG_NAME-*.tar.bz2 --force
