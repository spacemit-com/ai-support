#!/bin/bash
#
# Usage:
# $ bash pack.sh <local_install> [--skip-deb|--skip-ort|--skip-py|--debug]

set -e

[[ ${PY_CMD} ]] || PY_CMD=python3
REPO_DIR=$(readlink -f "$(dirname $0)/..")

# check dpkg/python3/...
function check() {
  set +e
  which $1 1>/dev/null 2>&1
  if [[ 0 -ne $? ]]; then
    echo -e "\033[5;31m[ERROR] Plz double check if '$1' is available or not!\033[0m"
    exit 1
  fi
  set -e
}

# make debian package
function make_deb_via_dpkg() {
  check dpkg

  PKG_NAME=bianbu-ai-support
  PKG_VER=$(cat ${REPO_DIR}/VERSION_NUMBER)
  PKG_DIR=${REPO_DIR}/${PKG_NAME}
  PKG_DESC="bianbu ai support library"

  INSTALL_LOCAL=$(readlink -f $1)
  # check existance
  if [[ ! -d $INSTALL_LOCAL ]]; then
    echo -e "\033[0;31m[ERROR] local installed directory'$INSTALL_LOCAL' doesn't exist!\033[0m"
    exit 2
  fi

  # get architecture
  ARCH=${INSTALL_LOCAL##*.}
  # update architecture for Debian && Ubuntu
  if [[ "$ARCH" == "x86_64" ]]; then
    ARCH="amd64"
  fi

  rm -rf ${PKG_DIR}
  # create debian control
  mkdir -p ${PKG_DIR}/DEBIAN
  cat <<EOF > ${PKG_DIR}/DEBIAN/control
Package: ${PKG_NAME}
Version: ${PKG_VER}
Section: Utils
Priority: optional
Architecture: ${ARCH}
Build-Depends: cmake, debhelper-compat (= 12), onnxruntime, libopencv-dev
Depends: onnxruntime, libopencv-dev
Maintainer: -
Description: ${PKG_DESC}.
EOF

  # copy bin, lib, include, etc.
  mkdir -p ${PKG_DIR}/usr/
  if [[ "$@" =~ "skip-ort" ]]; then
    :
  else
    cp -rd ${INSTALL_LOCAL}/lib/3rdparty/onnxruntime/* ${PKG_DIR}/usr/
  fi
  cp -rdf ${INSTALL_LOCAL}/* ${PKG_DIR}/usr/
  if [[ -d ${INSTALL_LOCAL}/../etc ]]; then
    cp -rdf ${INSTALL_LOCAL}/../etc ${PKG_DIR}/
  fi
  # post process
  rm -rdf ${PKG_DIR}/usr/lib/3rdparty ${PKG_DIR}/usr/demo

  # create debian package
  dpkg -b ${PKG_DIR} ${PKG_NAME}-${PKG_VER}.deb
}
[[ "$@" =~ "skip-deb" ]] || make_deb_via_dpkg $@

# make python package
function make_python_dist_pkg() {
  check ${PY_CMD}

  # prepare temp python source directory to avoid inplace constraction
  TMP_DIR=${REPO_DIR}/tmp
  PY_DIR=${REPO_DIR}/python
  rm -rf ${TMP_DIR} && cp -rdf ${PY_DIR} ${TMP_DIR}

  # make python dist package
  pushd ${TMP_DIR}
  ${PY_CMD} ${TMP_DIR}/setup.py sdist --dist-dir=${REPO_DIR}/dist
  ${PY_CMD} ${TMP_DIR}/setup.py bdist_wheel --dist-dir=${REPO_DIR}/dist
  popd
  [[ "$@" =~ "debug" ]] || rm -rf pushd ${TMP_DIR}
}
[[ "$@" =~ "skip-py" ]] || make_python_dist_pkg $@