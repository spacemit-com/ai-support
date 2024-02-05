#!/bin/bash
#
# Author: hongjie.qin@spacemit.com
# Brief:  Build demos and run smoke test.
#

set -e

# Note: update the following settings if necessary
_NAME=bianbu # cuspace
SDK=$(dirname $(which ${_NAME})) #$(dirname ${BASH_SOURCE[0]})

function config_native() {
  BIANBUAI_HOME=$SDK/bianbu-ai-support
  # Plz update the following settings !!!
  ORT_HOME=${PATH_TO_OFFICIAL_ONNXRUNTIME_RELEASE}
}

function config_x86_riscv64() {
  CROSS_TOOL=$SDK/spacemit-gcc/bin/riscv64-unknown-linux-gnu-
  SYSROOT=$SDK/spacemit-gcc/sysroot
  BIANBUAI_HOME=$SDK/bianbu-ai-support
  ORT_HOME=$SDK/spacemit-ort # cuspace-ai
  OPENCV_DIR=$SDK/bianbu-ai-support/lib/3rdparty/opencv4/lib/cmake/opencv4
  QEMU_CMD="$SDK/spacemit-qemu/bin/qemu-riscv64 -L $SYSROOT"
}

# config
if [[ $@ =~ "--native" ]]; then
  config_native
else
  config_x86_riscv64
fi

function build() {
  mkdir build && pushd build
  cmake .. -DBIANBUAI_HOME=${BIANBUAI_HOME} -DORT_HOME=${ORT_HOME} -DOpenCV_DIR=${OPENCV_DIR} -DCMAKE_C_COMPILER=${CROSS_TOOL}gcc -DCMAKE_CXX_COMPILER=${CROSS_TOOL}g++ -DCMAKE_SYSROOT=${SYSROOT}
  make -j4
  popd
  echo "[INFO] Building demos done."
}
if [[ ! -d build ]]; then
  build
else
  cd build && make && cd ..
fi

task_prepare=(
  "if [[ ! -d data && -d ../data ]]; then ln -sf ../data .; fi"
  "if [[ ! -d data ]]; then echo '[Error] Can not find data directory!'; exit 0; fi"
  "mkdir -p data/models"
  # TODO: add md5sum checking
  "if [[ ! -f data/models/squeezenet1.1-7.onnx ]]; then wget https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx -O data/models/squeezenet1.1-7.onnx; fi"
  "if [[ ! -f data/models/nanodet-plus-m_320.onnx ]]; then wget https://bj.bcebos.com/paddlehub/fastdeploy/nanodet-plus-m_320.onnx -O data/models/nanodet-plus-m_320.onnx; fi"
)
task_classification=(
  "build/classification_demo data/models/squeezenet1.1-7.onnx data/labels/synset.txt data/imgs/dog.jpg"
)
task_detection=(
  "build/detection_demo data/models/nanodet-plus-m_320.onnx data/labels/coco.txt data/imgs/person0.jpg result0.jpg"
)
function smoke_test() {
  # preparation(e.g. download models)
  echo "[INFO] Prepare ..."
  for cmd in "${task_prepare[@]}"; do eval "$cmd"; done
  # image classification task test
  echo "[INFO] Smoke test with image classification task ..."
  for cmd in "${task_classification[@]}"; do
    echo "[INFO] Run: $cmd"
    env LD_LIBRARY_PATH=${ORT_HOME}/lib:$LD_LIBRARY_PATH ${QEMU_CMD} $cmd
  done
  # object detection task test
  echo "[INFO] Smoke test with object detection task ..."
  for cmd in "${task_detection[@]}"; do
    echo "[INFO] Run: $cmd"
    env LD_LIBRARY_PATH=${ORT_HOME}/lib:$LD_LIBRARY_PATH ${QEMU_CMD} $cmd
  done
}

if [[ "$@" =~ "--test" ]]; then
  smoke_test
else
  echo "[INFO] Try '${BASH_SOURCE[0]} $@ --test' to run the demos."
fi
