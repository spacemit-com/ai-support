#!/bin/bash
#
# Author: hongjie.qin@spacemit.com
# Brief:  test demos
#
# <test_package>
# ├── bianbu-ai-support.riscv64
# │   ├── bin
# │   ├── include
# │   │   └── bianbuai
# │   └── lib
# │       ├── 3rdparty
# │       │   ├── onnxruntime -> spacemit-ort
# │       │   ├── opencv -> opencv.v4.6.0.qt5
# │       │   ├── opencv.v4.6.0.gtk2
# │       │   ├── opencv.v4.6.0.qt5
# │       │   └── spacemit-ort
# │       ├── libbianbuai.so -> libbianbuai.so.1
# │       ├── libbianbuai.so.1 -> libbianbuai.so.1.0.1
# │       └── libbianbuai.so.1.0.1
# ├── data
# │   ├── config
# │   │   └── nanodet.json
# │   ├── imgs
# │   ├── labels
# │   ├── models
# │   └── videos
# │       └── test.mp4
# └── test.sh

#set -e
#set -x

function show_usage() {
  echo "Usage:"
  echo "  ${BASH_SOURCE[0]} --x86_64  # native test with  x86_64 outputs"
  echo "  ${BASH_SOURCE[0]} --riscv64 # native test with riscv64 outputs"
  echo "  ${BASH_SOURCE[0]} --qemu <path_to_sdk>  # test riscv64 outputs on x86_64 with qemu"
}

WORKSPACE=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
BIANBUAI_HOME=$WORKSPACE/bianbu-ai-support.riscv64

if [[ $@ =~ "x86" ]]; then
  BIANBUAI_HOME=$WORKSPACE/bianbu-ai-support.x86_64
fi
if [[ $@ =~ "--qemu" ]]; then
  if [[ $@ =~ "x86" ]]; then
    echo "[ERROR] invalid options" && show_usage
    exit 0
  fi
  # set path for qemu and x86_64-riscv64-gcc
  SDK=$(getopt -l --qemu -- $@ | tr -d "'") # e.g. $HOME/workspace/bianbu-ai-release/workspace
  SDK=${SDK% --*}
  if [[ -z "$SDK" ]]; then
    echo "[ERROR] invalid options" && show_usage
    exit 0
  fi
  SYSROOT=$SDK/spacemit-gcc/sysroot
  QEMU_CMD="$SDK/spacemit-qemu/bin/qemu-riscv64 -L $SYSROOT"
fi

BIN_DIR=${BIANBUAI_HOME}/bin
LIB_BIANBUAI=${BIANBUAI_HOME}/lib
LIB_ONNXRUNTIME=${LIB_BIANBUAI}/3rdparty/onnxruntime/lib
LIB_OPENCV=${LIB_BIANBUAI}/3rdparty/opencv/lib:${LIB_BIANBUAI}/3rdparty/opencv/lib/3rdparty

task_prepare=(
  # TODO: add md5sum checking
  "if [[ ! -f data/models/squeezenet1.1-7.onnx ]]; then wget https://media.githubusercontent.com/media/onnx/models/main/archive/vision/classification/squeezenet/model/squeezenet1.1-7.onnx -O data/models/squeezenet1.1-7.onnx; fi"
  "if [[ ! -f data/models/nanodet-plus-m_320.onnx ]]; then wget https://bj.bcebos.com/paddlehub/fastdeploy/nanodet-plus-m_320.onnx -O data/models/nanodet-plus-m_320.onnx; fi"
)
task_classification=(
  "${BIN_DIR}/classification_demo data/models/squeezenet1.1-7.onnx data/labels/synset.txt data/imgs/dog.jpg"
)
task_detection=(
  "${BIN_DIR}/detection_demo data/models/nanodet-plus-m_320.onnx data/imgs/person0.jpg result0.jpg data/labels/coco.txt"
  "${BIN_DIR}/detection_demo data/models/nanodet-plus-m_320.int8.onnx data/imgs/person0.jpg result0.int8.jpg data/labels/coco.txt"
  "${BIN_DIR}/detection_video_demo data/models/nanodet-plus-m_320.int8.onnx data/labels/coco.txt data/videos/test.mp4 test.avi"
  "${BIN_DIR}/detection_stream_demo data/models/nanodet-plus-m_320.int8.onnx data/labels/coco.txt data/videos/test.mp4"
  "${BIN_DIR}/detection_stream_demo data/models/nanodet-plus-m_320.int8.onnx data/labels/coco.txt 0"
)
function smoke_test() {
  # preparation(e.g. download models)
  echo "[INFO] Prepare ..."
  for cmd in "${task_prepare[@]}"; do eval "$cmd"; done
  # image classification task test
  echo "[INFO] Smoke test with image classification task ..."
  for cmd in "${task_classification[@]}"; do
    echo "[INFO] Run: $cmd"
    env LD_LIBRARY_PATH=${LIB_BIANBUAI}:${LIB_ONNXRUNTIME}:${LIB_OPENCV}:$LD_LIBRARY_PATH ${QEMU_CMD} $cmd
  done
  # object detection task test
  echo "[INFO] Smoke test with object detection task ..."
  for cmd in "${task_detection[@]}"; do
    echo "[INFO] Run: $cmd"
    env LD_LIBRARY_PATH=${LIB_BIANBUAI}:${LIB_ONNXRUNTIME}:${LIB_OPENCV}:$LD_LIBRARY_PATH ${QEMU_CMD} $cmd
  done
}
smoke_test
