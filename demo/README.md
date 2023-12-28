# User Guide

## Quick Build

```bash
# Note: update the following settings if necessary
SDK=${PATH_TO_SPACEMIT_AI_SDK} # e.g. /opt/spacemit-ai-sdk.v0.5.0

# For cross compilation, try:
CROSS_TOOL=$SDK/spacemit-gcc/bin/riscv64-unknown-linux-gnu-
SYSROOT=$SDK/spacemit-gcc/sysroot
BIANBUAI_HOME=$SDK/bianbu-ai-support
ORT_HOME=$SDK/spacemit-ort
OPENCV_DIR=$SDK/bianbu-ai-support/lib/3rdparty/opencv4/lib/cmake/opencv4

# For native building, one may need to install opencv first, then try:
#CROSS_TOOL=
#SYSROOT=
#BIANBUAI_HOME=$SDK/bianbu-ai-support
#ORT_HOME=${PATH_TO_OFFICIAL_ONNXRUNTIME_RELEASE}
#OPENCV_DIR=

mkdir build && pushd build
cmake .. -DBIANBUAI_HOME=${BIANBUAI_HOME} -DORT_HOME=${ORT_HOME} -DOpenCV_DIR=${OPENCV_DIR} -DCMAKE_C_COMPILER=${CROSS_TOOL}gcc -DCMAKE_CXX_COMPILER=${CROSS_TOOL}g++ -DCMAKE_SYSROOT=${SYSROOT}
make -j4
popd
```

## Quick Test

```bash
# Add qemu settings if necessary(e.g. run riscv64 demo on x86_64).
QEMU_CMD="$SDK/qemu/bin/qemu-riscv64 -L $SYSROOT"
# For native test, just let:
#QEMU_CMD=

# Smoke test with image classification
env LD_LIBRARY_PATH=${ORT_HOME}/lib:$LD_LIBRARY_PATH ${QEMU_CMD} \
  build/classification_demo data/models/squeezenet1.1-7.onnx data/labels/synset.txt data/imgs/dog.jpg

# Smoke test with object detection
env LD_LIBRARY_PATH=${ORT_HOME}/lib:$LD_LIBRARY_PATH ${QEMU_CMD} \
  build/detection_demo data/models/nanodet-plus-m_320.onnx data/imgs/person0.jpg result0.jpg data/labels/coco.txt
```

* Model List

4f22f9a64ab9612ca4372a0343b3879a  [data/models/nanodet-plus-m_320.onnx](https://bj.bcebos.com/paddlehub/fastdeploy/nanodet-plus-m_320.onnx)  
497ad0774f4e0b59e4f2c77ae88fcdfc  [data/models/squeezenet1.1-7.onnx](https://github.com/onnx/models/blob/main/archive/vision/classification/squeezenet/model/squeezenet1.1-7.onnx)  
