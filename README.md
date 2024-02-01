# support


## Getting started

### install dependence
gcc version: 14

prepare opencv (at least version 4.2.0)
```bash
sudo apt-get install libopencv-dev
```
prepare onnxruntime

visit [onnxruntime release](https://github.com/microsoft/onnxruntime/releases/tag/v1.15.1) and download proper version.


### build demo with bash followed

```bash
ORT_HOME=${PATH_TO_ONNXRUNTIME}
# Note: Add the installation prefix of "OpenCV" to CMAKE_PREFIX_PATH or set
# "OpenCV_DIR" to a directory containing one of the following names:
#   OpenCVConfig.cmake
#   opencv-config.cmake
OPENCV_DIR=${PATH_TO_OPENCV_CMAKE_DIR}

mkdir build && pushd build
cmake .. -DORT_HOME=${ORT_HOME} -DOpenCV_DIR=${OPENCV_DIR} -DCMAKE_BUILD_TYPE=Debug -DTEST=OFF -DDEMO=ON
make install -j`nproc`
popd

# Or with cross compiler:
CROSS_TOOL=${PATH_TO_COMPILER_PREFIX}-
SYSROOT=${PATH_TO_SYSROOT}
cmake .. -DORT_HOME=${ORT_HOME} -DOpenCV_DIR=${OPENCV_DIR} \
   -DCMAKE_C_COMPILER=${CROSS_TOOL}gcc -DCMAKE_CXX_COMPILER=${CROSS_TOOL}g++ -DCMAKE_SYSROOT=${SYSROOT}
```

### run demo

```bash
./classification_demo <modelFilepath> <labelFilepath> <imageFilepath>
./detection_stream_demo <configFilepath> <input> <inputType>
./detection_demo <modelFilepath> <labelFilepath> <imageFilepath> <saveImgpath>
or 
./detection_demo <configFilepath> <imageFilepath> <saveImgpath>
./detection_video_demo <configFilepath> <videoFilepath> <saveFilepath>
./estimation_demo <detConfigFilepath> <poseConfigFilepath> <imageFilepath> <saveImgpath>
./tracker_stream_demo <detConfigFilepath> <poseConfigFilepath> <input> <inputType>
```
