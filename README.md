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
export ORT_HOME=
mkdir build
cd build
cmake -DORT_HOME=${ORT_HOME} -DBUILD_TYPE=${CMAKE_BUILD_TYPE} -DDEBUG=OFF -DTEST=ON ..
make -j16
make install
```

### run demo

```bash
./detection_demo
./detection_stream_demo
./classification_demo 
```
