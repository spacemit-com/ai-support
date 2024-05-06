## prepare env

```shell
# for ubuntu
sudo apt install python3-dev
# for centos
sudo yum install python3-devel
```

## quick build

* cmake project

```shell
mkdir build && cd build
# Note: static opencv libraries is required
cmake .. -DORT_HOME=${PATH_TO_ONNXRUNTIME} -DOpenCV_DIR=${PATH_TO_OPENCV_CMAKE_DIR} -DPYTHON=ON
make -j`nproc` bianbuai_pybind11_state VERBOSE=1
# Or
cmake --build . --config Release --verbose
```

## smoke test

```python
import bianbuai_pybind11_state as bianbuai
```