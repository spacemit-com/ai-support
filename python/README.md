## prepare env

```shell
# for ubuntu
sudo apt install python3-dev python3-pip cmake gcc g++ onnxruntime
# for centos
sudo yum install python3-devel python3-pip cmake gcc gcc-c++ onnxruntime

python3 -m pip install wheel setuptools
```

## quick build

* cmake project

```shell
git submodule update --init --recursive

mkdir build && cd build
# Note: static opencv libraries is required
cmake .. -DORT_HOME=${PATH_TO_ONNXRUNTIME} -DOpenCV_DIR=${PATH_TO_OPENCV_CMAKE_DIR} -DPYTHON=ON
make -j`nproc` bianbuai_pybind11_state VERBOSE=1
# Or
cmake --build . --config Release --verbose -j`nproc` --target bianbuai_pybind11_state
cmake --install . --config Release --verbose --component pybind11 # --strip
```

* python package

```shell
export ORT_HOME=${PATH_TO_ONNXRUNTIME}
export OPENCV_DIR=${PATH_TO_OPENCV_CMAKE_DIR}
python python/setup.py sdist bdist_wheel
```

## smoke unittest

```shell
# prepare env, e.g. with ubuntu22.04
python3 -m pip install opencv-python
# or just
sudo apt install python3-opencv

# run unittest under build diretctory
ln -sf ../rootfs/usr/share/ai-support data
cp ../tests/python/test_python_task.py .
python3 test_python_task.py
```