from setuptools import setup, Extension

ext_modules = [
    Extension(
        "support.wrapper",
        ["onnxruntime/wrapper/wrapper.cpython-310-x86_64-linux-gnu.so"],
    ),]

if __name__ == "__main__":
    # The information here can also be placed in setup.cfg - better separation of
    # logic and declaration, and simpler if you include description/version in a file.
    setup(
        name="support",
        version="0.0.1",
        author="Xinyu Ge",
        author_email="xinyu.ge@spacemit.com",
        description="A test project using pybind11 and CMake",
        long_description="",
        include_package_data = True,
        zip_safe=False,
        python_requires=">=3.6",
    )
