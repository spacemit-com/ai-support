import os, sys
import subprocess
from collections import namedtuple
from pathlib import Path
import shutil

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
#from setuptools.command.install import install as InstallCommandBase
#from setuptools.command.install_lib import install_lib as InstallLibCommandBase

package_name = "bianbuai"

SCRIPT_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, ".."))

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = TOP_DIR # os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve() / package_name / "capi"

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            #f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        build_args = ["--target", "bianbuai_pybind11_state"]
        install_args = ["--component", "pybind11", "--prefix", "."]
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # Pass other necessary args for pybind11 cmake project
        onnxruntime_install_dir = os.environ.get("ORT_HOME", "/usr")
        opencv4_cmake_dir = os.environ.get("OPENCV_DIR", "")
        cmake_args += [
            f"-DOpenCV_DIR={opencv4_cmake_dir}",
            f"-DORT_HOME={onnxruntime_install_dir}",
            f"-DPYTHON=ON",
        ]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        #print("cmake args:", cmake_args)
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )
        #print("build args:", build_args)
        subprocess.run(
            ["cmake", "--install", ".", *install_args], cwd=build_temp, check=True
        )
        #print("install args:", install_args)

        # Copy installed libraries files to Python modules directory
        shutil.copytree(os.path.join(build_temp, "lib"), f"{extdir}", dirs_exist_ok=True,)


# Get Git Version
try:
    git_version = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=TOP_DIR).decode('ascii').strip()
except (OSError, subprocess.CalledProcessError):
    git_version = None
print("GIT VERSION:", git_version)

# Get Release Version Number
with open(os.path.join(TOP_DIR, 'VERSION_NUMBER')) as version_file:
    VersionInfo = namedtuple('VersionInfo', ['version', 'git_version'])(
        version=version_file.read().strip(),
        git_version=git_version
    )

# Description
README = os.path.join(TOP_DIR, "README.md")
with open(README, encoding="utf-8") as fdesc:
    long_description = fdesc.read()


classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3 :: Only'
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
]

if __name__ == "__main__":
    # The information here can also be placed in setup.cfg - better separation of
    # logic and declaration, and simpler if you include description/version in a file.
    setup(
        name=package_name,
        version=VersionInfo.version,
        description="Bianbu AI Support Python Package",
        long_description=long_description,
        long_description_content_type="text/markdown",
        setup_requires=[],
        tests_require=[],
        cmdclass={"build_ext": CMakeBuild},
        packages=find_packages("python", exclude=[]),
        package_dir={"":"python"}, # tell distutils packages are under "python/"
        ext_modules=[CMakeExtension(package_name)],
        #package_data={},
        include_package_data=True,
        license='Apache License v2.0',
        author="bianbu-ai-support",
        author_email="bianbu-ai-support@spacemit.com",
        url='https://gitlab.dc.com:8443/bianbu/ai/support',
        install_requires=[],
        entry_points={},
        scripts=[],
        python_requires=">=3.6",
        classifiers=classifiers,
        zip_safe=False,
    )
