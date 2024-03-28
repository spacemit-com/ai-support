import os
import sys
import argparse
import shlex
import subprocess
import shutil
import platform

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(1, os.path.join(REPO_DIR, "python")) #Temporary variables

def copy_files_with_prefix(src_dir, dst_dir, prefix):   
    if not os.path.exists(dst_dir):  
        os.makedirs(dst_dir)  
  
    for root, dirs, files in os.walk(src_dir):  
        for file in files:  
            if file.startswith(prefix):  
                src_file = os.path.join(root, file)  
                dst_file = os.path.join(dst_dir, file)  
                if os.path.exists(dst_file)==False:
                    shutil.copy2(src_file, dst_file)  

def parse_arguments():
    class Parser(argparse.ArgumentParser):
        # override argument file line parsing behavior - allow multiple arguments per line and handle quotes
        def convert_arg_line_to_args(self, arg_line):
            return shlex.split(arg_line)

    parser = Parser(
        description="Bianbu ai support build driver.",
        usage="""
        """,
        fromfile_prefix_chars="@",
    )
    # Main arguments
    parser.add_argument(
        "--config",
        nargs="+",
        default="Release",
        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
        help="Configuration(s) to build.",
    )
    parser.add_argument(
        "--parallel",
        nargs="?",
        default="1",
        type=int,
        help="Use parallel build. The optional value specifies the maximum number of parallel jobs. "
        "If the optional value is 0 or unspecified, it is interpreted as the number of CPUs.",
    )
    parser.add_argument(
        "--python",
        default=1,
        type=int,
        help="If build with demo with python wheel",
    )
    parser.add_argument(
        "--demo",
        default=1,
        type=int,
        help="If build with demo",
    )
    parser.add_argument("--ort_dir", help="Onnxruntime home path")

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    cmake_args = [ f"-DCMAKE_BUILD_TYPE={args.config}"] 
    cmake_args.append(f"-DORT_HOME={args.ort_dir}")
    if(args.demo):
        cmake_args.append(f"-DDEMO=ON")
    if(args.python):
        cmake_args.append(f"-DPYTHON=ON")
    # not used on MSVC, but no harm
    # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
    # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
    # from Python.
    build_args = [ f"-j"+str(args.parallel)]
    # Adding CMake arguments set as environment variable
    # (needed e.g. to build for ARM OSx on conda-forge)
    folder = os.path.exists("build")
    if not folder:
        os.makedirs("build")  
    os.chdir("build")
    print(cmake_args)
    subprocess.run(
        ["cmake", "..", *cmake_args], check=True
    )
    subprocess.run(["make", *build_args], check=True)
    if(args.python):
        copy_files_with_prefix('.', 'lib', 'lib')
        copy_files_with_prefix('.', 'lib', 'support.cpython')
        copy_files_with_prefix(os.path.join(args.ort_dir,"lib"), 'lib', 'lib')
        os.chdir("..")
        subprocess.run([sys.executable, "setup.py", "bdist_wheel"])
        if(platform.machine()=="x86_64"):
            os.chdir("dist")
            subprocess.run(["auditwheel", "repair", "support-0.0.1-py3-none-any.whl", "--plat=manylinux_2_35_x86_64"], check=True)
        if(platform.machine()=="riscv64"):
            os.chdir("dist")
            subprocess.run(["auditwheel", "repair", "support-0.0.1-py3-none-any.whl", "--plat=linux_riscv64"], check=True)

if __name__ == "__main__":
    main()
    