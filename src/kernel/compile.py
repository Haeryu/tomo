import os
import subprocess

def compile_cuda_to_dll(output_dll="cuda_project.dll", extra_nvcc_flags=None):
    """
    Compiles all .cu files in the current directory into a single DLL.
    
    :param output_dll: Name of the generated DLL file.
    :param extra_nvcc_flags: A list of extra arguments to pass to nvcc.
    """
    if extra_nvcc_flags is None:
        extra_nvcc_flags = []

    # Gather all .cu files in current directory
    cu_files = [os.getcwd() + "/src/kernel/" + f for f in os.listdir(os.getcwd() + "/src/kernel/") if f.endswith('.cu')]

    
    if not cu_files:
        print("No .cu files found in the current directory.")
        return

    # nvcc command
    # -shared tells nvcc to build a shared library (DLL on Windows)
    # -o output_dll specifies the output filename
    cmd = ["nvcc", "-shared", "-o", "./src/kernel/out/" + output_dll] + cu_files + extra_nvcc_flags

    print("Compiling with command:")
    print(" ".join(cmd))

    # Run the command
    try:
        subprocess.check_call(cmd)
        print(f"Successfully created {output_dll}")
    except subprocess.CalledProcessError as e:
        print("Error during compilation:", e)

if __name__ == "__main__":
    # Example usage:
    compile_cuda_to_dll(output_dll="tomo_kernels.dll",
                        extra_nvcc_flags=[
                            # Xcompiler
                            "-Xcompiler", 
                            "/MD", 
                            "-Xcompiler", 
                            "/wd4819", 
                            "-Xcompiler", 
                            "/wd4711", 
                            "-Xcompiler", 
                            "/wd4514", 
                            "-Xcompiler", 
                            "/wd5031", 
                            "-Xcompiler", 
                            "/wd4668", 
                            "-Xcompiler", 
                            "/wd5039", 
                            "-Xcompiler", 
                            "/wd4505", 
                            "-Xcompiler", 
                            "/wd4100", 
                            "-Xcompiler", 
                            "/W3", 
                            # Xlinker
                            "-Xlinker", 
                            "/NODEFAULTLIB:LIBCMT", 
                            # nvcc
                            "-O2", 
                            "-gencode=arch=compute_89,code=sm_89", 
                            "-std=c++20", 
                            "-rdc=true", 
                            "--expt-relaxed-constexpr",
                            "--use_fast_math", 
                            "--extended-lambda",
                            "--diag-suppress=221",
                                        ])  
