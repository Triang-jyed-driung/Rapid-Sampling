"""Local editable installation of the CUDA extension."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rapid-sampling-cuda",
    version="0.1.0",
    py_modules=["rapid_sampling"],
    ext_modules=[CUDAExtension(
        "_rapid_sampling", ["sampling.cu"],
        extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-std=c++17"]},
    )],
    cmdclass={"build_ext": BuildExtension},
)
