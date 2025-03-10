import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ext_src_root = os.path.join(BASE_DIR, "_ext_src")
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_include_dirs = glob.glob("{}/include/*".format(_ext_src_root))

print(_include_dirs)

setup(
    name='spherical',
    packages = find_packages(),
    ext_modules=[
        CUDAExtension(
            name='spherical._ext',
            sources=_ext_sources,
            include_dirs = [os.path.join(_ext_src_root, "include")],
            extra_compile_args={
                # "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                # "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                "cxx": [],
                "nvcc": ["-O3", 
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)}
)
