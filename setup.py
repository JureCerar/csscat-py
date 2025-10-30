from setuptools import setup, Extension
import pybind11

try:
    # Try to build with CUDA support
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    cmdclass = {"build_ext": BuildExtension}
    ext_modules = [
        CUDAExtension(
            "_core",
            sources=["csscat/core.cpp", "csscat/cudacore.cu"],
            include_dirs=[
                pybind11.get_include(),
                "include",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-fopenmp", "-D_GPU"],
                "nvcc": ["-O3"],
            },
            extra_link_args=["-fopenmp"],
            language="c++",
        )
    ]

except ImportError:
    # Fallback to CPU-only build
    cmdclass = {}
    ext_modules = [
        Extension(
            "_core",
            sources=["csscat/core.cpp"],
            include_dirs=[
                pybind11.get_include(),
                "include",
            ],
            extra_compile_args=["-O3", "-fopenmp"],
            extra_link_args=["-fopenmp"],
            language="c++"
        )
    ]

# Setup the package
setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_package_data=True,
    extras_require={
        "gpu": ["torch>=2.0"],
    }
)
