"""Build the KronQ inference CUDA kernels. Run from this directory:
    python setup.py install
(requires nvcc on PATH matching your torch CUDA version, e.g. CUDA 12.1.)

Provides `kronq_kernels` with:
  matvec_int{4,2}      - fused dequant + int4/int2-weight x fp16-activation matvec
  matvec_int{4,2}_v3   - same, warp-per-row + uint4 vectorized loads + internal sum_x
  input_cast_scale     - fused fp16->fp32 cast * input rescale  (BiIP input stage)
  output_scale_cast    - fused output rescale * -> fp16          (BiIP output stage)
"""
from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='kronq_kernels',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'kronq_kernels',
            ['matvec_int4.cu'],
            extra_compile_args={
                'cxx': ['-std=c++17', '-O3'],
                'nvcc': ['-O3', '-std=c++17', '-Xcompiler', '-std=c++17',
                         '-Xcompiler', '-fPIC', '-Xcompiler', '-O3', '-lineinfo'],
            },
        ),
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
