set CUDA_HOME="C:\Progra~1\NVIDIA~1\CUDA\v5.0"
cd c_src
REM nvcc -arch=sm_35 -O2 -c -Xcompiler  -I%CUDA_HOME%\\include pcuda_ops.cu -o pcuda_ops.o
REM nvcc -arch=sm_35 -O2 -c -Xcompiler  -I%CUDA_HOME%\\include pcuda_string.cu -o pcuda_string.o

del *.o
nvcc -O2 -c  -Xcompiler  -I%CUDA_HOME%\\include pcuda_kernels.cu -o pcuda_kernels.o
nvcc -O2 -c  -Xcompiler  -I%CUDA_HOME%\\include pcuda_ops.cu -o pcuda_ops.o
nvcc -O2 -c  -Xcompiler  -I%CUDA_HOME%\\include pcuda_blas.cu -o pcuda_blas.o
nvcc -O2 -c  -Xcompiler  -I%CUDA_HOME%\\include pcuda_string.cu -o pcuda_string.o
nvcc -O2 -c  -Xcompiler  -I%CUDA_HOME%\\include pcuda_ml.cu -o pcuda_ml.o


set CUDA_HOME="\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0"