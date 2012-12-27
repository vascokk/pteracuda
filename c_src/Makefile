CC=cl

all: compile

compile: pcuda_ops.o pcuda_blas.o pcuda_kernels.o pcuda_string.o

pcuda_ops.o:
	@nvcc $(NVCCFLAGS) pcuda_ops.cu -o pcuda_ops.o

pcuda_blas.o:
	nvcc $(NVCCFLAGS) pcuda_blas.cu -o pcuda_blas.o 

pcuda_kernels.o:
	nvcc $(NVCCFLAGS) pcuda_kernels.cu -o pcuda_kernels.o

pcuda_string.o:
	nvcc $(NVCCFLAGS) pcuda_string.cu -o pcuda_string.o

clean:
	nvcc -c -clean *.cu
	rm *.o
