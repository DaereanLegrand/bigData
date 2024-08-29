NVCC = nvcc
NVCCFLAGS = -arch=sm_75
CUDA_LIBS = -lcudart -lcublas -lcurand

main:
	nvcc main.cu $(NVCCFLAGS) $(CUDA_LIBS) -o $@
