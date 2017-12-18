NVCC=/usr/local/cuda/bin/nvcc
NVFLAGS=-arch compute_30
CUDAFLAGS=-I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -std=c++11
LDLIBS=-lglfw -lGL -lX11 -lpthread -lXrandr -lXi -lGLEW -lGLU

pathtrace: pathtrace.o main.o denoise.o
	$(NVCC) $^ -o $@ $(LDLIBS) $(NVFLAGS)

pathtrace.o: pathtrace.cu
	$(NVCC) $(CUDAFLAGS) -c $^ $(NVFLAGS)

denoise.o: denoise.cu
	$(NVCC) $(CUDAFLAGS) -c $^ $(NVFLAGS)

main.o: main.cu
	$(NVCC) $(CUDAFLAGS) -c $^ $(NVFLAGS)
	
clean:
	rm -f *.o pathtrace
