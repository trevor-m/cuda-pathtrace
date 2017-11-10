NVCC=/usr/local/cuda/bin/nvcc
CUDAFLAGS=-I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -std=c++11

pathtrace: pathtrace.o
	$(NVCC) $^ -o $@

pathtrace.o: pathtrace.cu
	$(NVCC) $(CUDAFLAGS) -c $^
	
clean:
	rm *.o pathtrace
