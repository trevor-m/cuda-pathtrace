NVCC=/usr/local/cuda/bin/nvcc
NVFLAGS=-arch compute_30
CUDAFLAGS=-I include -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -I/usr/include/python2.7
LDLIBS=-lglfw -lGL -lX11 -lpthread -lXrandr -lXi -lGLEW -lGLU -lpython2.7 -lboost_system -lboost_python

pathtrace: pathtrace.o main.o denoise.o
	$(NVCC) $^ -o $@ $(LDLIBS) $(NVFLAGS)

pathtrace.o: src/pathtrace.cu
	$(NVCC) $(CUDAFLAGS) -c $^ $(NVFLAGS)

denoise.o: src/denoise.cu
	$(NVCC) $(CUDAFLAGS) -c $^ $(NVFLAGS)

main.o: src/main.cu
	$(NVCC) $(CUDAFLAGS) -c $^ $(NVFLAGS)
	
clean:
	rm -f *.o pathtrace
