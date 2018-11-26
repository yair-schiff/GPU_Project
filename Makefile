NVCC = nvcc
ARCH = 52
NVCCFLAGS = -arch=sm_$(ARCH)
OBJ = FW.cu

FW: $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

.PHONY: clean

clean:
	rm -f *.o FW
