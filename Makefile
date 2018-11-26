NVCC = nvcc
ARCH = sm_52
NVCCFLAGS = -arch=$(ARCH)
OBJ = FW_helper.c FW.cu

FW: $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

.PHONY: clean

clean:
	rm -f *.o FW
