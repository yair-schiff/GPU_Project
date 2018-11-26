CC = nvcc
ARCH=sm_52
CFLAGS=-Wall -arch=$(ARCH)
OBJ = FW_helper.c FW.cu

FW: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: clean

clean:
	rm -f *.o FW
