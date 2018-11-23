CC = gcc
CFLAGS=-Wall
OBJ = FW_seq.c

FW_seq:  $(OBJ)
	gcc $(CFLAGS) -o $@ $^

.PHONY: clean

clean:
	rm -f *.o FW_seq 
