CC = g++
CFLAGS = --std=c++0x -O3

matvec: matvec.cpp
	$(CC) $(CFLAGS) -o matvec matvec.cpp

clean:
	rm -f matvec *~ \#* 
