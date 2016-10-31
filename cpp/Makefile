CC = g++
CFLAGS = --std=c++0x -O3
LAPAK = -L/usr/lib -llapack -lcblas

hmat: main.cpp hmat.h
	$(CC) $(CFLAGS) -o hmat main.cpp $(LAPAK) 

clean:
	rm -rf hmat *~ \#*
