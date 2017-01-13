BCC = g++
CFLAGS = -std=c++0x -O3
LAPAK = -L/usr/lib -llapack -lcblas
ICC = icpc
IFLAGS = -mkl -qopenmp
DBG = 

hmat: main.cpp hmat.h objects.h ompUtils.h ompUtils.tcc
	$(ICC) $(IFLAGS) $(DBG) -o hmat main.cpp

clean:
	rm -rf hmat test *~ \#*
