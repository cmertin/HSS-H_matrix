#include <iostream>
#include <random>
#include <cblas.h>

using namespace std;

int main()
{
  mt19937_64 rnd;
  uniform_real_distribution<double> doubleDist(0, 1);

  unsigned int n = 100;
  double *A = new double[n * n];
  for(unsigned int i = 0; i < n; ++i)
    {
      for(unsigned int j = 0; j < n; ++j)
	A[i*n + j] = doubleDist(rnd);
    }
  return 0;
}
