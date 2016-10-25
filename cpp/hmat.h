#ifndef HMAT_H
#define HMAT_H

#include <vector>
#include <algorithm>
#include <cmath>

struct Split
{
  unsigned int n;
  unsigned int m;
  unsigned int i;
  unsigned int j;
};

struct SubMatrix
{
  Split info; // start_i, start_j, n, k, m
  std::vector<double> Y; // Uk * Sk
  std::vector<double> Z; // Vk^T
};

struct HMat
{
  std::vector<SubMatrix> submatrices;
  unsigned int n;
  unsigned int m;
};

std::vector<Split> MatrixSplit(unsigned int n, unsigned int m, unsigned int i = 0, unsigned int j = 0)
{
  std::vector<Split> splits(4);
  unsigned int n1 = std::floor(n/2) + (n % 2);
  unsigned int n2 = std::floor(n/2);
  unsigned int m1 = std::floor(m/2) + (m % 2);
  unsigned int m2 = std::floor(m/2);

  Split sp1;
  sp1.n = n1;
  sp1.m = m1;
  sp1.i = i;
  sp1.j = j;
  splits[0] = sp1;

  sp1.n = n1;
  sp1.m = m2;
  sp1.i = i;
  sp1.j = j + m1;
  splits[1] = sp1;

  sp1.n = n2;
  sp1.m = m1;
  sp1.i = i + n1;
  sp1.j = j;
  splits[2] = sp1;
  
  sp1.n = n2;
  sp1.m = m2;
  sp1.i = i + n1;
  sp1.j = j + m1;
  splits[3] = sp1;
  
  return splits;
}

double *SubMatrix(double &mat[], unsigned int &n, unsigned int &m, Split &sp)
{
  double *subMat = new double[sp.n * sp.m];
  for(unsigned int i = 0; i < sp.n; ++i)
    {
      for(unsigned int j = 0; j < sp.m; ++j)
	{
	  subMat[i * sp.n + j] = mat[(i + sp.i) * n + (j + sp.j)];
	}
    }

  return subMat;
}

unsigned int NNZ_Vec(double &vec[], unsigned int &n)
{
  unsigned int count = 0;
  for(int i = 0; i < n; ++i)
    {
      if(vec[i] > 0)
	++count;
    }
  return count;
}

double VecNorm(double &vec[], unsigned int &n)
{
  double norm = 0;
  for(int i = 0; i < n; ++i)
    norm += vec[i] * vec[i];

  return std::sqrt(norm);
}

// s = vector of eigenvalues
// n = size of the vector
// k = "new rank"
// tol = tolerance
void EigTol(double &s[], unsigned int &n, unsigned int &k, double &tol)
{
  unsigned int index = n-1;
  double norm1 = VecNorm(s, n);
  double norm2 = norm1;
  while(norm2 > tol * norm1)
    {
      s[index] = 0.0;
      norm2 = VecNorm(s, index);
      index = index - 1;
    }
  k = index;
  return;
}

// n = rows
// m = columns
// http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga6a6ce95c3fd616a7091df45287c75cfa.html#ga6a6ce95c3fd616a7091df45287c75cfa
unsigned int SVD(double mat[], double &Yk[], double &Zk[], unsigned int &n, unsigned int &m, unsigned int &k, double &tol)
{
  double *U = new double[n * n];
  double *S = new double[n * m];
  double *s = new double[std::min(n, m)];
  double *Vt = new double[m * m];
  unsigned int rank = 0;

  dgesvd('A', 'A', n, m, mat, n, s, U, n, Vt, m);

  rank = NNZ_Vec(s, std::min(n,m));

  EigTol(s, std::min(n,m), k, tol);

  // Creates s into diagonal matrix (S)
  for(int i = 0; i < k; ++i)
    S[i * n + i] = s[i];

  double *Y = new double[n * k];
  double *Z = new double[k * m];

  for(int i = 0; i < k; ++i)
    {
      for(int j = 0; j < m; ++j)
	Z[i * m + j] = Vt[i * m + j];
    }

  // Y = Uk * Sk
  dgemm('N', 'N', n, k, k, 1.0, U, n, S, k, 0.0, Y, n);

  Yk = Y;
  Zk = Z;

  delete U[];
  delete S[];
  delete s[];
  delete Vt[];

  return rank;
}

#endif
