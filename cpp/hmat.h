#ifndef HMAT_H
#define HMAT_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cblas.h>
#include <lapacke.h>

struct Split
{
  unsigned int n;
  unsigned int m;
  unsigned int i;
  unsigned int j;
  unsigned int k;
};

struct SubMatrix
{
  Split info; // start_i, start_j, n, k, m
  std::vector<double> Y; // Uk * Sk
  std::vector<double> Z; // Vk^T
}temp;

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

double *SubMatrix(double mat[], unsigned int &n, unsigned int &m, Split &sp)
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

unsigned int NNZ_Vec(double vec[], unsigned int &n)
{
  unsigned int count = 0;
  for(int i = 0; i < n; ++i)
    {
      if(vec[i] > 0)
	++count;
    }
  return count;
}

double VecNorm(double vec[], unsigned int &n)
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
void EigTol(double s[], unsigned int &n, unsigned int &k, double &tol)
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
// k = new/low rank after SVD truncation
// mat = matrix to take SVD of
// Yk = Uk * Sk
// Zk = Vk^T
// tol = tolerance
// http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga6a6ce95c3fd616a7091df45287c75cfa.html#ga6a6ce95c3fd616a7091df45287c75cfa
unsigned int SVD(double mat[], double Yk[], double Zk[], unsigned int &n, unsigned int &m, unsigned int &k, double &tol)
{
unsigned int min_nm = std::min(n,m);
  double *U = new double[n * n];
  double *S = new double[n * m];
  double *s = new double[min_nm];
  double *Vt = new double[m * m];
  unsigned int rank = 0;
  int n_s = (int)n;
  int m_s = (int)m;
  lapack_int * n_ = &n_s;
  lapack_int * m_ = &m_s;
  char jobChar = 'A';
  char *job = &jobChar;
  int info = 0;
  int lWork = -1;
  lapack_int *info_ = &info;
  lapack_int *lWork_ = &lWork;
  double *work = new double[1];
  double alpha = 1.0;
  double beta = 0.0;

  LAPACK_dgesvd(job, job, n_, m_, mat, n_, s, U, n_, Vt, m_, work, lWork_, info_);

  rank = NNZ_Vec(s, min_nm);

  EigTol(s, min_nm, k, tol);

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
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, k, k, alpha, U, n, S, k, beta, Y, n);

  Yk = Y;
  Zk = Z;

  delete U;
  delete S;
  delete s;
  delete Vt;

  return rank;
}

void MatVec(HMat &A, double x[], double result[], unsigned int &rows)
{
  assert(A.m == rows);

  for(int i = 0; i < rows; ++i)
    result[i] = 0.0;

  for(int submat = 0; submat < A.submatrices.size(); ++submat)
    {
      temp = A.submatrices[submat];
      unsigned int start_i = temp.info.i;
      unsigned int start_j = temp.info.j;
      unsigned int n = temp.info.n;
      unsigned int k = temp.info.k;
      unsigned int m = temp.info.m;
      for(int i = 0; i < n; ++i)
	{
	  unsigned int res_index = start_i + i;
	  for(int j = 0; j < m; ++j)
	    {
	      unsigned int x_index = start_j + j;
	      for(int k_ = 0; k_ < k; ++k)
		{
		  unsigned int y_index = i * k + k_;
		  unsigned int z_index = j * k + k_;
		  result[res_index] += temp.Y[y_index] * temp.Z[z_index] * x[x_index];
		}
	    }
	}
    }
  return;
}

#endif
