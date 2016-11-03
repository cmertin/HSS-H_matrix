#ifndef OBJECTS_H
#define OBJECTS_H
#include <cmath>
#include <vector>
#include <algorithm>
#include "lapacke.h"
#include "cblas.h"

struct Split
{
  unsigned int n;
  unsigned int m;
  unsigned int i;
  unsigned int j;
  unsigned int k;
};

// Returns the subMatrix from the original matrix
template <typename T>
T *MatrixSubset(T mat[], unsigned int &n, unsigned int &m, Split &sp)
{
  T *subMat = new T[sp.n * sp.m];
  for(unsigned int i = 0; i < sp.n; ++i)
    {
      for(unsigned int j = 0; j < sp.m; ++j)
	subMat[i * sp.n + j] = mat[(i + sp.i) * n + (j + sp.j)];
    }
  return subMat;
}

// Returns the norm of a vector
template <typename T>
double VecNorm(T vec[], unsigned int &n)
{
  double norm = 0;
  for(int i = 0; i < n; ++i)
    norm += vec[i] * vec[i];

  return std::sqrt(norm);
}

// Counts the number of non-zeros 
template <typename T>
unsigned int NNZ_Vec(T vec[], unsigned int &n)
{
  unsigned int count = 0;
  for(int i = 0; i < n; ++i)
    {
      if(vec[i] > 0)
	++count;
    }
  return count;
}

// s = vector of eigenvalues
// n = size of the vector
// k = "new rank"
// tol = tolerance
// Calculates the norm of a vector and removes
// elemenents until it's within a given tolerance * originalNorm
template <typename T>
void NormTol(double s[], unsigned int &n, unsigned int &k, double &tol)
{
  unsigned int index = n-1;
  double origNorm = VecNorm(s, n);
  double newNorm = origNorm;

  while(newNorm > origNorm * tol)
    {
      s[index] = 0.0;
      newNorm = VecNorm(s, index);
      --index;
    }
  return;
}

template <typename T>
class SubMatrix
{
 public:
  SubMatrix(T mat[], unsigned int &n, unsigned int &m, Split &sp, double &tol, unsigned int &min_rank);
  ~SubMatrix();
    
 private:
  Split info;
  T Yk[]; // Uk * Sk
  T Zk[]; // Vk^T
};

template <typename T>
SubMatrix<T>::SubMatrix(T mat[], unsigned int &n, unsigned int &m, Split &sp, double &tol, unsigned int &min_rank)
{
  T *subMat = MatrixSubset(mat, n, m, sp);
  unsigned int min_nm = std::min(sp.n,sp.m);
  T *U = new T[sp.n * sp.n];
  T *s = new T[min_nm];
  T *Vt = new T[sp.m * sp.m];
  unsigned int rank = 0;
  int n_s = (int)sp.n;
  int m_s = (int)sp.m;
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

  if(rank <= min_rank)
    {
      this->info.n = sp.n;
      this->info.m = sp.m;
      this->info.i = sp.i;
      this->info.j = sp.j;
      this->info.k = rank;

      this->Yk = new T[sp.n * sp.m];
      
      for(unsigned int i = 0; i < sp.n; ++i)
	{
	  for(unsigned int j = 0; j < sp.m; ++j)
	    this->Yk[i * sp.n + j] = subMat[i * sp.n + j];
	}
      
      this->Zk = new T[sp.m * sp.m];
      std::fill_n(this->Zk, sp.m * sp.m, 0); // Initialize with zeros
      
      for(unsigned int i = 0; i < sp.m; ++i)
	this->Zk[i * sp.m + i] = 1.0;

      delete subMat;
      delete U;
      delete Vt;
      delete s;
      delete work; 
    }
  else
    {
      NormTol(s, min_nm, rank, tol);

      // Creates s into diagonal matrix (S)
      T *S = new T[rank * rank];
      std::fill_n(S, rank * rank, 0);
      for(int i = 0; i < rank; ++i)
	S[i * rank + i] = s[i];
      
      this->Yk = new T[sp.n * rank];
      this->Zk = new T[rank * sp.m];
      
      for(int i = 0; i < rank; ++i)
	{
	  for(int j = 0; j < sp.m; ++j)
	    this->Zk[i * sp.m + j] = Vt[i * sp.m + j];
	}
      
      // Yk = Uk * Sk
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, rank, rank, alpha, U, n, S, rank, beta, this->Yk, n);
      
      this->info.n = sp.n;
      this->info.m = sp.m;
      this->info.i = sp.i;
      this->info.j = sp.j;
      this->info.k = rank;
      
      delete U;
      delete Vt;
      delete S;
      delete s;
      delete work;
    }
}

template <typename T>
SubMatrix<T>::~SubMatrix()
{
  delete this->Yk;
  delete this->Zk;
}



#endif
