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
	{
	  //std::cout << i << '\t' << j << std::endl;
	  subMat[i * sp.n + j] = mat[(i + sp.i) * n + (j + sp.j)];
	}
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
void NormTol(T s[], unsigned int &n, unsigned int &k, double &tol)
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
  unsigned int GetRank();
  unsigned int GetRows();
  unsigned int GetCols();
  unsigned int GetRowIndex();
  unsigned int GetColIndex();
  Split GetInfo();
    
 private:
  Split info;
  T *Yk; // Uk * Sk
  T *Zk; // Vk^T
};

template <typename T>
SubMatrix<T>::SubMatrix(T mat[], unsigned int &n, unsigned int &m, Split &sp, double &tol, unsigned int &min_rank)
{
  T *subMat = MatrixSubset(mat, n, m, sp);
  unsigned int min_nm = std::min(sp.n,sp.m);
  int min_nm_ = (int)min_nm;
  T *U = new T[sp.n * sp.n];
  T *s = new T[min_nm];
  T *Vt = new T[sp.m * sp.m];
  unsigned int rank = 0;
  int n_s = (int)sp.n;
  int m_s = (int)sp.m;
  lapack_int * n_ = &n_s;
  lapack_int * m_ = &min_nm_;//&m_s;
  char jobChar = 'A';
  char *job = &jobChar;
  int info = 0;
  int lWork = -1;
  lapack_int *info_ = &info;
  lapack_int *lWork_ = &lWork;
  double *work = new double[1];
  double alpha = 1.0;
  double beta = 0.0;

  std::cout << subMat[2] << std::endl;

  LAPACK_dgesvd(job, job, n_, m_, subMat, n_, s, U, n_, Vt, m_, work, lWork_, info_);

  std::cout << "min: " << min_nm << std::endl;
  rank = NNZ_Vec(s, min_nm);

  if(rank <= min_rank)
    {
      this->info.n = sp.n;
      this->info.m = sp.m;
      this->info.i = sp.i;
      this->info.j = sp.j;
      this->info.k = rank;

      // Possible bug: http://stackoverflow.com/questions/751878/determine-array-size-in-constructor-initializer
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

      delete [] subMat;
      delete [] U;
      delete [] Vt;
      delete [] s;
      delete [] work; 
    }
  else
    {
     
      NormTol(s, min_nm, rank, tol);

      // Creates s into diagonal matrix (S)
      T *S = new T[rank * rank];
      std::fill_n(S, rank * rank, 0);
      for(int i = 0; i < rank; ++i)
	S[i * rank + i] = s[i];

      std::cout << s[0] << '\t' << S[0] << std::endl;
      
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

      delete [] subMat;
      delete [] U;
      delete [] Vt;
      delete [] S;
      delete [] s;
      //delete [] work;
      std::cout << "Do you believe in magic?" << std::endl;

    }
}

template <typename T>
SubMatrix<T>::~SubMatrix()
{
  delete [] this->Yk;
  delete [] this->Zk;
  /*
  free(this->Yk);
  free(this->Zk);
  this->Yk = NULL;
  this->Zk = NULL;
  */
}

template <typename T>
unsigned int SubMatrix<T>::GetRank()
{
  return this->info.k;
}

template <typename T>
unsigned int SubMatrix<T>::GetRows()
{
  return this->info.n;
}

template <typename T>
unsigned int SubMatrix<T>::GetCols()
{
  return this->info.m;
}

template <typename T>
unsigned int SubMatrix<T>::GetRowIndex()
{
  return this->info.i;
}

template <typename T>
unsigned int SubMatrix<T>::GetColIndex()
{
  return this->info.j;
}

template <typename T>
Split SubMatrix<T>::GetInfo()
{
  return this->info;
}

#endif
