#ifndef OBJECTS_H
#define OBJECTS_H
#include <cmath>
#include <vector>
#include <algorithm>
#include "hmat.h"
#include "mkl_lapacke.h"
#include "mkl.h"
//#include "lapacke.h"
//#include "cblas.h"

struct Split
{
  unsigned int n; // rows
  unsigned int m; // columns
  unsigned int i; // start row index
  unsigned int j; // start column index
  unsigned int k; // rank
};

// Returns the subMatrix from the original matrix
template <typename T>
std::vector<T> MatrixSubset(std::vector<T> &mat, unsigned int &n, unsigned int &m, Split &sp)
{
  std::vector<T> subMat(sp.n * sp.m, 0.0);
  for(unsigned int i = 0; i < sp.n; ++i)
    {
      for(unsigned int j = 0; j < sp.m; ++j)
	{
	  unsigned int s_idx = i * sp.m + j;
	  unsigned int m_idx = (i + sp.i) * m + (j + sp.j);
	  assert(s_idx < subMat.size());
	  assert(m_idx < mat.size());
	  subMat[s_idx] = mat[m_idx];
	}
    }
  return subMat;
}

// Returns the norm of a vector
template <typename T>
double VecNorm(std::vector<T> &vec)
{
  double norm = 0;
  for(int i = 0; i < vec.size(); ++i)
    norm += vec[i] * vec[i];

  return std::sqrt(norm);
}

// Counts the number of non-zeros 
template <typename T>
unsigned int NNZ_Vec(std::vector<T> &vec)
{
  unsigned int count = 0;
  for(int i = 0; i < vec.size(); ++i)
    {
      if(vec[i] > 0)
	++count;
    }
  return count;
}

// s = vector of eigenvalues
// k = "new rank"
// tol = tolerance
// Calculates the norm of a vector and removes
// elemenents until it's within a given tolerance * originalNorm
template <typename T>
void NormTol(std::vector<T> &s, unsigned int &k, double &tol)
{
  double origNorm = VecNorm(s);
  double newNorm = origNorm;
  
  while(newNorm > origNorm * tol)
    {
      s.pop_back();
      newNorm = VecNorm(s);
    }

  k = s.size();
  return;
}

template <typename T>
class SubMatrix
{
  template <typename>
  friend class HMat;
 public:
  SubMatrix(std::vector<T> &mat, unsigned int &n, unsigned int &m, Split &sp, double &tol, unsigned int &min_rank);
  SubMatrix();
  ~SubMatrix();
  SubMatrix(const SubMatrix &rhs);
  unsigned int GetRank();
  unsigned int GetRows();
  unsigned int GetCols();
  unsigned int GetRowIndex();
  unsigned int GetColIndex();
  Split GetInfo();
    
 private:
  Split info;
  std::vector<T> Yk; // Uk * Sk
  std::vector<T> Zk; // Vk^T
};

template <typename T>
SubMatrix<T>::SubMatrix(std::vector<T> &mat, unsigned int &n, unsigned int &m, Split &sp, double &tol, unsigned int &min_rank)
{
  std::vector<T> subMat = MatrixSubset(mat, n, m, sp);
  unsigned int min_nm = std::min(sp.n, sp.m);
  unsigned int sz1 = sp.n * min_nm;
  unsigned int sz2 = sp.m * min_nm;
  std::vector<T> U(sz1, 0.0);
  std::vector<T> s(min_nm, 0.0);
  std::vector<T> Vt(sz2, 0.0);
  unsigned int rank = min_nm;
  MKL_INT n_ = (int)sp.n;
  MKL_INT m_ = (int)sp.m;
  MKL_INT min_nm_ = (int)min_nm;
  int info = 0;
  double alpha = 1.0;
  double beta = 0.0;
  double *superb = new double[min_nm - 1];

  if(rank <= min_rank)
    {
      this->info.n = sp.n;
      this->info.m = sp.m;
      this->info.i = sp.i;
      this->info.j = sp.j;
      this->info.k = sp.m;

      (this->Yk).resize(sp.n * sp.m, 0.0);
      (this->Zk).resize(sp.m * sp.m, 0.0);

      this->Yk = subMat;
      
      // Set Zk to identity matrix
      for(unsigned int i = 0; i < sp.m; ++i)
	{
	  unsigned int idx = i * sp.m + i;
	  assert(idx < this->Zk.size());
	  this->Zk[i * sp.m + i] = 1.0;
	}

      delete [] superb;
    }
  else
    {
      info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', n_, m_, 
			    &(*(subMat.begin())), m_, &(*(s.begin())), 
			    &(*(U.begin())), min_nm_, &(*(Vt.begin())), 
			    m_, superb);

      assert(info == 0);

      NormTol(s, rank, tol);

      // Creates s into diagonal matrix (S)
      std::vector<T> S(rank * rank, 0.0);
      for(int i = 0; i < rank; ++i)
	{
	  unsigned int idx = i * rank + i;
	  assert(idx < S.size());
	  S[idx] = s[i];
	}

      std::vector<double> U_small(sp.n * rank);

      // Create U_small = U_k
      for(int i = 0; i < sp.n; ++i)
	{
	  for(int j = 0; j < rank; ++j)
	    {
	      unsigned int Uidx = i * min_nm + j;
	      unsigned int Us_idx = i * rank + j;
	      assert(Uidx < U.size());
	      assert(Us_idx < U_small.size());
	      U_small[Us_idx] = U[Uidx];
	    }
	}

      (this->Yk).resize(sp.n * rank);
      (this->Zk).resize(rank * sp.m);

      // Set Zk = Vk^T
      for(int i = 0; i < rank; ++i)
	{
	  for(int j = 0; j < sp.m; ++j)
	    {
	      unsigned int idx_zk = i * sp.m + j;
	      unsigned int idx_vt = i * sp.m + j;
	      assert(idx_zk < this->Zk.size());
	      assert(idx_vt < Vt.size());
	      this->Zk[idx_zk] = Vt[idx_vt];
	    }
	}

      // Yk = Uk * Sk
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, sp.n, rank, 
		  rank, alpha, &(*(U_small.begin())), rank, &(*(S.begin())), 
		  rank, beta, &(*((this->Yk).begin())), rank);
      
      this->info.n = sp.n;
      this->info.m = sp.m;
      this->info.i = sp.i;
      this->info.j = sp.j;
      this->info.k = rank;
      delete [] superb;
    }
}

template <typename T>
SubMatrix<T>::SubMatrix()
{
  // Do nothing
}

template <typename T>
SubMatrix<T>::~SubMatrix()
{
  // Do nothing
}

template <typename T>
SubMatrix<T>::SubMatrix(const SubMatrix &rhs)
{
  this->info = rhs.info;
  this->Yk = rhs.Yk;
  this->Zk = rhs.Zk;
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
