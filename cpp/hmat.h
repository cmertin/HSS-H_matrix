#ifndef HMAT_H
#define HMAT_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include "objects.h"
#include "cblas.h"
#include "lapacke.h"

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

std::vector<Split> MatrixSplit(Split &tempSplit)
{
  std::vector<Split> splits(4);
  unsigned int n = tempSplit.n;
  unsigned int m = tempSplit.m;
  unsigned int i = tempSplit.i;
  unsigned int j = tempSplit.j;
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

template <typename T>
class HMat
{
 public:
  HMat(unsigned int &n, unsigned int &m);
  void MatVec(T x[], T result[], unsigned int &rows);
  unsigned int NumSubMat();
  void AddSubMatrix(SubMatrix<T> &A);
  unsigned int GetRows();
  unsigned int GetCols();

 private:
  unsigned int n; // rows
  unsigned int m; // columns
  std::vector<SubMatrix<T> > subMatrices;
};

template <typename T>
HMat<T>::HMat(unsigned int &n, unsigned int &m)
{
  this->n = n;
  this->m = m;
}

template<typename T>
unsigned int HMat<T>::NumSubMat()
{
  return this->subMatrices.size();
}

template <typename T>
void HMat<T>::AddSubMatrix(SubMatrix<T> &A)
{
  this->subMatrices.push_back(A);
  return;
}

template <typename T>
unsigned int HMat<T>::GetRows()
{
  return this->n;
}

template <typename T>
unsigned int HMat<T>::GetCols()
{
  return this->m;
}

template <typename T>
void BuildHMat(T mat[], HMat<T> &hmat, std::vector<Split> &splits, double &tol, unsigned int &min_rank)
{
  unsigned int n = hmat.GetRows();
  unsigned int m = hmat.GetCols();
  for(int i = 0; i < splits.size(); ++i)
    {
      SubMatrix<T> subMat(mat, n, m, splits[i], tol, min_rank);
      Split sub = subMat.GetInfo();

      if(sub.k < std::min(sub.n, sub.m)/2)
	{
	  hmat.AddSubMatrix(subMat);
	}
      else
	{
	  std::vector<Split> tempSplit = MatrixSplit(sub);
	  BuildHMat(mat, hmat, tempSplit, tol, min_rank);
	}
    }
}


/*
void MatVec(double x[], double result[], unsigned int &rows)
{
  assert(this->m == rows);

  for(int i = 0; i < rows; ++i)
    result[i] = 0.0;

  for(int submat = 0; submat < NumSubMat(); ++submat)
    {
      temp = this->subMatrices[submat];
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
*/

#endif
