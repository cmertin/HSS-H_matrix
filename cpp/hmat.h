#ifndef HMAT_H
#define HMAT_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <cstring>
#include <omp.h>
#include "objects.h"
#include "ompUtils.h"
#include "mkl_lapacke.h"
#include "mkl.h"
//#include "cblas.h"
//#include "lapacke.h"

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
  HMat(unsigned int &n, unsigned int &m, unsigned int &min_rank);
  void MatVec(std::vector<T> &x, std::vector<T> &result);
  unsigned int NumSubMat();
  void AddSubMatrix(SubMatrix<T> &A);
  std::vector<T> DenseMatrix();
  void OutputAll();
  unsigned int CountSub();
  unsigned int GetRows();
  unsigned int GetCols();
  unsigned int MinRank();
  void Storage();

 private:
  unsigned int n; // rows
  unsigned int m; // columns
  unsigned int min_rank; // Minimum rank to store
  std::vector<SubMatrix<T> > subMatrices;
};

template <typename T>
HMat<T>::HMat(unsigned int &n, unsigned int &m, unsigned int &min_rank)
{
  this->n = n;
  this->m = m;
  this->min_rank = min_rank;
}

template<typename T>
unsigned int HMat<T>::NumSubMat()
{
  return (this->subMatrices).size();
}

template <typename T>
void HMat<T>::AddSubMatrix(SubMatrix<T> &A)
{
  (this->subMatrices).push_back(A);
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
unsigned int HMat<T>::MinRank()
{
  return this->min_rank;
}

template <typename T>
void BuildHMat(std::vector<T> &mat, HMat<T> &hmat, std::vector<Split> &sp, double &tol)//, unsigned int &min_rank)
{
  unsigned int n = hmat.GetRows();
  unsigned int m = hmat.GetCols();
  unsigned int min_rank = hmat.MinRank();
  for(int i = 0; i < sp.size(); ++i)
    {
      SubMatrix<T> subMat(mat, n, m, sp[i], tol, min_rank);
      Split sub = subMat.GetInfo();

      if(sub.k < std::min(sp[i].n, sp[i].m)/2 || sub.k <= min_rank)
	{
	  hmat.AddSubMatrix(subMat);
	}
      else
	{
	  std::vector<Split> tempSplit = MatrixSplit(sub);
	  BuildHMat(mat, hmat, tempSplit, tol);//, min_rank);
	}
    }
}

template <typename T>
void HMat<T>::Storage()
{
  unsigned int full = (this->n) * (this->m);
  unsigned int new_size = 0;
  for(unsigned int subMat = 0; subMat < NumSubMat(); ++subMat)
    {
      SubMatrix<T> temp = this->subMatrices[subMat];
      unsigned int n = temp.info.n;
      unsigned int m = temp.info.m;
      unsigned int k = temp.info.k;

      if(k <= this->min_rank)
	{
	  new_size += n * m;
	}
      else
	{
	  new_size += k * n + k * m;
	}
    }

  double percentage = ((double)new_size)/((double)full) * 100;

  std::cout << "Old Storage: " << full << "\nNew Storage: " << new_size << "\nPercentage:  " << std::setprecision(3) << std::fixed << percentage << '%' << std::endl;

  return;
}

template <typename T>
std::vector<T> HMat<T>::DenseMatrix()
{
  unsigned int nm = this->n * this->m;
  
  std::vector<T> denseMatrix(nm, -100);

  std::cout << NumSubMat() << std::endl;
  
  for(int subMat = 0; subMat < NumSubMat(); ++subMat)
    {
      SubMatrix<T> temp = this->subMatrices[subMat];
      unsigned int start_i = temp.info.i;
      unsigned int start_j = temp.info.j;
      unsigned int n_s = temp.info.n;
      unsigned int k_s = temp.info.k;
      unsigned int m_s = temp.info.m;
      unsigned int nm_s = n_s * m_s;
      std::vector<T> sub_submat(nm_s, 0);
      
      for(unsigned int i = 0; i < n_s; ++i)
	{
	  for(unsigned int j = 0; j < m_s; ++j)
	    {
	      for(unsigned int k = 0; k < k_s; ++k)
		{
		  unsigned int y_index = i * k_s + k;
		  unsigned int z_index = m_s * k + j;
		  unsigned int sub_index = i * m_s + j;
		  assert(y_index < temp.Yk.size());
		  assert(z_index < temp.Zk.size());
		  assert(sub_index < sub_submat.size());
		  sub_submat[sub_index] += temp.Yk[y_index] * temp.Zk[z_index];
		}
	    }
	}
      for(unsigned int i = 0; i < n_s; ++i)
	{
	  for(unsigned int j = 0; j < m_s; ++j)
	    {
	      unsigned int sub_index = i * m_s + j;
	      unsigned int mat_index = (start_i + i) * this->m + (j + start_j);
	      assert(sub_index < sub_submat.size());
	      assert(mat_index < denseMatrix.size());
	      denseMatrix[mat_index] = sub_submat[sub_index];
	    }
	}
    }

  return denseMatrix;
}

template <typename T>
unsigned int HMat<T>::CountSub()
{
  unsigned int count = 0;
  for(int subMat = 0; subMat < NumSubMat(); ++subMat)
    {
      SubMatrix<T> temp = this->subMatrices[subMat];
      count += temp.info.n * temp.info.m;
    }

  return count;
}

template <typename T>
void HMat<T>::OutputAll()
{
  std::cout << "Total Sub Matrices: " << NumSubMat() << std::endl;
  for(int subMat = 0; subMat < NumSubMat(); ++subMat)
    {
      SubMatrix<T> temp = this->subMatrices[subMat];
      std::cout << subMat << ":\t" << temp.info.i << '\t' << temp.info.j << '\t' << temp.info.k << '\t' << temp.info.n << '\t' << temp.info.m << std::endl;
    }
}

template <typename T>
void HMat<T>::MatVec(std::vector<T> &x, std::vector<T> &result)
{
  unsigned int sz = std::floor(x.size()/2);
  unsigned int start_i = 0;
  unsigned int start_j = 0;
  unsigned int n = 0;
  unsigned int k = 0;
  unsigned int m = 0;
  unsigned int im_temp = 0;
  unsigned int r_temp = 0;
  unsigned int num_submat = (this->subMatrices).size();
  int n_threads = omp_get_max_threads();//omp_get_num_threads();
  unsigned int split = std::floor(num_submat/n_threads);
  //std::cout << "Number of threads: " << n_threads << std::endl;

  typename std::vector<SubMatrix<T> >::iterator subMat = (this->subMatrices).begin();
  typename std::vector<SubMatrix<T> >::iterator end = (this->subMatrices).begin();
  std::vector<T> im_r;//(sz, 0.0);
  result.resize(x.size());
  std::vector<T> r(x.size() * n_threads, 0.0);
  std::memset(&(*(result.begin())), 0.0, result.size() * sizeof(result[0]));
  //std::cout << "Using " << n_threads << " threads" << std::endl;

  #pragma omp parallel default(none) private(subMat, end, im_r, start_i, start_j, n, k, m, im_temp, r_temp) shared(result, r, x, split, sz, n_threads) num_threads(n_threads)
  {
    unsigned int thread = omp_get_thread_num();
    unsigned int idx = 0;
    subMat = (this->subMatrices).begin() + split * thread;
    end = subMat + split;
    // Gives remainder to the last thread
    if(thread == n_threads-1)
      end = (this->subMatrices).end();
    im_r.resize(sz);
    for(subMat; subMat < end; ++subMat)
      {
	start_i = (*subMat).info.i;
	start_j = (*subMat).info.j;
	n = (*subMat).info.n;
	k = (*subMat).info.k;
	m = (*subMat).info.m;
	std::memset(&(*(im_r.begin())), 0.0, sz * sizeof(im_r[0]));

	if(k > this->min_rank)
	  {
	    // im_r = Zk * x
	    for(unsigned int k_ = 0; k_ < k; ++k_)
	      {
		im_temp = k_ * m;
		for(unsigned int j = 0; j < m; ++j)
		  {
		    im_r[k_] += (*subMat).Zk[im_temp + j] * x[j + start_j];
		  }
	      }
	    
	    // r = Yk * im_r
	    for(unsigned int i = 0; i < n; ++i)
	      {
		r_temp = i * k;
		idx = (i + start_i) * n_threads + thread;
		for(unsigned int k_ = 0; k_ < k; ++k_)
		  {
		    // not thread safe
		    //result[i + start_i] += (*subMat).Yk[r_temp + k_] * im_r[k_];
		    r[idx] += (*subMat).Yk[r_temp + k_] * im_r[k_];
		  }
	      }
	  }
	else
	  {
	    // r = Yk * x
	    for(unsigned int i = 0; i < n; ++i)
	      {
		r_temp = i * m;
		idx = (i + start_i) * n_threads + thread;
		for(unsigned int j = 0; j < m; ++j)
		  {
		    r[idx] += (*subMat).Yk[r_temp + j] * x[j + start_j];
		  }
	      }
	  }
      }
  }

  // This can be parallelized as well
  unsigned int idx = 0;
  #pragma omp parallel for private(idx)
  for(unsigned int i = 0; i < result.size(); ++i)
    {
      idx = i * n_threads;
      for(unsigned int t_id = 0; t_id < n_threads; ++t_id)
	{
	  result[i] += r[idx + t_id];
	}
    }
  
  return;
}

#endif
