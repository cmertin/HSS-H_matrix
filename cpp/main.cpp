#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
//#include <chrono>
#include "hmat.h"
#include "mkl.h"
#include <omp.h>
// For checking openmp version
//#include <unordered_map>
//#include <cstdio>

using namespace std;

void ReadMatrix(vector<double> &mat, string &filename, unsigned int &n);
void ReadVector(vector<double> &vec, string &filename);
double CompareResult(vector<double> &x1, vector<double> &x2);
void FullMatVec(vector<double> &mat, unsigned int &n, unsigned int &m, vector<double> &vec, vector<double> &res);

int main()
{
  string matFile = "mob_H-1000.bin";
  unsigned int n = 1000;
  unsigned int n_ = 3 * n;
  unsigned int nm = n_ * n_;
  vector<double> mat(nm);
  vector<double> mat_cpy(nm);
  double tol = 0.99;
  unsigned int min_rank = 16;
  vector<Split> fullSplit = MatrixSplit(n_, n_);
  vector<double> x1;
  vector<double> x2;
  vector<double> res;
  ReadMatrix(mat, matFile, nm);
  string vecFile = "vec.dat";
  string resFile = "result.dat";
  unsigned int vec_cols = 1;
  double alpha = 1.0;
  double beta = 0.0;
  double h_time = 0.0;
  double n_time = 0.0;
  clock_t t;
  unsigned int num_itr = 1;
  mat_cpy = mat;
  srand(500);

  //cout << _OPENMP << endl;
  //unordered_map<unsigned, std::string> map{{200505, "2.5"},{200805,"3.0"},{201107, "3.1"},{201307, "4.0"},{201511, "4.5"}};
  //cout << "OpenMP Version: " << map.at(_OPENMP).c_str() << endl;


  /*
  x1.resize(n_);
  
  for(int i = 0; i < x1.size(); ++i)
    x1[i] = rand() % 100;
  */

  ReadVector(x1, vecFile);
  x2.resize(x1.size());
  
  HMat<double> hmat(n_, n_, min_rank);

  cout << "Building HMat..." << endl;
  BuildHMat(mat, hmat, fullSplit, tol);

  //vector<double> fullMat = hmat.DenseMatrix();


  //hmat.OutputAll();

  cout << "Performing matvec..." << endl;
  t = clock();
  for(unsigned int i = 0; i < num_itr; ++i)
    {
      hmat.MatVec(x1, res);
    }
  t = clock() - t;
  h_time = ((double)t/CLOCKS_PER_SEC)/((double)num_itr);

  cout << "Performing normal matvec..." << endl;
  t = clock();
  for(unsigned int i = 0; i < num_itr; ++i)
    {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_, vec_cols, n_, alpha, &(*(mat_cpy.begin())), n_, &(*(x1.begin())), vec_cols, beta, &(*(x2.begin())), vec_cols);
      //FullMatVec(mat_cpy, n_, n_, x1, x2);
    }

  t = clock() - t;
  n_time = ((double)t/CLOCKS_PER_SEC)/((double)num_itr);
  

  double norm = CompareResult(res, x2) * 100;
  
  cout << "Hierarchial Time: " << h_time << " s" << endl;
  cout << "CBLAS Time:       " << n_time << " s" << endl;

  cout << "Diff norm:   " << setprecision(3) << fixed << norm << '%' <<  endl;


  //hmat.Storage();

  return 0;
}

// http://www.cplusplus.com/forum/general/21018/
void ReadMatrix(vector<double> &mat, string &filename, unsigned int &n)
{
  ifstream file;
  file.open(filename.c_str(), ios::in | ios::binary);
  double temp;
  for(int i = 0; i < n; ++i)
    {
      file.read((char*)&temp, sizeof(double));
      mat[i] = temp;
    }
  return;
}

void ReadVector(vector<double> &vec, string &filename)
{
  ifstream vecFile(filename.c_str(), ios::in);
  string str = "";
  getline(vecFile, str);
  while(getline(vecFile, str))
    {
      istringstream ss(str);
      double val;
      ss >> val;
      vec.push_back(val);
    }
}

double CompareResult(vector<double> &x1, vector<double> &x2)
{
  assert(x1.size() == x2.size());

  vector<double> result(x1.size(), 0.0);
  
  for(int i = 0; i < result.size(); ++i)
    {
      result[i] = x2[i] - x1[i];
    }

  return (VecNorm(result)/VecNorm(x2));
}

void FullMatVec(vector<double> &mat, unsigned int &n, unsigned int &m, vector<double> &vec, vector<double> &res)
{
  res.resize(vec.size());
  for(int i = 0; i < res.size(); ++i)
    res[i] = 0.0;
  
  for(int i = 0; i < n; ++i)
    {
      for(int j = 0; j < m; ++j)
	{
	  res[i] += mat[i * m + j] * vec[j];
	}
    }
}
