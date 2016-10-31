#include <iostream>
#include <fstream>
#include "hmat.h"

using namespace std;

void ReadMatrix(double mat[], string &filename, unsigned int &n);

int main()
{
  string matFile = "mob_H.bin";
  unsigned int n = 1000;
  unsigned int nm = 3 * n * n;
  double *mat = new double[nm];
  
  ReadMatrix(mat, matFile, nm);
  return 0;
}

// http://www.cplusplus.com/forum/general/21018/
void ReadMatrix(double mat[], string &filename, unsigned int &n)
{
  ifstream file;
  file.open(filename.c_str(), ios::in | ios::binary);
  double temp;
  for(int i = 0; i < n; ++i)
    {
      file.read((char*)&temp, sizeof(double));
      cout << temp << endl;
    }
  return;
}
