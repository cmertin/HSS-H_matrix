#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

struct SubMatrix
{
  vector<unsigned int> info; // start_i, n, k, m
  vector<double> Y; // Uk * Sk
  vector<double> Z; // Vk^T
};

bool FileExists(string &filename);
void ReadHMat(string &filename, vector<SubMatrix> &HMat);
void ReadVec(string &filename, vector<double> &vec);
void MatVec(vector<SubMatrix> &HMat, vector<double> &vec, vector<double> &result);
void OutputFile(string &filename, vector<double> &result);

int main(int argc, char *argv[])
{
  if(argc != 4)
    {
      cout << "Usage: ./matvec Hmat_file.ext vec_file.ext output_file.ext" << endl;
      return -1;
    }

  vector<SubMatrix> HMat;
  vector<double> vec;
  vector<double> result;

  string HMat_file = argv[1];
  string vec_file = argv[2];
  string out_file = argv[3];

  if(FileExists(HMat_file) == false)
    {
      cout << "ERROR: File \"" + HMat_file + "\" does not exist. Exiting..." << endl;
      return -1;
    }

  if(FileExists(vec_file) == false)
    {
      cout << "ERROR: File \"" + vec_file + "\" does not exist. Exiting..." << endl;
      return -1;
    }

  ReadHMat(HMat_file, HMat);
  cout << '\t' << "Finished reading " << HMat_file << endl;
  ReadVec(vec_file, vec);
  cout << '\t' << "Finished reading " << vec_file << endl;
  MatVec(HMat, vec, result);
  cout << '\t' << "Finished MatVec " << endl;
  OutputFile(out_file, result);
  cout << '\t' << "Wrote result to " << out_file << endl;

  return 0;
}

bool FileExists(string &filename)
{
  ifstream file(filename.c_str());
  return file.good();
}


void ReadHMat(string &filename, vector<SubMatrix> &HMat)
{
  ifstream file;
  file.open(filename.c_str());
  string line;
  unsigned int num_submat;
  getline(file, line);
  getline(file, line);
  num_submat = stoi(line);
  for(int submat = 0; submat < num_submat; ++submat)
    {
      SubMatrix A;
      size_t pos = 0;
      string substr;
      // Reads in the info for the sub matrix
      // start_i, n, k, m
      for(int i = 0; i < 4; ++i)
	{
	  getline(file, line);
	  A.info.push_back(stoi(line));
	}
      // Reads in Yk
      for(int y = 0; y < A.info[1]; ++y)
	     {
	       for(int k = 0; k < A.info[2]; ++k)
		 {
		   getline(file, line);
		   double val = stod(line);
		   A.Y.push_back(val);
		 }
	      }
      // Reads in Zk
      for(int z = 0; z < A.info[3]; ++z)
	     {
	       for(int k = 0; k < A.info[2]; ++k)
		 {
		   getline(file, line);
		   double val = stod(line);
		   A.Z.push_back(val);
		 }
	     }
      HMat.push_back(A);
    }
  file.close();
  return;
}

void ReadVec(string &filename, vector<double> &vec)
{
  ifstream file;
  file.open(filename.c_str());
  string line;
  getline(file, line);
  while(getline(file, line))
    {
      double val = stod(line);
      vec.push_back(val);
    }
  file.close();
  return;
}

void MatVec(vector<SubMatrix> &HMat, vector<double> &vec, vector<double> &result)
{
  result.resize(vec.size(), 0.0);
  for(int submat = 0; submat < HMat.size(); ++submat)
    {
      unsigned int start_i = HMat[submat].info[0];
      unsigned int n = HMat[submat].info[1];
      unsigned int k = HMat[submat].info[2];
      unsigned int m = HMat[submat].info[3];
      for(int i = 0; i < n; ++i)
	     {
	        unsigned int x_index = start_i + i;
	        for(int j = 0; j < m; ++j)
	         {
	            for(int k_ = 0; k_ < k; ++k_)
		            {
		              unsigned int y_index = i * k + k_;
		              unsigned int z_index = j * k + k_;
		              result[x_index] += HMat[submat].Y[y_index] * HMat[submat].Z[z_index] * vec[x_index];
		            }
	        }
	    }
    }
  return;
}

void OutputFile(string &filename, vector<double> &result)
{
  ofstream output(filename.c_str());
  for(int i = 0; i < result.size(); ++i)
    {
      output << result[i] << endl;
    }
  output.close();
  return;
}
