#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <map>
#include <string>
#include "mgard/mgard_api.h"

#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_RESET "\x1b[0m"


using namespace std;
void print_usage_message(char *argv[], FILE *fp) 
{
	fprintf(
		fp, 
		"Usage: %s [input file] [tolerance] [s] [1st dim.] [2nd dim.] [3rd. dim]\n",
 		argv[0]
	);
}

int main(int argc, char *argv[])
{
	if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) 
	{
 		print_usage_message(argv, stdout);
 		return 0;
 	}
	
	cout << "\n=============================== MGARD =================================\n";
	// File
	char *infile;
	infile = argv[1];
	cout << "\tInput file: " <<infile << endl;

	// L-infinity tolerance and smoothness
	double tol, s;
	tol = atof(argv[2]);
	cout << "\tAbsolution L-infinity error bound: " << tol << endl;	
	s = atof(argv[3]);	
	cout << "\tSmoothness: " << s << endl;	

	// Shape
	vector<size_t> shape;
	cout << "\tShape: ( ";
	for (size_t d = 0; d < 3; d++) 
	{
 		shape.push_back(atoi(argv[4 + d]));
 		cout << shape[shape.size() - 1] << " ";
	}
	cout << ")\n";

	// Check file data size and load data
	cout << "\n\tLoading data...";
	FILE *pFile;
	pFile = fopen(infile, "rb");
	if (pFile == NULL)
	{
		fputs(ANSI_RED "File error" ANSI_RESET "\n", stderr);
		exit(1);
	}	
	fseek(pFile, 0, SEEK_END);
	size_t lSize = ftell(pFile);
	rewind(pFile);
	vector<double> in_buff;
	size_t num_dpts = lSize / sizeof(unsigned short int);
	unsigned short int num;
	for (size_t i = 0; i < num_dpts; i++)
	{
		fread(&num, sizeof(unsigned short int), 1, pFile);
		in_buff.push_back((double)num);
	}
	cout << "Done\n";
	cout << "\tFile size (bytes): " << lSize << endl;
	cout << "\tNumber of data points: " << in_buff.size() << endl;
	fclose(pFile);	

	// Initialization
	mgard_cuda::Handle<3, double> handle(shape);
	mgard_cuda::Array<3, double> in_array(shape);
	in_array.loadData(&in_buff[0]);

	// Compression
	cout << "\n\tCompressing...";
	mgard_cuda::Array<1, unsigned char> compressed_array = mgard_cuda::compress(handle, in_array, mgard_cuda::ABS, tol, s);
	size_t out_size = compressed_array.getShape()[0];
	cout << "Done\n";
	cout << "\tOutput size (bytes): " << out_size << endl;
	cout << "\tCompression ratio: " << (double)out_size / (double)lSize << endl;
	
	// Decompression
	cout << "\n\tDecompressing...";
	mgard_cuda::Array<3, double> out_array = mgard_cuda::decompress(handle, compressed_array);	
	double* mgard_out_buff = new double[num_dpts];
	memcpy(mgard_out_buff, out_array.getDataHost(), num_dpts * sizeof(double));
	cout << "Done\n";

	// Evaluation
	double error_L_inf_norm = 0;
	double sum = 0;
	double nonzero_sum = 0;
	size_t nonzero_count = 0;
	for (int i = 0; i < num_dpts; ++i) 
	{
		double temp = fabs(in_buff[i] - mgard_out_buff[i]);
		if (temp > error_L_inf_norm)
		{
      		error_L_inf_norm = temp;
		}
		sum += temp * temp;
		if (in_buff[i] > 64)
		{
			nonzero_sum += temp * temp;
			nonzero_count++;
		}
	}
	delete mgard_out_buff;
	double absolute_L_inf_error = error_L_inf_norm;
	double mse = sum / num_dpts;
	double nonzero_mse = nonzero_sum / nonzero_count;
	cout << "\tMSE: " << mse << endl;
	cout << "\tNon-zero count: " << nonzero_count << endl;
	cout << "\tNon-zero MSE: " << nonzero_mse << endl;

	printf("\n\tAbs. L-infinity error bound: %10.5E \n", tol);
	printf("\tAbs. L-infinity error: %10.5E \n\n", absolute_L_inf_error);
	if (absolute_L_inf_error < tol) 
	{
		printf(ANSI_GREEN "\tSUCCESS: Error tolerance met!" ANSI_RESET "\n");
	} 
	else 
	{
		printf(ANSI_RED "\tFAILURE: Error tolerance NOT met!" ANSI_RESET "\n");
	}
	cout << "=======================================================================\n\n";
	return 0;
}

