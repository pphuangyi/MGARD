#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>
#include "mgard/mgard_api.h"

#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_RESET "\x1b[0m"

void print_usage_message(char *argv[], FILE *fp) 
{
	fprintf(
		fp, 
		"Usage: %s [input file] [float precision (either 32 or 64)] "
		"[tolerance] [s] "
		"[number of dimensions] [1st dim.] [2nd dim.] [3rd. dim] ... \n",
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
	
	std::cout << "\n=============================== MGARD =================================\n";
	// File
	char *infile;
	infile = argv[1];
	std::cout << "\tInput file: " <<infile << std::endl;

	// Precision of the data in the file
	size_t precision = (size_t) atoi(argv[2]);
	std::cout << "\tInput file precision: " << precision << std::endl;

	// L-infinity tolerance and smoothness
	int i = 3;	
	double tol, s;
	tol = atof(argv[i++]);
	std::cout << "\tAbsolution L-infinity error bound: " << tol << std::endl;	
	s = atof(argv[i++]);	
	std::cout << "\tSmoothness: " << s << std::endl;	

	// Shape
	size_t D = (size_t) atoi(argv[i++]);
	std::vector<size_t> shape;
	std::cout << "\tShape: " << D << " ( ";
	for (size_t d = 0; d < D; d++) 
	{
 		shape.push_back(atoi(argv[i++]));
 		std::cout << shape[shape.size() - 1] << " ";
	}
	std::cout << ")\n";

	// Check file data size
	FILE *pFile;
	pFile = fopen(infile, "rb");
	if (pFile == NULL)
	{
		fputs("File error\n", stderr);
		exit(1);
	}	
	fseek(pFile, 0, SEEK_END);
	size_t lSize = ftell(pFile);
	rewind(pFile);
	std::cout << "\tFile size (bytes): " << lSize << std::endl;

	// Loading data
	std::cout << "\n\tLoading data ...";
	std::vector<double> in_buff;
	size_t num_dpts; 
	if (precision == 32)
	{
		num_dpts = lSize / sizeof(float);
		float num;
		for (size_t i = 0; i < num_dpts; i++)
		{
			fread(&num, sizeof(float), 1, pFile);
			in_buff.push_back((double)num);
		}
	}
	else if (precision == 64)
	{
		num_dpts = lSize / sizeof(double);
		double num;
		for (size_t i = 0; i < num_dpts; i++)
		{
			fread(&num, sizeof(double), 1, pFile);
			in_buff.push_back(num);
		}
	}
	else
	{
		std::cout << "wrong precision, choose between 32 and 64" << std::endl;
		exit(1);
	}
	std::cout << "Done\n";
	std::cout << "\tnumber of data points in in_buff = " << in_buff.size() << std::endl;
	fclose(pFile);	

	// Initialization
	mgard_cuda::Handle<3, double> handle(shape);
	mgard_cuda::Array<3, double> in_array(shape);
	in_array.loadData(&in_buff[0]);

	// Compression
	std::cout << "\n\tCompressing ...";
	mgard_cuda::Array<1, unsigned char> compressed_array = mgard_cuda::compress(handle, in_array, mgard_cuda::ABS, tol, s);
	size_t out_size = compressed_array.getShape()[0];

	std::cout << "Done\n";
	std::cout << "\toutput size (bytes): " << out_size << std::endl;
	
	// Decompression
	std::cout << "\n\tDecompressing ...";
	mgard_cuda::Array<3, double> out_array = mgard_cuda::decompress(handle, compressed_array);
		
	double* mgard_out_buff = new double[num_dpts];
	memcpy(mgard_out_buff, out_array.getDataHost(), num_dpts * sizeof(double));
	std::cout << "Done\n";

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
	std::cout << "\tMean squared error: " << mse << std::endl;

	std::cout << "\tNon-zero count: " << nonzero_count << std::endl;
	double nonzero_mse = nonzero_sum / nonzero_count;
	std::cout << "\tNon-zero mse: " << nonzero_mse << std::endl;


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
	
	std::cout << "=======================================================================\n\n";
	return 0;
}

