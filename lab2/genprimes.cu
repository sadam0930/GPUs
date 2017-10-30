/*
 *  Please write your name and net ID below
 *  
 *  Last name: Adam
 *  First name: Steven
 *  Net ID: sna219
 * 
 */


/* 
 * Compile with:
 *      nvcc -o genprimes genprimes.cu   
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

__global__ static void init(char* primes) {
   primes[0] = 0;
   primes[1] = 0;
}

__global__ static void removeEvens(char* primes, int N) {
   int index = blockIdx.x * blockDim.x *2 + threadIdx.x + threadIdx.x + 4;
   if (index <= N)
      primes[index] = 0;
}

__global__ static void removeNonPrimes(char* primes, int N, const int limit) {
   // get the starting index, remove odds starting at 3
   // block 0: 3,   5,  7,  9, 11, 13, ...,  65
   // block 1: 67, 69, 71, 73, 75, 77, ..., 129
   int index = blockIdx.x * blockDim.x *2 + threadIdx.x + threadIdx.x + 3;

   // make sure index won't go out of bounds, also don't start the execution
   // on numbers that are already composite
   if (index <= limit && primes[index] == 1) {
      for (int j=index*2; j <= N; j+=index) {
         primes[j] = 0;
      }
   }
}

// query the Device and decide on the block size
__host__ int checkDevice() {
   int devID = 0; // the default device ID
   cudaDeviceProp deviceProp;
   cudaGetDeviceProperties(&deviceProp, devID);
   return (deviceProp.major < 2) ? 16 : 32;
}

int main(int argc, char * argv[]) {
	unsigned int N;
	N = (unsigned int) atoi(argv[1]);	

	// create array of chars; 1 is prime
	// we will set non primes to 0
	char* primes = new char[N+1];

	for(int j=2; j <= N; j++) {
		primes[j] = 1;
	}

	// allocate device memory
	char* d_primes = NULL;
	int sizePrimes = sizeof(char) * N;
	int limit = floor((N+1)/2); //only need to compute up to this point

	cudaMalloc(&d_primes, sizePrimes);
	cudaMemset(d_primes, 1, sizePrimes);

	int blocksize = checkDevice();
	if (blocksize == EXIT_FAILURE)
		return 1;

	dim3 dimBlock(blocksize);
	dim3 dimGrid(ceil((limit + dimBlock.x)/(double) dimBlock.x) / (double) 2);
	dim3 dimGridEven(ceil((N + dimBlock.x)/(double) dimBlock.x) / (double) 2);

	init<<<1,1>>>(d_primes); //init shared memory cells in single GPU thread
	removeEvens<<<dimGridEven, dimBlock>>>(d_primes, N);
	removeNonPrimes<<<dimGrid, dimBlock>>>(d_primes, N, limit);	

	cudaMemcpy(primes, d_primes, sizePrimes, cudaMemcpyDeviceToHost);
	cudaFree(d_primes);

	//print output
	std::ofstream f;
	std::string filename = std::to_string(N) + ".txt";
	f.open (filename);
	//skip 0 and 1 are not primes
	for(int p=2; p <= N; p++) {
		if(primes[p] == 1) {
			f << std::to_string(p) << " ";
		}
	}
	f.close();
	return 0;
}