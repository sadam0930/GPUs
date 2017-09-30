/*
 *  Please write your name and net ID below
 *  
 *  Last name: Adam
 *  First name: Steven
 *  Net ID: sna219
 * 
 */


/* 
 * This file contains the code for doing the heat distribution problem. 
 * You do not need to modify anything except starting  gpu_heat_dist() at the bottom
 * of this file.
 * In gpu_heat_dist() you can organize your data structure and the call to your
 * kernel(s) that you need to write too. 
 * 
 * You compile with:
 * 		nvcc -o heatdist heatdist.cu   
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

/*****************************************************************/

// Function declarations: Feel free to add any functions you want.
void  seq_heat_dist(float *, unsigned int, unsigned int);
void  gpu_heat_dist(float *, unsigned int, unsigned int);


/*****************************************************************/

int main(int argc, char * argv[])
{
  unsigned int N; /* Dimention of NxN matrix */
  int type_of_device = 0; // CPU or GPU
  int iterations = 0;
  int i;
  
  /* The 2D array of points will be treated as 1D array of NxN elements */
  float * playground; 
  
  // to measure time taken by a specific part of the code 
  double time_taken;
  clock_t start, end;
  
  if(argc != 4)
  {
    fprintf(stderr, "usage: heatdist num  iterations  who\n");
    fprintf(stderr, "num = dimension of the square matrix (50 and up)\n");
    fprintf(stderr, "iterations = number of iterations till stopping (1 and up)\n");
    fprintf(stderr, "who = 0: sequential code on CPU, 1: GPU execution\n");
    exit(1);
  }
  
  type_of_device = atoi(argv[3]);
  N = (unsigned int) atoi(argv[1]);
  iterations = (unsigned int) atoi(argv[2]);
 
  
  /* Dynamically allocate NxN array of floats */
  playground = (float *)calloc(N*N, sizeof(float));
  if( !playground )
  {
   fprintf(stderr, " Cannot allocate the %u x %u array\n", N, N);
   exit(1);
  }
  
  /* Initialize it: calloc already initalized everything to 0 */
  // Edge elements to 80F
  for(i = 0; i < N; i++)
    playground[index(0,i,N)] = 80;
    
  for(i = 0; i < N; i++)
    playground[index(i,0,N)] = 80;
  
  for(i = 0; i < N; i++)
    playground[index(i,N-1, N)] = 80;
  
  for(i = 0; i < N; i++)
    playground[index(N-1,i,N)] = 80;
  
  // from (0,10) to (0,30) inclusive are 150F
  for(i = 10; i <= 30; i++)
    playground[index(i,0,N)] = 150;
  
  
  if( !type_of_device ) // The CPU sequential version
  {  
    start = clock();
    seq_heat_dist(playground, N, iterations);
    end = clock();
  }
  else  // The GPU version
  {
     start = clock();
     gpu_heat_dist(playground, N, iterations); 
     end = clock();    
  }
  
  
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  
  printf("Time taken for %s is %lf\n", type_of_device == 0? "CPU" : "GPU", time_taken);
  
  free(playground);
  
  return 0;

}


/*****************  The CPU sequential version (DO NOT CHANGE THAT) **************/
void  seq_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{
  // Loop indices
  int i, j, k;
  int upper = N-1;
  
  // number of bytes to be copied between array temp and array playground
  unsigned int num_bytes = 0;
  
  float * temp; 
  /* Dynamically allocate another array for temp values */
  /* Dynamically allocate NxN array of floats */
  temp = (float *)calloc(N*N, sizeof(float));
  if( !temp )
  {
   fprintf(stderr, " Cannot allocate temp %u x %u array\n", N, N);
   exit(1);
  }
  
  num_bytes = N*N*sizeof(float);
  
  /* Copy initial array in temp */
  memcpy((void *)temp, (void *) playground, num_bytes);
  
  for( k = 0; k < iterations; k++)
  {
    /* Calculate new values and store them in temp */
    for(i = 1; i < upper; i++)
      for(j = 1; j < upper; j++)
	temp[index(i,j,N)] = (playground[index(i-1,j,N)] + 
	                      playground[index(i+1,j,N)] + 
			      playground[index(i,j-1,N)] + 
			      playground[index(i,j+1,N)])/4.0;
  
			      
   			      
    /* Move new values into old values */ 
    memcpy((void *)playground, (void *) temp, num_bytes);
  }
  
}

/***************** The GPU version: Write your code here *********************/
__global__ void compute_heat(float * d_temp, float * d_playground, unsigned int N)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row > 0 && row < N-1 && col > 0 && col < N-1) {
    d_temp[index(row,col,N)] = (d_playground[index(row-1,col,N)] + 
                                d_playground[index(row+1,col,N)] + 
                                d_playground[index(row,col-1,N)] + 
                                d_playground[index(row,col+1,N)])/4.0;
  }
}

/* This function can call one or more kenels *********************************/
void  gpu_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{
  int k;
  //number of bytes to be copied between array temp and array playground
  unsigned int num_bytes = N*N*sizeof(float);

  //allocate memory to device
  float * d_temp;
  float * d_playground;

  cudaMalloc((void **)&d_temp, num_bytes);
  cudaMalloc((void **)&d_playground, num_bytes);

  //copy array from host to device
  cudaMemcpy(d_playground, playground, num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_temp, d_playground, num_bytes, cudaMemcpyDeviceToDevice); //data locality

  //divide problem into 2d blocks in 2d grid
  dim3 blockDim(10, 10); //problem size is multiples of 100
  dim3 gridDim(ceil(N/10), ceil(N/10));

  //calulations performed in device
  for(k=0; k < iterations; k++) {
    compute_heat<<<gridDim, blockDim>>>(d_temp, d_playground, N);
    //copy results in device from temp to playground
    cudaMemcpy(d_playground, d_temp, num_bytes, cudaMemcpyDeviceToDevice);
  }

  //copy array from device to host
  cudaMemcpy(playground, d_temp, num_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_playground);
  cudaFree(d_temp);
}

