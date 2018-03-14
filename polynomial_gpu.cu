
#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


int THREADS_PER_BLOCK;
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}


__global__ void polynomial(float* array, float* poly, int degree, int n) {
  
  int index = threadIdx.x+ blockIdx.x* blockDim.x;
  if(index < n){
  float out = 0.;
  float xtothepowerof = 1.;
  for (int i=0; i<=degree; i++) {
    out += xtothepowerof * poly[i];
    xtothepowerof *= array[index];
  }
  array[index] = out;
  }
}

void polynomial_expansion (float* poly, int degree,
			   int n, float* array) {
 float *d_poly, *d_array;
 
 int size_array = n * sizeof(float);
 int size_poly = (degree+1) * sizeof(float);
 

//std::cout<<1<<std::endl;
 //Allocating memory on the GPU.
  HANDLE_ERROR(cudaMalloc(&d_array, size_array));  //std::cout<<2<<std::endl;

  HANDLE_ERROR(cudaMalloc(&d_poly, size_poly));  //std::cout<<3<<std::endl;


 //Copying variables from cpu to gpu.
 HANDLE_ERROR(cudaMemcpy(d_poly, poly, size_poly, cudaMemcpyHostToDevice)); 
 HANDLE_ERROR(cudaMemcpy(d_array, array, size_array, cudaMemcpyHostToDevice)); 


  // Launch add() kernel on GPU
  polynomial<<<(n+ THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_array,d_poly, degree, n);

// Copy result back to host

  HANDLE_ERROR(cudaMemcpy(array, d_array, size_array, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(poly, d_poly, size_poly, cudaMemcpyDeviceToHost));

 
 //Cleanup
 HANDLE_ERROR(cudaFree(d_array)); 
 HANDLE_ERROR(cudaFree(d_poly)); 

}


int main (int argc, char* argv[]) {
  //TODO: add usage
  
  if (argc < 4) {
     std::cerr<<"usage: "<<argv[0]<<" n degree blocksize"<<std::endl;
     return -1;
  }

  int n = atoi(argv[1]); //TODO: atoi is an unsafe function
  int degree = atoi(argv[2]);
  THREADS_PER_BLOCK= atoi(argv[3]);
  int nbiter = 1;

  float* array = new float[n];
  float* poly = new float[degree+1];
  for (int i=0; i<n; ++i)
    array[i] = 1.;

  for (int i=0; i<degree+1; ++i)
    poly[i] = 1.;

 

  std::chrono::time_point<std::chrono::system_clock> begin, end;
  begin = std::chrono::system_clock::now();

 for (int iter = 0; iter<nbiter; ++iter)
    polynomial_expansion (poly, degree, n, array);



  end = std::chrono::system_clock::now();
  std::chrono::duration<double> totaltime = (end-begin)/nbiter;

  std::cerr<<array[0]<<std::endl;
  std::cout<<n<<" "<<degree<<" "<<totaltime.count()<<std::endl;


  delete[] array;
  delete[] poly;

  return 0;
}
