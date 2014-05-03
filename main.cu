#include <iostream>
#include <time.h>
#include <stdlib.h>

#include "Matrix.cuh"
#include "Vector.cuh"

#define CPU 1
#define GPU 0

#define MINI_NUM_TESTS 16

#define CHECKBOARD_BLOCK_MAX_WIDTH  1
#define CHECKBOARD_BLOCK_MAX_HEIGHT 256

/*
	Serial implementation of Matrix-Vector product
	for CPU
	
	TODO check: m.getWidth() = v.getSize()
*/
Vector productMatrixVectorCPU(const Matrix &m, const Vector &v) {
	Vector r(m.getHeight(), CPU);
	
	unsigned int id(0);
	for (unsigned int i(0) ; i != m.getHeight() ; i++) {
		r[i] = 0;
		for (unsigned int j(0) ; j != m.getWidth() ; j++) {
			r[i] += m.get(id) * v[j]; //r[i] += m[i][j] * v[j];
			id++;
		}
	}
	return r;
}

__global__ void productMatrixVectorGPU_naive_kernel(const Matrix d_m, const Vector d_v, Vector d_r) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i >= d_m.getHeight() || j >= d_m.getWidth())
		return;
	
	atomicAdd(&d_r[i], d_m.get(i, j) * d_v[j]);
}

/*
	Parallel implementation of Matrix-Vector product
	for GPU
	
	Naïve implementation:
		Checkboard partitioning
	
	TODO check: m.getWidth() = v.getSize()
*/
Vector productMatrixVectorGPU_naive(const Matrix &h_m, const Vector &h_v) throw (cudaError_t) {
	Matrix d_m(h_m, GPU);
	Vector d_v(h_v, GPU);
	
	Vector d_r(h_m.getHeight(), GPU);
	d_r.memsetZero();
	
	const dim3 num_threads(CHECKBOARD_BLOCK_MAX_HEIGHT, CHECKBOARD_BLOCK_MAX_WIDTH, 1);
	const dim3 num_blocks((d_m.getHeight() + CHECKBOARD_BLOCK_MAX_HEIGHT -1)/CHECKBOARD_BLOCK_MAX_HEIGHT, (d_m.getWidth() + CHECKBOARD_BLOCK_MAX_WIDTH -1)/CHECKBOARD_BLOCK_MAX_WIDTH, 1);
	productMatrixVectorGPU_naive_kernel<<<num_blocks, num_threads>>>(d_m, d_v, d_r);
	cudaThreadSynchronize(); // block until the device is finished
	
	// check for error
	cudaError_t error = cudaGetLastError();
  	if(error != cudaSuccess) {
    		std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
		throw error;
	}
	
	Vector h_r(d_r, CPU);
	return h_r;
}

__global__ void productMatrixVectorGPU_shared_kernel(const Matrix d_m, const Vector d_v, Vector d_r) {
	extern __shared__ float block_result[]; // need: blockDim.x + blockDim.y
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i >= d_m.getHeight() || j >= d_m.getWidth())
		return;
	
	block_result[threadIdx.x] = 0;
	block_result[blockDim.x + threadIdx.y] = d_v[j];
	__syncthreads();
	
	atomicAdd(&block_result[threadIdx.x], d_m.get(i, j) * block_result[blockDim.x + threadIdx.y]);
	__syncthreads();
	
	if (threadIdx.y == 0)
		atomicAdd(&d_r[i], block_result[threadIdx.x]);
}

/*
	Parallel implementation of Matrix-Vector product
	for GPU
	
	Shared implementation:
		Checkboard partitioning
		Shared memory
	
	TODO check: m.getWidth() = v.getSize()
*/
Vector productMatrixVectorGPU_shared(const Matrix &h_m, const Vector &h_v) throw (cudaError_t) {
	Matrix d_m(h_m, GPU);
	Vector d_v(h_v, GPU);
	
	Vector d_r(h_m.getHeight(), GPU);
	d_r.memsetZero();
	
	const dim3 num_threads(CHECKBOARD_BLOCK_MAX_HEIGHT, CHECKBOARD_BLOCK_MAX_WIDTH, 1);
	const dim3 num_blocks((d_m.getHeight() + CHECKBOARD_BLOCK_MAX_HEIGHT -1)/CHECKBOARD_BLOCK_MAX_HEIGHT, (d_m.getWidth() + CHECKBOARD_BLOCK_MAX_WIDTH -1)/CHECKBOARD_BLOCK_MAX_WIDTH, 1);
	const unsigned int SHARED_MEMORY_SIZE((CHECKBOARD_BLOCK_MAX_HEIGHT + CHECKBOARD_BLOCK_MAX_WIDTH) * sizeof(float));
	productMatrixVectorGPU_shared_kernel<<<num_blocks, num_threads, SHARED_MEMORY_SIZE>>>(d_m, d_v, d_r);
	cudaThreadSynchronize(); // block until the device is finished
	
	// check for error
	cudaError_t error = cudaGetLastError();
  	if(error != cudaSuccess) {
    		std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
  		throw error;
  	}
	
	Vector h_r(d_r, CPU);
	return h_r;
}

void randMatrix(Matrix &h_m) {
	for (unsigned int i(0) ; i != h_m.getHeight() ; i++) {
		for (unsigned int j(0) ; j != h_m.getWidth() ; j++) {
			h_m[i][j] = rand() % 10;
		}
	}
}

void randVector(Vector &h_v) {
	for (unsigned int j(0) ; j != h_v.getSize() ; j++) {
		h_v[j] = rand() % 10;
	}
}

bool checkResultsProductMatrixVector(Vector(*model)(const Matrix &, const Vector &), Vector(*func)(const Matrix &, const Vector &)) {
	std::cout << "CHECKS for Matrix-Vector product:" << std::endl;
	
	for (unsigned int i(0) ; i!= 100 ; i++) {
		const unsigned int width(1 + rand()%2048);
		const unsigned int height(1 + rand()%2048);
		
		Matrix h_m(width, height, CPU);
		randMatrix(h_m);
		
		Vector h_v(width, CPU);
		randVector(h_v);
		
		try {
			Vector h_model = model(h_m, h_v);
			Vector h_func = func(h_m, h_v);
			
			if (h_model.getSize() != height || h_func.getSize() != height) {
				std::cout << "INCORRECT RESULTS (size)" << std::endl;
				return false;
			}
			
			for (unsigned int j(0) ; j != height ; j++) {
				if (h_model[j] != h_func[j]) {
					std::cout << "INCORRECT RESULTS (value): " << h_model[j] << "!=" << h_func[j] << " for " << width << "x" << height << std::endl;
					return false;
				}
			}
		} catch(cudaError_t &e) {
			std::cout << "INCORRECT RESULTS (cuda exception)" << std::endl;
			return false;
		}
	}
	
	std::cout << "NO ERROR DETECTED" << std::endl;
	return true;
}

void measureProductMatrixVector(Vector(*func)(const Matrix &, const Vector &)) {
	std::cout << "NEW MEASUREMENTS for Matrix-Vector product:" << std::endl;
	
	unsigned long int LAST_NUM_TESTS(MINI_NUM_TESTS);
	for (unsigned int size(2) ; size < 2049 ; size <<= 1) {
		Matrix h_m(size, size, CPU);
		randMatrix(h_m);
		
		Vector h_v(size, CPU);
		randVector(h_v);
		
		double time_required(0);
		unsigned long int NUM_TESTS(LAST_NUM_TESTS >> 2);
		if (NUM_TESTS < MINI_NUM_TESTS)
			NUM_TESTS = MINI_NUM_TESTS;
		
		while (time_required < 1000.) {
			clock_t t_chrono;
			t_chrono = clock();
			for (unsigned long int i(0) ; i != NUM_TESTS ; i++) {
				try {		
					func(h_m, h_v);
				} catch(cudaError_t &e) {
					return;
				}
			}
			t_chrono = clock() - t_chrono;
			time_required = 1000. * ((double) t_chrono)/CLOCKS_PER_SEC;
			
			if (time_required < 1000. && NUM_TESTS < (NUM_TESTS<<1))
				NUM_TESTS <<= 1;
			else
				break;
		}
		std::cout << "Size: " << size << "\tTests: " << NUM_TESTS << "\tTime: " << time_required << "ms" << std::endl;
		LAST_NUM_TESTS = NUM_TESTS;
	}
}

int main(int argc, char **argv) {
	srand(time(NULL));
	
	checkResultsProductMatrixVector(productMatrixVectorCPU, productMatrixVectorGPU_naive);
	checkResultsProductMatrixVector(productMatrixVectorCPU, productMatrixVectorGPU_shared);
	
	measureProductMatrixVector(productMatrixVectorCPU);
	measureProductMatrixVector(productMatrixVectorGPU_naive);
	measureProductMatrixVector(productMatrixVectorGPU_shared);
	
	return 0;
}

