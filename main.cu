#include <iostream>
#include <time.h>
#include <stdlib.h>

#include "Matrix.cuh"
#include "Vector.cuh"

#define CPU 1
#define GPU 0

#define NUM_TESTS 16

#define CHECKBOARD_BLOCK_MAX_WIDTH  1
#define CHECKBOARD_BLOCK_MAX_HEIGHT 256

#define MAX_NUM_THREADS 256

/*
	Serial implementation of Matrix-Vector product
	for CPU
	
	TODO check: m.getWidth() = v.getSize()
*/
Vector productMatrixVectorCPU(const Matrix &m, const Vector &v) {
	Vector r(m.getHeight(), CPU);
	
	r.memsetZero();
	
	unsigned int id(0);
	for (unsigned int j(0) ; j != m.getWidth() ; j++) {
		for (unsigned int i(0) ; i != m.getHeight() ; i++) {
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
	// Each block deals with only one vector element
	__shared__ float vector_element;
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y;
	
	if (i >= d_m.getHeight() || j >= d_m.getWidth())
		return;
	
	vector_element = d_v[j];
	__syncthreads();
	
	atomicAdd(&d_r[i], d_m.get(i, j) * vector_element);
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
	
	const dim3 num_threads(MAX_NUM_THREADS, 1, 1);
	const dim3 num_blocks((d_m.getHeight() + MAX_NUM_THREADS -1)/MAX_NUM_THREADS, d_m.getWidth(), 1);
	productMatrixVectorGPU_shared_kernel<<<num_blocks, num_threads>>>(d_m, d_v, d_r);
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

__global__ void productMatrixVectorGPU_row_kernel(const Matrix d_m, const Vector d_v, Vector d_r) {
	// Each thread deals with only one vector element of the result
	extern __shared__ float d_v_copy[];
	
	unsigned int w = d_v.getSize();
	unsigned int start_to_fill = threadIdx.x * w / blockDim.x;
	unsigned int end_to_fill = (threadIdx.x +1) * w / blockDim.x;
	
	for (unsigned int j(start_to_fill) ; j != end_to_fill ; j++) {
		d_v_copy[j] = d_v[j];
	}
	__syncthreads();
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i >= d_m.getHeight())
		return;
	
	float result_vector_element = 0;
	for (unsigned int j(0) ; j != w ; j++) {
		result_vector_element += d_m.get(i, j) * d_v_copy[j];
	}
	
	d_r[i] = result_vector_element;
}

/*
	Parallel implementation of Matrix-Vector product
	for GPU
	
	Shared implementation:
		Checkboard partitioning
		Shared memory
	
	TODO check: m.getWidth() = v.getSize()
*/
Vector productMatrixVectorGPU_row(const Matrix &h_m, const Vector &h_v) throw (cudaError_t) {
	Matrix d_m(h_m, GPU);
	Vector d_v(h_v, GPU);
	Vector d_r(h_m.getHeight(), GPU);
	
	const dim3 num_threads(MAX_NUM_THREADS, 1, 1);
	const dim3 num_blocks((d_m.getHeight() + MAX_NUM_THREADS -1)/MAX_NUM_THREADS, 1, 1);
	productMatrixVectorGPU_row_kernel<<<num_blocks, num_threads, d_v.getSize() * sizeof(float)>>>(d_m, d_v, d_r);
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
	std::cout << "NEW MEASUREMENTS for Matrix-Vector product: " << std::endl;
	
	for (unsigned int size(2) ; size < 10000 ; size <<= 1) {
		Matrix h_m(size, size, CPU);
		randMatrix(h_m);
		
		Vector h_v(size, CPU);
		randVector(h_v);
			
		float time_required(0);
		
		for (unsigned long int i(0) ; i != NUM_TESTS ; i++) {
			float time(0);
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);
			try {		
				func(h_m, h_v);
			} catch(cudaError_t &e) {
				return;
			}
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			time_required += time;
		}
		time_required /= NUM_TESTS;
		std::cout << "Size: " << size << "\tTests: " << NUM_TESTS << "\tTime: " << time_required << "ms" << std::endl;
	}
}

int main(int argc, char **argv) {
	srand(time(NULL));
	
	checkResultsProductMatrixVector(productMatrixVectorCPU, productMatrixVectorGPU_naive);
	checkResultsProductMatrixVector(productMatrixVectorCPU, productMatrixVectorGPU_shared);
	checkResultsProductMatrixVector(productMatrixVectorCPU, productMatrixVectorGPU_row);
	
	measureProductMatrixVector(productMatrixVectorCPU);
	measureProductMatrixVector(productMatrixVectorGPU_naive);
	measureProductMatrixVector(productMatrixVectorGPU_shared);
	measureProductMatrixVector(productMatrixVectorGPU_row);
	
	return 0;
}

