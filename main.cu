#include <string.h>
#include <iostream>

#include <time.h>
#include <stdlib.h>

#include "Matrix.cuh"
#include "Vector.cuh"

#define WIDTH 2048
#define HEIGHT 2048
#define NUM_TESTS 1000

#define CPU 1
#define GPU 0

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
Vector productMatrixVectorGPU_naive(const Matrix &h_m, const Vector &h_v) {
	Matrix d_m(h_m, GPU);
	Vector d_v(h_v, GPU);
	
	Vector d_r(h_m.getHeight(), GPU);
	d_r.memsetZero();
	
	const dim3 num_threads(CHECKBOARD_BLOCK_MAX_HEIGHT, CHECKBOARD_BLOCK_MAX_WIDTH, 1);
	const dim3 num_blocks((d_m.getHeight() + CHECKBOARD_BLOCK_MAX_HEIGHT -1)/CHECKBOARD_BLOCK_MAX_HEIGHT, (d_m.getWidth() + CHECKBOARD_BLOCK_MAX_WIDTH -1)/CHECKBOARD_BLOCK_MAX_WIDTH, 1);
	productMatrixVectorGPU_naive_kernel<<<num_blocks, num_threads>>>(d_m, d_v, d_r);
	
	Vector h_r(d_r, CPU);
	return h_r;
}

__global__ void productMatrixVectorGPU_shared_kernel(const Matrix d_m, const Vector d_v, Vector d_r) {
	extern __shared__ float block_result[]; // best value: blockDim.x + blockDim.y
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	block_result[threadIdx.x] = 0;
	block_result[blockDim.x + threadIdx.y] = d_v[j];
	__syncthreads();
	
	if (i >= d_m.getHeight() || j >= d_m.getWidth())
		return;
	
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
Vector productMatrixVectorGPU_shared(const Matrix &h_m, const Vector &h_v) {
	Matrix d_m(h_m, GPU);
	Vector d_v(h_v, GPU);
	
	Vector d_r(h_m.getHeight(), GPU);
	d_r.memsetZero();
	
	const dim3 num_threads(CHECKBOARD_BLOCK_MAX_HEIGHT, CHECKBOARD_BLOCK_MAX_WIDTH, 1);
	const dim3 num_blocks((d_m.getHeight() + CHECKBOARD_BLOCK_MAX_HEIGHT -1)/CHECKBOARD_BLOCK_MAX_HEIGHT, (d_m.getWidth() + CHECKBOARD_BLOCK_MAX_WIDTH -1)/CHECKBOARD_BLOCK_MAX_WIDTH, 1);
	productMatrixVectorGPU_shared_kernel<<<num_blocks, num_threads, CHECKBOARD_BLOCK_MAX_HEIGHT + CHECKBOARD_BLOCK_MAX_WIDTH>>>(d_m, d_v, d_r);
	
	Vector h_r(d_r, CPU);
	return h_r;
}

int main(int argc, char **argv) {
	srand(time(NULL));
	
	Matrix h_m(WIDTH, HEIGHT, CPU);
	for (unsigned int i(0) ; i != HEIGHT ; i++) {
		for (unsigned int j(0) ; j != WIDTH ; j++) {
			h_m[i][j] = rand() % 10;
		}
	}
	//h_m.print();
	
	Vector h_v(WIDTH, CPU);
	for (unsigned int j(0) ; j != WIDTH ; j++) {
		h_v[j] = rand() % 10;
	}
	//h_v.print();
	
	clock_t t_chrono;
	t_chrono = clock();
	for (int i(0) ; i != NUM_TESTS ; i++) {
		productMatrixVectorCPU(h_m, h_v);
	}
	t_chrono = clock() - t_chrono;
	std::cout << "Measured time for <CPU>: " << ((float) t_chrono)/CLOCKS_PER_SEC << "s" << std::endl;
	
	t_chrono = clock();
	for (int i(0) ; i != NUM_TESTS ; i++) {
		productMatrixVectorGPU_naive(h_m, h_v);
	}
	t_chrono = clock() - t_chrono;
	std::cout << "Measured time for <GPU naive>: " << ((float) t_chrono)/CLOCKS_PER_SEC << "s" << std::endl;
	
	t_chrono = clock();
	for (int i(0) ; i != NUM_TESTS ; i++) {
		productMatrixVectorGPU_shared(h_m, h_v);
	}
	t_chrono = clock() - t_chrono;
	std::cout << "Measured time for <GPU shared>: " << ((float) t_chrono)/CLOCKS_PER_SEC << "s" << std::endl;
	
	return 0;
}

