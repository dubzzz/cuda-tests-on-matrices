#include <string.h>
#include <iostream>

#include <time.h>
#include <stdlib.h>

#define WIDTH 1024
#define HEIGHT 256
#define NUM_TESTS 1000

#define CHECKBOARD_BLOCK_MAX_WIDTH  16
#define CHECKBOARD_BLOCK_MAX_HEIGHT 16

struct Matrix {
	unsigned int width;
	unsigned int height;
	float *data;
	
	class MatrixRow { // cf. http://stackoverflow.com/questions/3755111/how-do-i-define-a-double-brackets-double-iterator-operator-similar-to-vector-of
		const Matrix &m_;
		const unsigned int x_;
		
		public:
			__device__ __host__ MatrixRow(const Matrix &m, const unsigned int &x) : m_(m), x_(x) {}
			__device__ __host__ float& operator[](const unsigned int &y) const {
				return m_.data[x_ * m_.width + y];
			}
	};
	
	__device__ __host__ MatrixRow operator[](const unsigned int &x) const {
		return MatrixRow(*this, x);
	}
	
	void print() {
		std::cout << "[" << std::endl;
		for (unsigned int i(0) ; i != HEIGHT ; i++) {
			for (unsigned int j(0) ; j != WIDTH ; j++) {
				std::cout << " " << (*this)[i][j];
			}
			std::cout << std::endl;
		}
		std::cout << "]" << std::endl;
	}
};

struct Vector {
	unsigned int size;
	float *data;
	
	__device__ __host__ float& operator[](const unsigned int &x) const {
		return data[x];
	}
	
	void print() {
		std::cout << "[";
		for (unsigned int j(0) ; j != size ; j++) {
			std::cout << " " << (*this)[j];
		}
		std::cout << " ]" << std::endl;
	}
};

/*
	Serial implementation of Matrix-Vector product
	for CPU
	
	TODO check: m.width = v.size
*/
Vector productMatrixVectorCPU(const Matrix &m, const Vector &v) {
	Vector r;
	r.size = m.height;
	r.data = new float[r.size];
	
	for (unsigned int i(0) ; i != m.height ; i++) {
		r[i] = 0;
		for (unsigned int j(0) ; j != m.width ; j++) {
			r[i] += m[i][j] * v[j];
		}
	}
	return r;
}

__global__ void productMatrixVectorGPU_naive_kernel(const Matrix d_m, const Vector d_v, Vector d_r) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i >= d_m.height || j >= d_m.width)
		return;
	
	float mat(d_m[i][j]);
	float vec(d_v[j]);	
	atomicAdd(&d_r[i], d_m[i][j] * d_v[j]);
}

/*
	Parallel implementation of Matrix-Vector product
	for GPU
	
	Naïve implementation:
		Checkboard partitioning
	
	TODO check: m.width = v.size
*/
Vector productMatrixVectorGPU_naive(const Matrix &h_m, const Vector &h_v) {
	Matrix d_m;
	d_m.width = h_m.width;
	d_m.height = h_m.height;
	cudaMalloc(&d_m.data, d_m.width * d_m.height * sizeof(float));
	cudaMemcpy(d_m.data, h_m.data, d_m.width * d_m.height * sizeof(float), cudaMemcpyHostToDevice);
	
	Vector d_v;
	d_v.size = h_v.size;
	cudaMalloc(&d_v.data, d_v.size * sizeof(float));
	cudaMemcpy(d_v.data, h_v.data, d_v.size * sizeof(float), cudaMemcpyHostToDevice);
	
	Vector d_r;
	d_r.size = d_m.height;
	cudaMalloc(&d_r.data, d_r.size * sizeof(float));
	cudaMemset(d_r.data, 0, d_r.size * sizeof(float));
	
	const dim3 num_threads(CHECKBOARD_BLOCK_MAX_HEIGHT, CHECKBOARD_BLOCK_MAX_WIDTH, 1);
	const dim3 num_blocks((d_m.height + CHECKBOARD_BLOCK_MAX_HEIGHT -1)/CHECKBOARD_BLOCK_MAX_HEIGHT, (d_m.width + CHECKBOARD_BLOCK_MAX_WIDTH -1)/CHECKBOARD_BLOCK_MAX_WIDTH, 1);
	productMatrixVectorGPU_naive_kernel<<<num_blocks, num_threads>>>(d_m, d_v, d_r);
	
	cudaFree(d_m.data);
	cudaFree(d_v.data);
	
	Vector h_r;
	h_r.size = d_r.size;
	h_r.data = new float[h_r.size];
	cudaMemcpy(h_r.data, d_r.data, d_r.size * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(d_r.data);
	
	return h_r;
}

int main(int argc, char **argv) {
	srand(time(NULL));
	
	Matrix h_m;
	h_m.width = WIDTH;
	h_m.height = HEIGHT;
	h_m.data = new float[WIDTH * HEIGHT];
	for (unsigned int i(0) ; i != HEIGHT ; i++) {
		for (unsigned int j(0) ; j != WIDTH ; j++) {
			h_m[i][j] = rand() % 10;
		}
	}
	//h_m.print();
	
	Vector h_v;
	h_v.size = WIDTH;
	h_v.data = new float[WIDTH];
	for (unsigned int j(0) ; j != WIDTH ; j++) {
		h_v[j] = rand() % 10;
	}
	//h_v.print();
	
	clock_t t_chrono;
	t_chrono = clock();
	for (int i(0) ; i != NUM_TESTS ; i++) {
		Vector h_r_cpu = productMatrixVectorCPU(h_m, h_v);
		delete [] h_r_cpu.data;
		//h_r_cpu.print();
	}
	t_chrono = clock() - t_chrono;
	std::cout << "Measured time for <CPU>: " << ((float) t_chrono)/CLOCKS_PER_SEC << "s" << std::endl;
	
	t_chrono = clock();
	for (int i(0) ; i != NUM_TESTS ; i++) {
		Vector h_r_gpu = productMatrixVectorGPU_naive(h_m, h_v);
		delete [] h_r_gpu.data;
		//h_r_gpu.print();
	}
	t_chrono = clock() - t_chrono;
	std::cout << "Measured time for <GPU>: " << ((float) t_chrono)/CLOCKS_PER_SEC << "s" << std::endl;
	
	return 0;
}

