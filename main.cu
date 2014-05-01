#include <string.h>
#include <iostream>

#include <time.h>
#include <stdlib.h>

#define WIDTH 1024
#define HEIGHT 256
#define NUM_TESTS 1000

#define CPU 1
#define GPU 0

#define CHECKBOARD_BLOCK_MAX_WIDTH  16
#define CHECKBOARD_BLOCK_MAX_HEIGHT 16

class Matrix {
private:
	const unsigned int width_;
	const unsigned int height_;
	const bool isCPU_;
	const bool isOriginal_;
	
	float *data_;

public:
	
	Matrix(const unsigned int &width, const unsigned int &height, const bool &isCPU) : width_(width), height_(height), isCPU_(isCPU), isOriginal_(true) {
		if (isCPU_)
			data_ = new float[width_ * height_];
		else
			cudaMalloc(&data_, width_ * height_ * sizeof(float));
	}
	
	Matrix(const Matrix &m) : width_(m.width_), height_(m.height_), data_(m.data_), isCPU_(m.isCPU_), isOriginal_(false) {}
	
	Matrix(const Matrix &m, const bool &isCPU) : width_(m.width_), height_(m.height_), isCPU_(isCPU), isOriginal_(true) {
		if (isCPU_) {
			data_ = new float[width_ * height_];
			if (m.isCPU_)
				memcpy(data_, m.data_, width_ * height_ * sizeof(float));
			else
				cudaMemcpy(data_, m.data_, width_ * height_ * sizeof(float), cudaMemcpyDeviceToHost);
		} else {
			cudaMalloc(&data_, width_ * height_ * sizeof(float));
			if (m.isCPU_)
				cudaMemcpy(data_, m.data_, width_ * height_ * sizeof(float), cudaMemcpyHostToDevice);
			else
				cudaMemcpy(data_, m.data_, width_ * height_ * sizeof(float), cudaMemcpyDeviceToDevice);
		}
	}
	
	~Matrix() {
		if (! isOriginal_)
			return;
		
		if (isCPU_)
			delete [] data_;
		else
			cudaFree(data_);
	}
	
	void memsetZero() {
		if (isCPU_)
			memset(data_, 0, width_ * height_ * sizeof(float));
		else
			cudaMemset(data_, 0, width_ * height_ * sizeof(float));
	}
	
	class MatrixRow { // cf. http://stackoverflow.com/questions/3755111/how-do-i-define-a-double-brackets-double-iterator-operator-similar-to-vector-of
		const Matrix &m_;
		const unsigned int x_;
		
		public:
			__device__ __host__ MatrixRow(const Matrix &m, const unsigned int &x) : m_(m), x_(x) {}
			__device__ __host__ float& operator[](const unsigned int &y) const {
				return m_. data_[x_ * m_.width_ + y];
			}
	};
	
	__device__ __host__ MatrixRow operator[](const unsigned int &x) const {
		return MatrixRow(*this, x);
	}
	
	__device__ __host__ unsigned int getWidth() const { return width_; }
	__device__ __host__ unsigned int getHeight() const { return height_; }
	__device__ __host__ bool isCPU() const { return isCPU_; }
	
	void print() {
		std::cout << "[" << std::endl;
		for (unsigned int i(0) ; i != height_ ; i++) {
			for (unsigned int j(0) ; j != width_ ; j++) {
				std::cout << " " << (*this)[i][j];
			}
			std::cout << std::endl;
		}
		std::cout << "]" << std::endl;
	}
};

class Vector {
private:
	const unsigned int size_;
	const bool isCPU_;
	const bool isOriginal_;
	
	float *data_;

public:
	Vector(const unsigned int &size, const bool &isCPU) : size_(size), isCPU_(isCPU), isOriginal_(true) {
		if (isCPU_)
			data_ = new float[size_];
		else
			cudaMalloc(&data_, size_ * sizeof(float));
	}
	
	Vector(const Vector &v) : size_(v.size_), data_(v.data_), isCPU_(v.isCPU_), isOriginal_(false) {}
	
	Vector(const Vector &v, const bool &isCPU) : size_(v.size_), isCPU_(isCPU), isOriginal_(true) {
		if (isCPU_) {
			data_ = new float[size_];
			if (v.isCPU_)
				memcpy(data_, v.data_, size_ * sizeof(float));
			else
				cudaMemcpy(data_, v.data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
		} else {
			cudaMalloc(&data_, size_ * sizeof(float));
			if (v.isCPU_)
				cudaMemcpy(data_, v.data_, size_ * sizeof(float), cudaMemcpyHostToDevice);
			else
				cudaMemcpy(data_, v.data_, size_ * sizeof(float), cudaMemcpyDeviceToDevice);
		}
	}
	
	~Vector() {
		if(! isOriginal_) // cuda-kernel constructs a copy of the object and then call its destructor
			return;
		if (isCPU_)
			delete [] data_;
		else
			cudaFree(data_);
	}
	
	void memsetZero() {
		if (isCPU_)
			memset(data_, 0, size_ * sizeof(float));
		else
			cudaMemset(data_, 0, size_ * sizeof(float));
	}
	
	__device__ __host__ float& operator[](const unsigned int &x) const {
		return data_[x];
	}
	
	__device__ __host__ unsigned int getSize() const { return size_; }
	__device__ __host__ bool isCPU() const { return isCPU_; }
	
	void print() {
		std::cout << "[";
		for (unsigned int j(0) ; j != size_ ; j++) {
			std::cout << " " << (*this)[j];
		}
		std::cout << " ]" << std::endl;
	}
};

/*
	Serial implementation of Matrix-Vector product
	for CPU
	
	TODO check: m.getWidth() = v.getSize()
*/
Vector productMatrixVectorCPU(const Matrix &m, const Vector &v) {
	Vector r(m.getHeight(), CPU);
	
	for (unsigned int i(0) ; i != m.getHeight() ; i++) {
		r[i] = 0;
		for (unsigned int j(0) ; j != m.getWidth() ; j++) {
			r[i] += m[i][j] * v[j];
		}
	}
	return r;
}

__global__ void productMatrixVectorGPU_naive_kernel(const Matrix d_m, const Vector d_v, Vector d_r) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i >= d_m.getHeight() || j >= d_m.getWidth())
		return;
	
	atomicAdd(&d_r[i], d_m[i][j] * d_v[j]);
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
		Vector h_r_cpu = productMatrixVectorCPU(h_m, h_v);
		//h_r_cpu.print();
	}
	t_chrono = clock() - t_chrono;
	std::cout << "Measured time for <CPU>: " << ((float) t_chrono)/CLOCKS_PER_SEC << "s" << std::endl;
	
	t_chrono = clock();
	for (int i(0) ; i != NUM_TESTS ; i++) {
		Vector h_r_gpu = productMatrixVectorGPU_naive(h_m, h_v);
		//h_r_gpu.print();
	}
	t_chrono = clock() - t_chrono;
	std::cout << "Measured time for <GPU>: " << ((float) t_chrono)/CLOCKS_PER_SEC << "s" << std::endl;
	
	return 0;
}

