#ifndef __MATRIX_CUH__
#define __MATRIX_CUH__

/*
http://stackoverflow.com/questions/11994010/in-cuda-how-can-we-call-a-device-function-in-another-translation-unit
	* CUDA 4.2 and does not support static linking so device functions must be defined in the same compilation unit. A common technique is to write the device function in a .cuh file and include it in the .cu file.
	* CUDA 5.0 supports a new feature called separate compilation. The CUDA 5.0 VS msbuild rules should be available in the CUDA 5.0 RC download.
*/

#include <string.h>
#include <iostream>

class Matrix {
private:
	const unsigned int width_;
	const unsigned int height_;
	const bool isCPU_;
	
	int *smart_ptr_counter_; // num of instances of Matrix which share data_
	float *data_;

public:
	
	Matrix(const unsigned int &width, const unsigned int &height, const bool &isCPU) : width_(width), height_(height), isCPU_(isCPU) {
		smart_ptr_counter_ = new int(1);
		if (isCPU_)
			data_ = new float[width_ * height_];
		else
			cudaMalloc(&data_, width_ * height_ * sizeof(float));
	}
	
	Matrix(const Matrix &m) : width_(m.width_), height_(m.height_), data_(m.data_), isCPU_(m.isCPU_), smart_ptr_counter_(m.smart_ptr_counter_) {
		(*smart_ptr_counter_) += 1;
	}
	
	Matrix(const Matrix &m, const bool &isCPU) : width_(m.width_), height_(m.height_), isCPU_(isCPU) {
		smart_ptr_counter_ = new int(1);
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
		if (*smart_ptr_counter_ > 1) {
			(*smart_ptr_counter_) -= 1;
			return;
		}
		
		delete smart_ptr_counter_;
		
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
			
			/*
				WARNING
				Prefer using .get(x, y)
				for performances
			*/
			__device__ __host__ float& operator[](const unsigned int &y) const {
				return m_.data_[x_ * m_.width_ + y];
			}
	};
	
	/*
		Return the ith element of the matrix
		Equivalent to:
			(*this)[row][col]
		With:
			row = i / width_
			col = i % width_
		
		More efficient to retrieve an element
		It avoids having to create an instance of MatrixRow
		
		(*this).get(x * width_ + y) is more efficient than (*this)[x][y]
	*/
	__device__ __host__ float& get(const unsigned int &i) const {
		return data_[i];
	}
	
	/*
		More efficient than [][] to retrieve an element
		(*this).get(x, y) is more efficient than (*this)[x][y]
	*/
	__device__ __host__ float& get(const unsigned int &i, const unsigned int &j) const {
		return data_[i * width_ + j];
	}
		
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
				std::cout << " " << this->get(i, j);
			}
			std::cout << std::endl;
		}
		std::cout << "]" << std::endl;
	}
};

#endif

