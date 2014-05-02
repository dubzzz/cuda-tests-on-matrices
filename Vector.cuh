#ifndef __VECTOR_CUH__
#define __VECTOR_CUH__

#include <string.h>
#include <iostream>

class Vector {
private:
	const unsigned int size_;
	const bool isCPU_;
	
	int *smart_ptr_counter_; // num of instances of Vector which share data_
	float *data_;

public:
	Vector(const unsigned int &size, const bool &isCPU) : size_(size), isCPU_(isCPU) {
		smart_ptr_counter_ = new int(1);
		if (isCPU_)
			data_ = new float[size_];
		else
			cudaMalloc(&data_, size_ * sizeof(float));
	}
	
	Vector(const Vector &v) : size_(v.size_), data_(v.data_), isCPU_(v.isCPU_), smart_ptr_counter_(v.smart_ptr_counter_) {
		(*smart_ptr_counter_) += 1;
	}
	
	Vector(const Vector &v, const bool &isCPU) : size_(v.size_), isCPU_(isCPU) {
		smart_ptr_counter_ = new int(1);
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
		if(*smart_ptr_counter_ > 1) {// cuda-kernel constructs a copy of the object and then call its destructor
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
			memset(data_, 0, size_ * sizeof(float));
		else
			cudaMemset(data_, 0, size_ * sizeof(float));
	}
	
	__device__ __host__ float& get(const unsigned int &x) const {
		return data_[x];
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

#endif

