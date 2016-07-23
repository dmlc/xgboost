#pragma once
#include <xgboost/base.h>
#include <cuda_fp16.h>

namespace xgboost{
	namespace tree{
		//gpair type defined with device accessible functions
		struct gpu_gpair
		{
            float _grad;
			float _hess;

            __host__ __device__ __forceinline__ float grad() const {
                return _grad;
            }

            __host__ __device__ __forceinline__ float hess() const {
                return _hess;
            }

			__host__ __device__ gpu_gpair() : _grad(0), _hess(0)
			{
			}

			__host__ __device__ gpu_gpair(float g, float h) : _grad(g), _hess(h)
			{
			}

			__host__ __device__ gpu_gpair(bst_gpair gpair) : _grad(gpair.grad), _hess(gpair.hess)
			{
			}

			__host__ __device__ bool operator==(const gpu_gpair& rhs) const
			{
				return (_grad == rhs._grad) && (_hess == rhs._hess);
			}

			__host__ __device__ bool operator!=(const gpu_gpair& rhs) const
			{
				return !(*this == rhs);
			}

			__host__ __device__ gpu_gpair& operator+=(const gpu_gpair& rhs)
			{
                _grad += rhs._grad;
				_hess += rhs._hess;
				return *this;
			}

			__host__ __device__ gpu_gpair operator+(const gpu_gpair& rhs) const
			{
                gpu_gpair g = *this;
                g += rhs;
				return g;
			}

			__host__ __device__ gpu_gpair& operator-=(const gpu_gpair& rhs)
			{
				_grad -= rhs._grad;
				_hess -= rhs._hess;
				return *this;
			}

			__host__ __device__ gpu_gpair operator-(const gpu_gpair& rhs) const
			{
                gpu_gpair g = *this;
                g -= rhs;
				return g;
			}

			friend std::ostream& operator<<(std::ostream& os, const gpu_gpair& g)
			{
				os << g.grad() << "/" << g.hess();
				return os;
			}

			__host__ __device__ void print() const
			{
				printf("%1.4f/%1.4f\n", grad(), hess());
			}

			__host__ __device__ bool approximate_compare(const gpu_gpair& b, float g_eps = 0.1, float h_eps = 0.1) const
			{
				float gdiff = abs(this->grad() - b.grad());
				float hdiff = abs(this->hess() - b.hess());

				return (gdiff <= g_eps) && (hdiff <= h_eps);
			}
		};
	}
}