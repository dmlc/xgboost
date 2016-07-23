#pragma once
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <thrust/device_vector.h>
#include <sstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <algorithm>

#define safe_cuda(ans) throw_on_cuda_error((ans), __FILE__, __LINE__)

cudaError_t throw_on_cuda_error(cudaError_t code, const char* file, int line)
{
	if (code != cudaSuccess)
	{
		std::cout << file;
		std::cout << line;
		std::stringstream ss;
		ss << file << "(" << line << ")";
		std::string file_and_line;
		ss >> file_and_line;
		throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
	}

	return code;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
//Utility function: rounds up integer division. 
template <typename T>
T div_round_up(const T a, const T b)
{
	return (T)ceil((double)a/b);
}

template <typename T>
thrust::device_ptr<T> dptr(T*d_ptr){
	return thrust::device_pointer_cast(d_ptr);
}


//#define DEVICE_TIMER
#define MAX_WARPS 32 //Maximum number of warps to time
#define MAX_SLOTS 10
#define TIMER_BLOCKID 0 //Block to time
struct DeviceTimerGlobal{

#ifdef DEVICE_TIMER

	clock_t total_clocks[MAX_SLOTS][MAX_WARPS];
	long count[MAX_SLOTS][MAX_WARPS];

#endif

	//Clear device memory. Call at start of kernel.
	__device__ void Init()
	{
#ifdef DEVICE_TIMER
		if (blockIdx.x == TIMER_BLOCKID && threadIdx.x< MAX_WARPS)
		{
			for (int SLOT = 0; SLOT < MAX_SLOTS; SLOT++){
				total_clocks[SLOT][threadIdx.x] = 0;
				count[SLOT][threadIdx.x] = 0;
			}
		}
#endif
	}

	void HostPrint()
	{
#ifdef DEVICE_TIMER
		DeviceTimerGlobal h_timer;
		safe_cuda(cudaMemcpyFromSymbol(&h_timer, (* this), sizeof(DeviceTimerGlobal)));
		
		for (int SLOT = 0; SLOT < MAX_SLOTS; SLOT++){
			if (h_timer.count[SLOT][0] == 0)
			{
				continue;
			}

			clock_t sum_clocks = 0;
			long sum_count = 0;

			for (int WARP = 0; WARP < MAX_WARPS; WARP++)
			{
				if (h_timer.count[SLOT][WARP] == 0)
				{
					continue;
				}

				sum_clocks += h_timer.total_clocks[SLOT][WARP];
				sum_count += h_timer.count[SLOT][WARP];
			}

			printf("Slot %d: %d clocks per call, called %d times.\n", SLOT, sum_clocks / sum_count, h_timer.count[SLOT][0]);
		}
#endif
	}

};

struct DeviceTimer
{
#ifdef DEVICE_TIMER
	clock_t start;
	int slot;
	DeviceTimerGlobal &GTimer;
#endif

#ifdef DEVICE_TIMER
	__device__ DeviceTimer(DeviceTimerGlobal &GTimer, int slot) : GTimer(GTimer), start(clock()), slot(slot){}
#else
	__device__ DeviceTimer(DeviceTimerGlobal &GTimer, int slot){}
#endif

	__device__ void End()
	{
#ifdef DEVICE_TIMER
		int warp_id = threadIdx.x / 32;
		int lane_id = threadIdx.x % 32;
		if (blockIdx.x == TIMER_BLOCKID && lane_id == 0)
		{
			GTimer.count[slot][warp_id] += 1;
			GTimer.total_clocks[slot][warp_id] += clock() - start;
		}
#endif
	}

};

//#define TIMERS
struct Timer{
	size_t start;
	Timer(){
		reset();
	}
	void reset(){
		start = clock();
	}
	double elapsed(){
		return ((double)clock() - start) / CLOCKS_PER_SEC;
	}
	void printElapsed(char * label){
#ifdef TIMERS
		safe_cuda(cudaDeviceSynchronize());
		std::cout << label << ": " << elapsed() << "s\n";
#endif
	}

};

template <typename T>
void print(const thrust::device_vector<T>& v,  size_t max_items = 10)
{
	thrust::host_vector<T> h = v;
	for(int i = 0; i < std::min(max_items,h.size()); i++){
		std::cout << " " << h[i];
	}
	std::cout << "\n";
}

template <typename T>
void print(char *label, const thrust::device_vector<T>& v, const char * format = "%d ",int max = 10)
{
	thrust::host_vector<T> h_v = v;

	std::cout << label << ":\n";
	for (int i = 0; i < std::min((int)h_v.size(), max); i++)
	{
		printf(format, h_v[i]);
	}
	std::cout << "\n";
}

class range {
public:
	class iterator {
		friend class range;
	public:
		__host__ __device__
		long int operator *() const { return i_; }
		__host__ __device__
		const iterator &operator ++() { i_ += step_; return *this; }
		__host__ __device__
		iterator operator ++(int) { iterator copy(*this); i_ += step_; return copy; }

		__host__ __device__
		bool operator ==(const iterator &other) const { return i_ >= other.i_; }
		__host__ __device__
		bool operator !=(const iterator &other) const { return i_ < other.i_; }

		__host__ __device__
		void step(int s){ step_ = s; }
	protected:
		__host__ __device__
		iterator(long int start) : i_(start) { }

	//private:
	public:
		unsigned long i_;
		int step_ = 1;
	};

	__host__ __device__
	iterator begin() const { return begin_; }
	__host__ __device__
	iterator end() const { return end_; }
	__host__ __device__
	range(long int  begin, long int end) : begin_(begin), end_(end) {}
	__host__ __device__
	void step(int s) { begin_.step(s); }
private:
	iterator begin_;
	iterator end_;
};

template <typename T>
__device__ range grid_stride_range(T begin, T end){
	begin += blockDim.x * blockIdx.x + threadIdx.x;
	range r(begin, end);
	r.step(gridDim.x * blockDim.x);
	return r;
}

template <typename T>
__device__ range block_stride_range(T begin, T end){
	begin += threadIdx.x;
	range r(begin, end);
	r.step(blockDim.x);
	return r;
}

//Converts device_vector to raw pointer
template <typename T>
T * raw(thrust::device_vector<T>& v){
	return raw_pointer_cast(v.data());
}