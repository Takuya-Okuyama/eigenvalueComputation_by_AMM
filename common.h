#ifndef __COMMON_HEADER__
#define __COMMON_HEADER__

#ifdef DEBUG
#define print_if_debugging(fmt, ...) printf(fmt, ##__VA_ARGS__);
#else
#define print_if_debugging(fmt, ...)
#endif

// Header
#include <cassert>
#include <cinttypes>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <thrust/sort.h>
#include "helper_math.h"
#include "host_memory.h"
#include "device_memory.h"

// Device function
__device__ inline void vload(float4 &v, float const *addr)
{
  v = *((float4 *)(addr));
}

__device__ inline void vstore(float const *addr, float4 v)
{
  *((float4 *)(addr)) = v;
}

__global__ __launch_bounds__(128) void set_weights(
    float *__restrict__ weight,
    const float *__restrict__ normA)
{
  const uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  weight[idx] = normA[idx] * normA[idx];
}

__global__ __launch_bounds__(128) void update_weights(
    float *__restrict__ weight,
    const float *total_weight,
    const uint64_t original_m,
    const uint64_t c)
{
  const uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < original_m)
  {
    weight[idx] = (*total_weight) / (weight[idx] * ((float)c));
  }
}

template <int nthreads>
__global__ __launch_bounds__(nthreads) void dots(
    double *__restrict__ param,
    const float *__restrict__ x,
    const float *__restrict__ x_memo,
    const float *__restrict__ z,
    const uint64_t m)
{
  double2 val = make_double2(0.0f, 0.0f);
  for (int i = threadIdx.x; i < m / 4; i += nthreads)
  {
    float4 gx = reinterpret_cast<const float4 *>(x)[i] - reinterpret_cast<const float4 *>(x_memo)[i];
    float4 gz = reinterpret_cast<const float4 *>(z)[i];

    double dx[4] = {gx.x, gx.y, gx.z, gx.w};
    val.x += (dx[0] + dx[1]) + (dx[2] + dx[3]);

    val.y = fma(dx[0], (double)gz.x, val.y);
    val.y = fma(dx[1], (double)gz.y, val.y);
    val.y = fma(dx[2], (double)gz.z, val.y);
    val.y = fma(dx[3], (double)gz.w, val.y);
  }

  typedef cub::BlockReduce<double2, nthreads> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  double2 ret = BlockReduce(temp_storage).Sum(val);

  if (threadIdx.x == 0)
  {
    param[0] = ret.x;
    param[1] = ret.y;
  }
}

template <int nthreads>
__global__ __launch_bounds__(nthreads) void dotProduct_and_generate_x(
    double *__restrict__ d_eig,
    double *__restrict__ d_err,
    float *__restrict__ d_x,
    double *__restrict__ d_eigenvector,
    const int m)
{
  __shared__ float scale;

  /*------------------------------------------------------------
   * Calculate dot(x, y) and |y|
   *------------------------------------------------------------*/
  double3 val = make_double3(0.0, 0.0, 0.0);
  for (int i = threadIdx.x; i < m / 4; i += nthreads)
  {
    float4 gx = reinterpret_cast<float4 *>(d_x)[i];
    float4 gy = reinterpret_cast<float4 *>(d_x)[i + m / 4];
    double4 gz = reinterpret_cast<double4 *>(d_eigenvector)[i];

    double d = (double)gy.x;
    val.x = fma(d, (double)gx.x, val.x);
    val.y = fma(d, d, val.y);
    val.z = fma(d, gz.x, val.z);

    d = (double)gy.y;
    val.x = fma(d, (double)gx.y, val.x);
    val.y = fma(d, d, val.y);
    val.z = fma(d, gz.y, val.z);

    d = (double)gy.z;
    val.x = fma(d, (double)gx.z, val.x);
    val.y = fma(d, d, val.y);
    val.z = fma(d, gz.z, val.z);

    d = (double)gy.w;
    val.x = fma(d, (double)gx.w, val.x);
    val.y = fma(d, d, val.y);
    val.z = fma(d, gz.w, val.z);
  }

  typedef cub::BlockReduce<double3, nthreads> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  double3 ret = BlockReduce(temp_storage).Sum(val);

  if (threadIdx.x == 0)
  {
    *d_eig = ret.x;
    double dscale = rsqrt(ret.y);
    *d_err = ret.z * dscale;
    scale = (float)dscale;
  }
  __syncthreads();

  /*------------------------------------------------------------
   * Generate x
   *------------------------------------------------------------*/
  for (int i = threadIdx.x; i < m / 4; i += nthreads)
  {
    float4 gy = reinterpret_cast<float4 *>(d_x)[i + m / 4];
    reinterpret_cast<float4 *>(d_x)[i] = scale * gy;
  }
}

template <int nthreads>
__global__ __launch_bounds__(nthreads) void normalize(
    float *__restrict__ d_vector,
    const uint64_t m)
{
  double val = 0.0;
  for (uint64_t i = threadIdx.x; i < m / 4; i += nthreads)
  {
    float4 gx = reinterpret_cast<float4 *>(d_vector)[i];
    val += (double)dot(gx, gx);
  }

  typedef cub::BlockReduce<double, nthreads> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  double ret = BlockReduce(temp_storage).Sum(val);

  __shared__ float scale;
  if (threadIdx.x == 0)
  {
    scale = rsqrt(ret);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < m / 4; i += nthreads)
  {
    reinterpret_cast<float4 *>(d_vector)[i] *= scale;
  }
}

#endif
