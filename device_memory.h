#ifndef __DEVICE_MEMORY__
#define __DEVICE_MEMORY__

#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cub/cub.cuh>

#include <vector>
#include <algorithm>
#include <random>
#include <sstream>

__global__ __launch_bounds__(128) void shift_and_scale(
    float *d_ptr,
    const uint64_t n,
    const float a,
    const float b)
{
  uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n)
  {
    d_ptr[idx] = a + (b - a) * d_ptr[idx];
  }
}

__global__ __launch_bounds__(128) void get_norms(
    float *d_normA,
    float *d_normAprime,
    float *d_alpha,
    const float *dA,
    const uint64_t m)
{
  uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < m)
  {
    double sum = 0.0, sum_pow2 = 0.0;
    uint64_t pos = m * idx;
    for (int i = 0; i < m; i++)
    {
      double v = dA[pos];
      sum += v;
      sum_pow2 = fma(v, v, sum_pow2);
      pos++;
    }

    double mean = sum / (double)m;
    d_alpha[idx] = (float)mean;
    d_normA[idx] = (float)sqrt(sum_pow2);
    d_normAprime[idx] = (float)sqrt(sum_pow2 - mean * mean * (double)m);
  }
}

__global__ __launch_bounds__(128) void convert_to_symmetry(
    float *d_ptr,
    const uint64_t size)
{
  const uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < size * size)
  {
    uint64_t y = idx % size;
    uint64_t x = idx / size;

    if (x < y)
    {
      d_ptr[x + y * size] = d_ptr[idx];
    }
  }
}

__global__ __launch_bounds__(128) void round_matrix(
    float *d_ptr,
    const uint64_t size)
{
  const uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < size * size)
  {
    d_ptr[idx] = round(4 * d_ptr[idx]) / 4;
  }
}

__global__ __launch_bounds__(128) void find_offset(
    float *dtmp,
    const float *d_ptr,
    const uint64_t size)
{
  const uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  const float *ptr = d_ptr + size * idx;
  float sum = 0.0f;

  for (int y = 0; y < size; ++y)
  {
    sum += abs(*ptr);
    ptr++;
  }

  dtmp[idx] = max(0.0f, sum - 2.0f * abs(d_ptr[idx + idx * size]));
}

__global__ __launch_bounds__(128) void add_offset(
    float *d_ptr,
    const float diagonal_offset,
    const uint64_t size)
{
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  d_ptr[idx + idx * size] += diagonal_offset;
}

struct device_memory
{
  float *dA = nullptr;
  float *dx = nullptr;
  float *dy = nullptr;
  float *dz = nullptr;
  float *dv = nullptr;
  float *dtmp = nullptr;
  double *dprm = nullptr;
  float *dx_memo = nullptr;
  float *dy_memo = nullptr;

  float *d_normA = nullptr;
  float *d_normAprime = nullptr;

  float *d_alpha = nullptr;
  float *d_w = nullptr;
  float *d_lambda = nullptr;

  float *d_weight = nullptr;
  float *d_acc_weight = nullptr;
  void *d_tmp_inclusiveSum = nullptr;
  void *d_tmp_sort = nullptr;

  double *d_dots = nullptr;
  double *d_eigs = nullptr;
  double *d_errs = nullptr;
  double *d_eigenvector = nullptr;

  float *d_vector = nullptr;
  float *d_memo = nullptr;

  std::size_t storageBytes_inclusiveSum = 0;
  std::size_t storageBytes_sum = 0;
  std::size_t storageBytes_sort = 0;

  uint32_t *d_rnd = nullptr;
  uint32_t *d_pos = nullptr;
  uint32_t *d_sorted_pos = nullptr;

  cublasHandle_t handle;
  curandGenerator_t gen;
  cudaStream_t stream_1;
  cudaStream_t stream_2;

  uint64_t m = 128;
  uint64_t original_m = 128;
  uint64_t c = 1;
  float const_one = 1.0f;
  float const_zero = 0.0f;
  std::string kernel;

  bool is_init = true;

  std::ofstream ofs_output;

  device_memory(host_memory &p)
  {
    m = p.m;
    original_m = p.original_m;
    c = p.c;
    seed = p.seed;
    kernel = p.kernel;

    cudaMalloc((void **)&dA, sizeof(float) * m * m);
    cudaMalloc((void **)&dz, sizeof(float) * 2 * m);
    cudaMalloc((void **)&dv, sizeof(float) * m);
    cudaMalloc((void **)&dprm, sizeof(double) * 2);

    cudaMalloc((void **)&d_vector, sizeof(float) * 3 * m);
    cudaMalloc((void **)&d_memo, sizeof(float) * 2 * m);

    cudaMalloc((void **)&d_normA, sizeof(float) * m);
    cudaMalloc((void **)&d_normAprime, sizeof(float) * m);

    cudaMalloc((void **)&d_w, sizeof(float));
    cudaMalloc((void **)&d_lambda, sizeof(float));

    cudaMalloc((void **)&d_weight, sizeof(float) * m);
    cudaMalloc((void **)&d_acc_weight, sizeof(float) * m);

    cudaMalloc((void **)&d_rnd, sizeof(uint32_t) * m);
    cudaMalloc((void **)&d_pos, sizeof(uint32_t) * m);
    cudaMalloc((void **)&d_sorted_pos, sizeof(uint32_t) * m);

    cudaMalloc((void **)&d_dots, sizeof(double));
    cudaMalloc((void **)&d_eigs, sizeof(double) * p.step);
    cudaMalloc((void **)&d_errs, sizeof(double) * p.step);
    cudaMalloc((void **)&d_eigenvector, sizeof(double) * p.m);

    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);

    cublasCreate(&handle);

    // Create pseudo-random number generator
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetStream(gen, stream_1);

    // Determine temporary device storage requirements
    cub::DeviceScan::InclusiveSum(
        d_tmp_inclusiveSum,
        storageBytes_inclusiveSum,
        d_weight,
        d_acc_weight,
        m,
        stream_1);

    cub::DeviceRadixSort::SortKeys(
        d_tmp_sort,
        storageBytes_sort,
        d_pos,
        d_sorted_pos,
        c,
        0, sizeof(uint32_t) * 8,
        stream_1);

    // Allocate temporary storage
    cudaMalloc(&d_tmp_inclusiveSum, storageBytes_inclusiveSum);
    cudaMalloc(&d_tmp_sort, storageBytes_sort);

    d_alpha = d_vector;
    dx = d_vector + m;
    dy = d_vector + 2 * m;

    dtmp = dz + m;

    dx_memo = d_memo;
    dy_memo = d_memo + m;

    // Load solution of eigenvector
    cudaMemset(d_eigenvector, 0, sizeof(double) * m);
    double *temp = new double[original_m];
    std::ifstream ifs("data/exact_eigenvector", std::ios::in | std::ios::binary);
    ifs.seekg(0);
    ifs.read((char *)temp, sizeof(double) * original_m);
    ifs.close();
    cudaMemcpy(d_eigenvector, temp, sizeof(double) * original_m, cudaMemcpyDefault);
    delete temp;

    auto ary = split(p.filepath, '/');
    std::string path_output("result/eigenvalues_by");
    path_output += "_" + kernel;
    path_output += "_" + ary[ary.size() - 1];
    path_output += "_r=" + std::to_string(p.nreps);
    path_output += "_step=" + std::to_string(p.step);
    if (p.kernel != "exact")
    {
      path_output += "_c=" + std::to_string(p.c);
      if (p.vr_adaptive_threshold == 0.0)
      {
        path_output += "_vr=" + std::to_string(p.vr_period);
      }
      else
      {
        path_output += "_vr=adaptive-" + p.str_vr_adaptive_threshold;
      }
    }
    path_output += ".binary";
    ofs_output.open(path_output, std::ios::out | std::ios::binary);
  }

  ~device_memory()
  {
    ofs_output.close();

    cudaFree(dA);
    cudaFree(dz);
    cudaFree(dv);
    cudaFree(dprm);

    cudaFree(d_memo);
    cudaFree(d_vector);

    cudaFree(d_normA);
    cudaFree(d_normAprime);

    cudaFree(d_w);
    cudaFree(d_lambda);

    cudaFree(d_weight);
    cudaFree(d_acc_weight);

    cudaFree(d_rnd);
    cudaFree(d_pos);
    cudaFree(d_sorted_pos);

    cudaFree(d_tmp_inclusiveSum);
    cudaFree(d_tmp_sort);

    cudaFree(d_dots);
    cudaFree(d_eigs);
    cudaFree(d_errs);
    cudaFree(d_eigenvector);

    curandDestroyGenerator(gen);
    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);
  }

  void write_data(host_memory &p)
  {
    // Set seed
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    p.diagonal_offset = generate_matrix(dA, m, p.original_m, p.matrixType, p.filepath);
    // output_matrix(dA, m, p.diagonal_offset);

    // zero clear
    zero_clear();

    // calculate norms
    get_norms<<<m / 128, 128>>>(
        d_normA,
        d_normAprime,
        d_alpha,
        dA,
        m);

    cublasSnrm2(handle, m, d_alpha, 1, &p.alpha2);
    p.alpha2 *= p.alpha2;
  }

  void set_internal_randomness()
  {
    curandSetPseudoRandomGeneratorSeed(gen, seed + 1234);
    curandGenerate(gen, d_rnd, m);
  }

  void zero_clear()
  {
    cudaMemset(dz, 0, sizeof(float) * 2 * m);
    cudaMemset(dv, 0, sizeof(float) * m);
    cudaMemset(dprm, 0, sizeof(float) * 2);

    cudaMemset(dx, 0, sizeof(float) * 2 * m);
    cudaMemset(d_memo, 0, sizeof(float) * 2 * m);

    is_init = true;
  }

private:
  uint32_t seed;

  float generate_matrix(float *d_ptr,
                        const uint64_t size,
                        const uint64_t original_size,
                        const std::string &type,
                        const std::string &path) const
  {
    float diagonal_offset = 0.0f;
    const uint64_t len = size * size;

    if (path != "")
    {
      float *h_matrix = new float[len];
      memset(h_matrix, 0, sizeof(float) * len);

      constexpr uint64_t max_height = 512;
      const uint64_t buffer_size = original_size * max_height;
      float *temp = new float[buffer_size];

      std::ifstream ifs(path, std::ios::in | std::ios::binary);
      ifs.seekg(0);
      ifs.read((char *)temp, sizeof(float)); // skip first element

      for (uint64_t offset = 0; offset < original_size; offset += max_height)
      {
        const uint64_t height = std::min(max_height, original_size - offset);
        ifs.read((char *)temp, sizeof(float) * original_size * height);
        uint64_t pos = 0;

        for (int y = offset; y < offset + height; y++)
        {
          for (int x = 0; x < original_size; x++)
          {
            h_matrix[y * size + x] = temp[pos++];
          }
        }
      }

      ifs.close();
      cudaMemcpy(d_ptr, h_matrix, sizeof(float) * len, cudaMemcpyDefault);
      delete h_matrix;
      delete temp;
    }
    else
    {
      std::string matrixType;
      float param1, param2;
      extract_parames(type, matrixType, param1, param2);

      if (matrixType == "gaussian")
      {
        curandGenerateNormal(gen, d_ptr, len, param1, param2);
        convert_to_symmetry<<<len / 128, 128>>>(d_ptr, size);
      }
      else if (matrixType == "lognormal")
      {
        curandGenerateLogNormal(gen, d_ptr, len, param1, param2);
        convert_to_symmetry<<<len / 128, 128>>>(d_ptr, size);
      }
      else if (matrixType == "uniform")
      {
        curandGenerateUniform(gen, d_ptr, len);
        shift_and_scale<<<len / 128, 128>>>(d_ptr, len, param1, param2);
        convert_to_symmetry<<<len / 128, 128>>>(d_ptr, size);
      }

      find_offset<<<m / 128, 128>>>(dtmp, d_ptr, size);

      int idx;
      cublasIsamax(handle, size, dtmp, 1, &idx);
      cudaMemcpy((void *)(&diagonal_offset), &dtmp[idx], sizeof(float), cudaMemcpyDefault);
      add_offset<<<m / 128, 128>>>(d_ptr, diagonal_offset, size);
    }

    return diagonal_offset;
  }

  void output_matrix(float *d_ptr,
                     const uint64_t size,
                     const float diagonal_offset) const
  {
    float *ptr = new float[size * size];
    cudaMemcpy(ptr, d_ptr, sizeof(float) * size * size, cudaMemcpyDefault);

    std::ofstream fs;
    fs.open("matrix.dat", std::ios::out | std::ios::binary);

    uint64_t idx = 0;
    for (int y = 0; y < size; y++)
    {
      float val = ptr[idx++];
      if (y == 0)
      {
        val -= diagonal_offset;
      }
      // fs << std::setprecision(12) << val;
      fs.write((char *)&val, sizeof(float));

      for (int x = 1; x < size; x++)
      {
        float val = ptr[idx++];
        if (y == x)
        {
          val -= diagonal_offset;
        }
        // fs << "," << std::setprecision(12) << val;
        fs.write((char *)&val, sizeof(float));
      }

      // fs << std::endl;
    }
    fs.close();

    delete ptr;
  }

  std::vector<std::string> split(const std::string &str, char sep) const
  {
    std::vector<std::string> v;
    std::stringstream ss(str);
    std::string buffer;
    while (std::getline(ss, buffer, sep))
    {
      v.push_back(buffer);
    }
    return v;
  }

  void extract_parames(
      const std::string str,
      std::string &matrixType,
      float &param1,
      float &param2) const
  {
    auto ary = split(str, '_');
    matrixType = ary[0];
    param1 = (ary.size() <= 1) ? 0.0 : std::atof(ary[1].c_str());
    param2 = (ary.size() <= 2) ? 0.0 : std::atof(ary[2].c_str());
  }
};

__global__ void set_value_for_sanity_check(float *d_w)
{
  uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  d_w[idx] = 1.0f;
}

__device__ __forceinline__ float get_rand(uint32_t *x)
{
  uint32_t y = *x;
  y ^= (y << 13);
  y ^= (y >> 17);
  y ^= (y << 5);
  *x = y;
  return __int_as_float((y & 0x007FFFFF) | 0x3f800000) - 1.0f;
}

__global__ __launch_bounds__(128) void pick_index(
    uint32_t *__restrict__ d_pos,
    uint32_t *__restrict__ d_rnd,
    const float *__restrict__ d_acc_weight,
    const uint64_t ntotal,
    const uint64_t nselect)
{
  uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= nselect)
  {
    return;
  }

  int bound_idx[2] = {-1, (int)ntotal - 1};
  float bound_val[2] = {0.0f, d_acc_weight[bound_idx[1]]};
  const float target_val = bound_val[1] * get_rand(&d_rnd[idx]);

  while (bound_idx[0] + 1 != bound_idx[1])
  {
    const int middle_idx = (bound_idx[0] + bound_idx[1]) / 2;
    const float middle_val = d_acc_weight[middle_idx];

    int offset = (middle_val <= target_val) ? 0 : 1;
    bound_idx[offset] = middle_idx;
    bound_val[offset] = middle_val;
  }

  d_pos[idx] = bound_idx[1];
  print_if_debugging("d_pos[%d] = %d\n", int(idx), int(d_pos[idx]));
}

__global__ __launch_bounds__(32) void set_sequential(uint32_t *d_pos)
{
  uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  *((uint4 *)(d_pos + 4 * idx)) = make_uint4(4 * idx, 4 * idx + 1, 4 * idx + 2, 4 * idx + 3);
}

#endif
