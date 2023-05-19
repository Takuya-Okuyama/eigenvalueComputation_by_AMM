#ifndef __CORE_PROPOSEDAMM___
#define __CORE_PROPOSEDAMM___

template <int nthreads>
__global__ __launch_bounds__(nthreads) void generate_v(
    float *__restrict__ dv,
    const float *__restrict__ dz,
    const float val,
    const int m)
{
  for (int i = threadIdx.x; i < m / 4; i += nthreads)
  {
    reinterpret_cast<float4 *>(dv)[i] = reinterpret_cast<const float4 *>(dz)[i] - val;
  }
}

template <uint64_t nthreads_per_block,
          uint64_t NUM_UNROLL>
__global__ __launch_bounds__(nthreads_per_block) void proposedAMM_kernel1(
    const uint32_t m,
    const float *__restrict__ A,
    const float *__restrict__ alpha,
    const float *__restrict__ x,
    const float *__restrict__ x_memo,
    float *__restrict__ tmp,
    float *__restrict__ y,
    const float *__restrict__ y_memo,
    const uint32_t *__restrict__ d_ary,
    const float *__restrict__ d_weight,
    const double *__restrict__ d_param,
    const float *__restrict__ v)
{
  // total_items, sub_items, and main_items are guaranteed to be a multiple of 4.
  const uint32_t src = 4 * ((threadIdx.x * (m / 4)) / nthreads_per_block);
  const uint32_t dst = 4 * (((threadIdx.x + 1) * (m / 4)) / nthreads_per_block);
  const uint32_t total_items = dst - src;
  const uint64_t sub_items = total_items % (4 * NUM_UNROLL);
  const uint64_t main_items = total_items - sub_items;
  const double sum_x = d_param[0];
  const double dot_zx = d_param[1];

  x += src;
  x_memo += src;

  typedef cub::BlockReduce<float4, nthreads_per_block> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ uint4 sary;

  float4 regw, regA[8], regx[2], regx_memo[2], regalpha;
  float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  if (threadIdx.x == 0)
  {
    __pipeline_memcpy_async(&sary, d_ary + 4 * blockIdx.x, sizeof(int4));
    __pipeline_commit();
    __pipeline_wait_prior(0);

    regw.x = d_weight[sary.x];
    regw.y = d_weight[sary.y];
    regw.z = d_weight[sary.z];
    regw.w = d_weight[sary.w];

    regalpha.x = alpha[sary.x];
    regalpha.y = alpha[sary.y];
    regalpha.z = alpha[sary.z];
    regalpha.w = alpha[sary.w];
  }
  __syncthreads();

  uint4 idx_ary = m * sary + src;

  for (int j = 0; j < main_items; j += 4 * NUM_UNROLL)
  {
    vload(regx[0], x + j);
    vload(regx_memo[0], x_memo + j);
    regx[0] -= regx_memo[0];

    vload(regA[0], A + idx_ary.x);
    vload(regA[1], A + idx_ary.y);
    vload(regA[2], A + idx_ary.z);
    vload(regA[3], A + idx_ary.w);
    idx_ary += 4;

#pragma unroll
    for (int i = 4; i <= 4 * (NUM_UNROLL - 2);)
    {
      // prefetch
      vload(regx[1], x + j + i);
      vload(regx_memo[1], x_memo + j + i);
      regx[1] -= regx_memo[1];

      vload(regA[4], A + idx_ary.x);
      vload(regA[5], A + idx_ary.y);
      vload(regA[6], A + idx_ary.z);
      vload(regA[7], A + idx_ary.w);
      idx_ary += 4;

      val.x += dot(regA[0], regx[0]);
      val.y += dot(regA[1], regx[0]);
      val.z += dot(regA[2], regx[0]);
      val.w += dot(regA[3], regx[0]);
      i += 4;

      vload(regx[0], x + j + i);
      vload(regx_memo[0], x_memo + j + i);
      regx[0] -= regx_memo[0];

      vload(regA[0], A + idx_ary.x);
      vload(regA[1], A + idx_ary.y);
      vload(regA[2], A + idx_ary.z);
      vload(regA[3], A + idx_ary.w);
      idx_ary += 4;

      val.x += dot(regA[4], regx[1]);
      val.y += dot(regA[5], regx[1]);
      val.z += dot(regA[6], regx[1]);
      val.w += dot(regA[7], regx[1]);
      i += 4;
    }

    vload(regx[1], x + j + 4 * (NUM_UNROLL - 1));
    vload(regx_memo[1], x_memo + j + 4 * (NUM_UNROLL - 1));
    regx[1] -= regx_memo[1];

    vload(regA[4], A + idx_ary.x);
    vload(regA[5], A + idx_ary.y);
    vload(regA[6], A + idx_ary.z);
    vload(regA[7], A + idx_ary.w);
    idx_ary += 4;

    val.x += dot(regA[0], regx[0]) + dot(regA[4], regx[1]);
    val.y += dot(regA[1], regx[0]) + dot(regA[5], regx[1]);
    val.z += dot(regA[2], regx[0]) + dot(regA[6], regx[1]);
    val.w += dot(regA[3], regx[0]) + dot(regA[7], regx[1]);
  }

  if (sub_items > 0)
  {
    idx_ary = m * sary + src + main_items;
    x += main_items;
    x_memo += main_items;

    vload(regx[0], x);
    vload(regx_memo[0], x_memo);
    regx[0] -= regx_memo[0];

    vload(regA[0], A + idx_ary.x);
    vload(regA[1], A + idx_ary.y);
    vload(regA[2], A + idx_ary.z);
    vload(regA[3], A + idx_ary.w);
    idx_ary += 4;

    // sub_items is guaranteed to be a multiple of 4.
    for (int j = 4; j <= sub_items - 4; j += 4)
    {
      // prefetch
      vload(regx[1], x + j);
      vload(regx_memo[1], x_memo + j);
      regx[1] -= regx_memo[1];

      vload(regA[4], A + idx_ary.x);
      vload(regA[5], A + idx_ary.y);
      vload(regA[6], A + idx_ary.z);
      vload(regA[7], A + idx_ary.w);
      idx_ary += 4;

      val.x += dot(regA[0], regx[0]);
      val.y += dot(regA[1], regx[0]);
      val.z += dot(regA[2], regx[0]);
      val.w += dot(regA[3], regx[0]);

      regx[0] = regx[1];
      regx_memo[0] = regx_memo[1];
      regA[0] = regA[4];
      regA[1] = regA[5];
      regA[2] = regA[6];
      regA[3] = regA[7];
    }

    val.x += dot(regA[0], regx[0]);
    val.y += dot(regA[1], regx[0]);
    val.z += dot(regA[2], regx[0]);
    val.w += dot(regA[3], regx[0]);
  }

  float4 ret = BlockReduce(temp_storage).Sum(val);

  if (threadIdx.x == 0)
  {
    vstore(tmp + 4 * blockIdx.x, (ret - regalpha * sum_x) * regw);
  }

  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < m / 4; i += blockDim.x * gridDim.x)
  {
    float4 tmpy = reinterpret_cast<const float4 *>(y_memo)[i];
    float4 tmpf = reinterpret_cast<const float4 *>(v)[i];
    double4 tmpd = make_double4(fma(sum_x, (double)tmpf.x, dot_zx) + (double)tmpy.x,
                                fma(sum_x, (double)tmpf.y, dot_zx) + (double)tmpy.y,
                                fma(sum_x, (double)tmpf.z, dot_zx) + (double)tmpy.z,
                                fma(sum_x, (double)tmpf.w, dot_zx) + (double)tmpy.w);

    reinterpret_cast<float4 *>(y)[i] = make_float4((float)tmpd.x,
                                                   (float)tmpd.y,
                                                   (float)tmpd.z,
                                                   (float)tmpd.w);
  }
}

template <uint64_t nthreads_per_block,
          uint64_t nsubsamples,
          uint64_t NUM_UNROLL>
__global__ __launch_bounds__(nthreads_per_block) void proposedAMM_kernel2(
    const uint64_t m,
    const uint64_t k,
    const float *__restrict__ A,
    const float *__restrict__ alpha,
    const float *__restrict__ x,
    float *__restrict__ y,
    const uint32_t *__restrict__ d_ary)
{
  const uint64_t tx = threadIdx.x;
  const uint64_t src = 4 * ((blockIdx.y * (k / 4)) / nsubsamples);
  const uint64_t dst = 4 * (((blockIdx.y + 1) * (k / 4)) / nsubsamples);
  const uint64_t total_items = dst - src;
  const uint64_t sub_items = total_items % NUM_UNROLL;
  const uint64_t main_items = total_items - sub_items;

  __shared__ float sx[NUM_UNROLL];
  __shared__ uint32_t idx[NUM_UNROLL];

  A += 4 * (blockIdx.x * nthreads_per_block + tx);
  y += 4 * (blockIdx.x * nthreads_per_block + tx);
  float4 resY = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

#pragma unroll
  for (int j = 0; j < main_items; j += NUM_UNROLL)
  {
    __syncthreads();
    int offset = src + j;

#pragma unroll
    for (int k = tx; k < NUM_UNROLL / 4; k += nthreads_per_block)
    {
      __pipeline_memcpy_async(idx + 4 * k, d_ary + offset + 4 * k, sizeof(uint4));
      __pipeline_memcpy_async(sx + 4 * k, x + offset + 4 * k, sizeof(float4));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    float4 vecA[2];
    float shift[2];
    vload(vecA[0], A + m * idx[0]);
    shift[0] = alpha[idx[0]];

#pragma unroll
    for (int i = 0; i < NUM_UNROLL - 2; i += 2)
    {
      vload(vecA[1], A + m * idx[i + 1]);
      shift[1] = alpha[idx[i + 1]];
      resY += (vecA[0] - shift[0]) * sx[i];

      vload(vecA[0], A + m * idx[i + 2]);
      shift[0] = alpha[idx[i + 2]];
      resY += (vecA[1] - shift[1]) * sx[i + 1];
    }

    vload(vecA[1], A + m * idx[NUM_UNROLL - 1]);
    shift[1] = alpha[idx[NUM_UNROLL - 1]];
    resY += (vecA[0] - shift[0]) * sx[NUM_UNROLL - 2] +
            (vecA[1] - shift[1]) * sx[NUM_UNROLL - 1];
  }
  __syncthreads();

  if (sub_items > 0)
  {
    int offset = src + main_items;

    for (int k = tx; k < sub_items; k += nthreads_per_block)
    {
      idx[k] = d_ary[offset + k];
      sx[k] = x[offset + k];
    }
    __syncthreads();

    float4 vecA[2];
    float shift[2];
    vload(vecA[0], A + m * idx[0]);
    shift[0] = alpha[idx[0]];

    for (int i = 0; i < sub_items - 1; ++i)
    {
      vload(vecA[1], A + m * idx[i + 1]);
      shift[1] = alpha[idx[i + 1]];
      resY += (vecA[0] - shift[0]) * sx[i];

      vecA[0] = vecA[1];
      shift[0] = shift[1];
    }
    resY += (vecA[0] - shift[0]) * sx[sub_items - 1];
  }

  atomicAdd(y + 0, resY.x);
  atomicAdd(y + 1, resY.y);
  atomicAdd(y + 2, resY.z);
  atomicAdd(y + 3, resY.w);
}

#endif
