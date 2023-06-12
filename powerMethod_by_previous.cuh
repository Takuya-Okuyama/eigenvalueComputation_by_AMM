#ifndef __POWERMETHOD_BY_PREVIOUS__
#define __POWERMETHOD_BY_PREVIOUS__

#include "common.h"
#include "core_previousAMM.cuh"

void square_of_A_by_previous(device_memory &dm, const bool sanity_check = false)
{
  if (dm.is_init)
  {
    /*------------------------------------------------------------
     * Set weight
     *------------------------------------------------------------*/
    if (sanity_check && dm.c == dm.m)
    {
      set_value_for_sanity_check<<<dm.c / 128, 128, 0, dm.stream_1>>>(dm.d_weight);
    }
    else
    {
      set_weights<<<dm.m / 128, 128, 0, dm.stream_1>>>(
          dm.d_weight,
          dm.d_normA);
    }

    // Run inclusive prefix sum
    cub::DeviceScan::InclusiveSum(
        dm.d_tmp_inclusiveSum,
        dm.storageBytes_inclusiveSum,
        dm.d_weight,
        dm.d_acc_weight,
        dm.original_m,
        dm.stream_1);

    /*------------------------------------------------------------
     * Update weight
     *------------------------------------------------------------*/
    update_weights<<<dm.m / 128, 128, 0, dm.stream_1>>>(
        dm.d_weight,
        dm.d_acc_weight + dm.original_m - 1,
        dm.original_m,
        dm.c);

    dm.is_init = false;
  }

  {
    /*------------------------------------------------------------
     * Select columns / rows
     *------------------------------------------------------------*/
    if (sanity_check && dm.c == dm.m)
    {
      set_sequential<<<dm.m / 128, 32, 0, dm.stream_1>>>(dm.d_sorted_pos);
    }
    else
    {
      pick_index<<<DIV_CEIL(dm.c, (uint64_t)128), 128, 0, dm.stream_1>>>(
          dm.d_pos,
          dm.d_rnd,
          dm.d_acc_weight,
          dm.original_m,
          dm.c);

      cub::DeviceRadixSort::SortKeys(
          dm.d_tmp_sort,
          dm.storageBytes_sort,
          dm.d_pos,
          dm.d_sorted_pos,
          dm.c,
          0, sizeof(uint32_t) * 8,
          dm.stream_1);
    }

    constexpr uint64_t nthreads_per_block1 = 512;
    constexpr uint64_t NUM_UNROLL1 = 4;
    previousAMM_kernel1<nthreads_per_block1, NUM_UNROLL1>
        <<<dm.c / 4,
           nthreads_per_block1,
           0, dm.stream_1>>>(
            dm.m,
            dm.dA, dm.dx, dm.dx_memo, dm.dtmp,
            dm.dy, dm.dy_memo,
            dm.d_sorted_pos,
            dm.d_weight);

    constexpr uint64_t nthreads_per_block2 = 32;
    constexpr uint64_t nsubsamples2 = 16;
    constexpr uint64_t NUM_UNROLL2 = 32;
    previousAMM_kernel2<nthreads_per_block2, nsubsamples2, NUM_UNROLL2>
        <<<dim3(dm.m / (4 * nthreads_per_block2), nsubsamples2),
           nthreads_per_block2,
           0, dm.stream_1>>>(
            dm.m, dm.c,
            dm.dA,
            dm.dtmp,
            dm.dy,
            dm.d_sorted_pos);
  }
}

void powerMethod_by_previous(
    host_memory &p,
    device_memory &dm,
    const bool is_output = true)
{
  cudaEvent_t start;
  cudaEvent_t stop;

  float *ret_time;
  double *ret_eigs;
  double *ret_errs;
  dm.zero_clear();

  cublasSetStream(dm.handle, dm.stream_1);

  if (is_output)
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    ret_time = new float[p.step];
    ret_eigs = new double[p.step];
    ret_errs = new double[p.step];
  }

  // genearate dx[]
  constexpr int nthreads = 512;
  curandGenerateNormal(dm.gen, dm.dx, dm.m, 0.0, 1.0);
  normalize<nthreads><<<1, nthreads, 0, dm.stream_1>>>(dm.dx, dm.m);

  double dot = 0.0;
  const int step = is_output ? p.step : p.vr_period;
  for (std::size_t i = 0; i < step; ++i)
  {
    if (p.variance_reduction &&
        ((p.vr_period > 0 && (i + 1) % p.vr_period == 0) ||
         (p.vr_adaptive_threshold > 0.0f && dot > 1.0 - p.vr_adaptive_threshold)))
    {
      cublasSgemv(dm.handle,
                  CUBLAS_OP_N,
                  p.m, p.m,
                  &p.const_one,
                  dm.dA, p.m,
                  dm.dx, 1,
                  &p.const_zero,
                  dm.dtmp, 1);

      cublasSgemv(dm.handle,
                  CUBLAS_OP_N,
                  p.m, p.m,
                  &p.const_one,
                  dm.dA, p.m,
                  dm.dtmp, 1,
                  &p.const_zero,
                  dm.dy, 1);

      cudaMemcpy(dm.d_memo, dm.dx, sizeof(float) * 2 * p.m, cudaMemcpyDefault);
    }
    else
    {
      square_of_A_by_previous(dm, p.sanity_check);
    }

    constexpr int nthreads = 64;
    dotProduct_and_generate_x<nthreads><<<1, nthreads, 0, dm.stream_1>>>(
        dm.d_dots,
        dm.d_eigs + i,
        dm.d_errs + i,
        dm.dx,
        dm.d_eigenvector,
        dm.m);
    cudaMemcpy(&dot, dm.d_dots, sizeof(double), cudaMemcpyDefault);

    if (is_output)
    {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(ret_time + i, start, stop);
    }
  }

  if (is_output)
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(ret_eigs, dm.d_eigs, sizeof(double) * p.step, cudaMemcpyDefault);
    cudaMemcpy(ret_errs, dm.d_errs, sizeof(double) * p.step, cudaMemcpyDefault);

    for (std::size_t i = 0; i < p.step; ++i)
    {
      double time = ret_time[i];
      double eig = std::max(0.0, sqrt(std::max(0.0, ret_eigs[i])) - p.diagonal_offset);
      double err = 1.0 - ret_errs[i] * ret_errs[i];

      if (p.display)
      {
        printf("%.12lf,%.12lf,%.12lf\n", time, eig, err);
      }

      dm.ofs_output.write((char *)&time, sizeof(double));
      dm.ofs_output.write((char *)&eig, sizeof(double));
      dm.ofs_output.write((char *)&err, sizeof(double));
    }

    delete ret_time;
    delete ret_eigs;
    delete ret_errs;
  }
}

#endif
