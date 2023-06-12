#ifndef __POWERMETHOD_BY_EXACT__
#define __POWERMETHOD_BY_EXACT__

void powerMethod_by_exact(
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

  for (std::size_t i = 0; i < p.step; ++i)
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

    constexpr int nthreads = 64;
    dotProduct_and_generate_x<nthreads><<<1, nthreads, 0, dm.stream_1>>>(
        dm.d_dots,
        dm.d_eigs + i,
        dm.d_errs + i,
        dm.dx,
        dm.d_eigenvector,
        dm.m);

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
