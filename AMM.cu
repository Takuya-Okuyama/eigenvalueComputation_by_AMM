#include "common.h"
#include "powerMethod_by_exact.cuh"
#include "powerMethod_by_previous.cuh"
#include "powerMethod_by_proposed.cuh"
#include "powerMethod_by_proposed_v2.cuh"

void execute_kernel(
    host_memory &p,
    device_memory &dm,
    const bool warmup = false)
{
  if (p.kernel == "exact")
  {
    if (warmup)
    {
      powerMethod_by_exact(p, dm, false);
    }
    else
    {
      printf("Time,Eigenvalue,NormError\n");
      for (int i = 0; i < p.nreps; ++i)
      {
        powerMethod_by_exact(p, dm);
      }
    }
  }
  else if (dm.kernel == "previousAMM")
  {
    if (warmup)
    {
      powerMethod_by_previous(p, dm, false);
    }
    else
    {
      printf("Time,Eigenvalue,NormError\n");
      for (int i = 0; i < p.nreps; ++i)
      {
        powerMethod_by_previous(p, dm);
      }
    }
  }
  else if (p.kernel == "proposedAMM")
  {
    if (warmup)
    {
      powerMethod_by_proposed(p, dm, false);
    }
    else
    {
      printf("Time,Eigenvalue,NormError\n");
      for (int i = 0; i < p.nreps; ++i)
      {
        powerMethod_by_proposed(p, dm);
      }
    }
  }
  else if (p.kernel == "proposedAMM_v2")
  {
    if (warmup)
    {
      powerMethod_by_proposed_v2(p, dm, false);
    }
    else
    {
      printf("Time,Eigenvalue,NormError\n");
      for (int i = 0; i < p.nreps; ++i)
      {
        powerMethod_by_proposed_v2(p, dm);
      }
    }
  }
}

int main(int argc, char *argv[])
{
  // set host memory
  host_memory p;
  assert(p.parser(argc, argv));

  // set device memory
  device_memory dm(p);
  dm.write_data(p);
  dm.set_internal_randomness();

  curandSetPseudoRandomGeneratorSeed(dm.gen, 1234ULL + p.seed);

  // warm-up
  execute_kernel(p, dm, true);
  cudaDeviceSynchronize();

  // measure time
  execute_kernel(p, dm);

  return 0;
}

