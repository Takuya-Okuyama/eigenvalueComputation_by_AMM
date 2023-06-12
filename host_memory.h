#ifndef __HOST_MEMORY__
#define __HOST_MEMORY__

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <omp.h>

template <typename T>
inline T DIV_CEIL(const T x, const T y)
{
  return (x + y - 1) / y;
}

class host_memory
{
public:
  uint64_t original_m = 128;
  uint64_t m = 128;
  uint32_t c = 0;
  uint64_t step = 10;
  uint64_t vr_period = 0;

  std::string str_vr_adaptive_threshold = "";
  double vr_adaptive_threshold = 0.0f;

  std::string kernel = "exact";
  std::string matrixType = "gaussian_0.0_1.0";
  std::string filepath = "";

  int verbose = 0;
  uint32_t nreps = 10;
  uint32_t seed;
  float alpha2 = 0.0f;
  const float const_one = 1.0f;
  const float const_zero = 0.0f;
  float diagonal_offset = 0.0f;

  bool sanity_check = false;
  bool variance_reduction = false;
  bool display = false;
  std::vector<double> error;

  host_memory()
  {
    seed = time(NULL);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~host_memory()
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  bool parser(int argc, char **argv)
  {
    for (int i = 1; i < argc; ++i)
    {
      if (!strcmp(argv[i], "-m"))
      {
        m = std::atoi(argv[++i]);
        m = DIV_CEIL(m, (uint64_t)128) * 128;
        original_m = m;
      }
      else if (!strcmp(argv[i], "-c"))
      {
        c = std::atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "-r"))
      {
        nreps = std::atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "-step"))
      {
        step = std::atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "-kernel"))
      {
        kernel = std::string(argv[++i]);
      }
      else if (!strcmp(argv[i], "-matrixType"))
      {
        matrixType = std::string(argv[++i]);
      }
      else if (!strcmp(argv[i], "-filepath"))
      {
        filepath = std::string(argv[++i]);
      }
      else if (!strcmp(argv[i], "-seed"))
      {
        seed = std::atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "-sanity"))
      {
        sanity_check = true;
      }
      else if (!strcmp(argv[i], "-display"))
      {
        display = true;
      }
      else if (!strcmp(argv[i], "-vr"))
      {
        vr_period = std::atoi(argv[++i]);
        variance_reduction = (vr_period > 0);
      }
      else if (!strcmp(argv[i], "-vr_adaptive"))
      {
        str_vr_adaptive_threshold = std::string(argv[++i]);
        vr_adaptive_threshold = std::atof(str_vr_adaptive_threshold.c_str());
        variance_reduction = (vr_adaptive_threshold > 0.0f);
      }
      else if (!strcmp(argv[i], "-verbose"))
      {
        verbose = std::atoi(argv[++i]);
      }
      else
      {
        printf("[warning] unknown parameter: %s\n", argv[i]);
      }
    }

    if (kernel == "exact")
    {
      c = m;
      sanity_check = false;
    }
    else if (kernel != "previousAMM" && kernel != "proposedAMM" && kernel != "proposedAMM_v2")
    {
      printf("[error] '-kernel %s' is not valid.\n", kernel.c_str());
      return false;
    }

    if (filepath != "")
    {
      std::ifstream ifs(filepath, std::ios::in | std::ios::binary);

      if (ifs.fail())
      {
        std::cerr << "Failed to open file." << std::endl;
        return false;
      }

      uint32_t val;
      ifs.read((char *)&val, sizeof(uint32_t));

      original_m = val;
      m = DIV_CEIL(val, (uint32_t)128) * 128;
      ifs.close();
    }

    if (c == 0)
    {
      c = m;
    }

    assert(1 <= c);
    assert(c <= m);
    assert(m % 128 == 0);
    assert(c % 8 == 0);

    if (verbose >= 2)
    {
      printf("[info] Parameters are as follows.\n");
      printf("\tkernel           : %s\n", kernel.c_str());
      printf("\tm                : %" PRIu64 "\n", m);
      printf("\tc                : %u\n", c);
      printf("\tnreps            : %u\n", nreps);
      printf("\tseed             : %d\n", seed);
      printf("\t# of threads     : %d\n", omp_get_max_threads());
      printf("\tdiagonal offset  : %f\n", diagonal_offset);
      printf("\tvr period        : %" PRIu64 "\n", vr_period);
      printf("[info] Sanity check      : %s\n", sanity_check ? "on" : "off");
      printf("[info] Variance reduction: %s\n", variance_reduction ? "on" : "off");
    }

    return true;
  }

private:
  cudaEvent_t start;
  cudaEvent_t stop;
};

#endif
