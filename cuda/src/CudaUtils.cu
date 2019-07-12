#include <CudaUtils.h>

namespace miniFE {

  cudaStream_t CudaManager::s1;
  cudaStream_t CudaManager::s2;
  cudaEvent_t CudaManager::e1;
  cudaEvent_t CudaManager::e2;
  cudaEvent_t CudaManager::et[ET_COUNT];
  bool CudaManager::initialized=false;

}
