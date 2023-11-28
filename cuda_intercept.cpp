/*********************

MIT License

Copyright (c) 2020 Christos Konstantinos Matzoros
   - Original
Copyright (c) 2023 Sam Iredale 
   - Refactor for load-time function pointer initialization
   - Addition of CUDA_DEBUG_MSG macro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

***********************/

//Headers
#include <stdio.h>
#include <list>
#include <map>
#include <cassert>
#include <vector_types.h>
#include <dlfcn.h>  //for dynamic linking
#include <cuda.h>
#include <driver_types.h>
using namespace std;

#if defined(CUDA_DEBUG)
  #define CUDA_DEBUG_MSG(...) printf(__VA_ARGS__)
#else
  #define CUDA_DEBUG_MSG(...) 
#endif

typedef struct {
    dim3 gridDim;
    dim3 blockDim;
    list <void*> arguments;
    int counter;
} kernel_info_t;

static list<kernel_info_t> kernels_list;

kernel_info_t &kernelInfo() {
    static kernel_info_t kernelInfo;
    return kernelInfo;
}

/////////////////////////
//   PRINT FUNCTIONS   //
/////////////////////////

void print_grid_dimensions(dim3 gridDim){
    if (gridDim.y == 1 && gridDim.z == 1) {     //1D grid (x)
        printf("gridDim=%d ", gridDim.x);
    } else if (gridDim.z == 1) {    //2D grid (x,y)
        printf("gridDim=[%d,%d] ", gridDim.x, gridDim.y);
    } else { //3D grid (x,y,z)
        printf("gridDim=[%d,%d,%d] ", gridDim.x, gridDim.y, gridDim.z);
    }
}

void print_block_dimensions(dim3 blockDim){
    if (blockDim.y == 1 && blockDim.z == 1) {   //1D block (x)
        printf("blockDim=%d ", blockDim.x);
    } else if (blockDim.z == 1) {   //2D block (x,y)
        printf("blockDim=[%d,%d] ", blockDim.x, blockDim.y);
    } else {    //3D block (x,y,z)
        printf("blockDim=[%d,%d,%d] ", blockDim.x, blockDim.y, blockDim.z);
    }
}

void print_dimensions(dim3 gridDim, dim3 blockDim){
    print_grid_dimensions(gridDim);
    print_block_dimensions(blockDim);
}

void print_args(list <void*> arg){
    for (std::list<void *>::iterator it = arg.begin(), end = arg.end(); it != end; ++it) {
        unsigned i = std::distance(arg.begin(), it);
        printf("%d:%d \n", i, *(static_cast<int *>(*it)));
    }
}

void print_kernel_invocation(const char *entry) {
    printf("New kernel invocation\n");
    print_dimensions(kernelInfo().gridDim,kernelInfo().blockDim);
    //print_args(kernelInfo().arguments);
    printf("\n");
}

//**********************************************//
//      Cuda function pointer types             //
//**********************************************//
typedef const char* (*cudaGetErrorName_t)(cudaError_t error);
typedef const char* (*cudaGetErrorString_t)(cudaError_t error);
typedef cudaError_t (*cudaGetLastError_t)(void);
typedef cudaError_t (*cudaPeekAtLastError_t)(void);
typedef cudaError_t (*cudaChooseDevice_t)(int * device, const struct cudaDeviceProp * prop);
typedef cudaError_t (*cudaDeviceGetAttribute_t)(int* value, cudaDeviceAttr attr, int device);
typedef cudaError_t (*cudaDeviceGetByPCIBusId_t)(int* device, const char* pciBusId);
typedef cudaError_t (*cudaDeviceGetCacheConfig_t)(cudaFuncCache ** pCacheConfig);
typedef cudaError_t (*cudaDeviceGetLimit_t)(size_t* pValue, cudaLimit limit);
typedef cudaError_t (*cudaDeviceGetNvSciSyncAttributes_t)( void* nvSciSyncAttrList, int device, int flags);
typedef cudaError_t (*cudaDeviceGetP2PAttribute_t)(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);
typedef cudaError_t (*cudaDeviceGetPCIBusId_t)(char* pciBusId, int len, int device);
typedef cudaError_t (*cudaDeviceGetSharedMemConfig_t)( cudaSharedMemConfig ** pConfig );
typedef cudaError_t (*cudaDeviceGetStreamPriorityRange_t)( int* leastPriority, int* greatestPriority);
typedef cudaError_t (*cudaDeviceSetCacheConfig_t)(cudaFuncCache cacheConfig);
typedef cudaError_t (*cudaDeviceSetLimit_t)(cudaLimit limit, size_t value);
typedef cudaError_t (*cudaDeviceSetSharedMemConfig_t)(cudaSharedMemConfig config);
typedef cudaError_t (*cudaDeviceSynchronize_t)(void);
typedef cudaError_t (*cudaGetDevice_t)(int *device);
typedef cudaError_t (*cudaGetDeviceCount_t)(int * count);
typedef cudaError_t (*cudaGetDeviceFlags_t)(unsigned int* flags);
typedef cudaError_t (*cudaGetDeviceProperties_t)(struct cudaDeviceProp * prop, int device);
typedef cudaError_t (*cudaIpcCloseMemHandle_t)(void* devPtr);
typedef cudaError_t (*cudaIpcGetEventHandle_t)(cudaIpcEventHandle_t* handle, cudaEvent_t event);
typedef cudaError_t (*cudaIpcGetMemHandle_t)(cudaIpcMemHandle_t* handle, void* devPtr);
typedef cudaError_t (*cudaIpcOpenEventHandle_t)(cudaEvent_t* event, cudaIpcEventHandle_t handle);
typedef cudaError_t (*cudaIpcOpenMemHandle_t)(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags);
typedef cudaError_t (*cudaSetDevice_t)(int device);
typedef cudaError_t (*cudaSetDeviceFlags_t)(int flags);
typedef cudaError_t (*cudaSetValidDevices_t)(int * device_arr, int len);
typedef cudaError_t (*cudaStreamAttachMemAsync_t)(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags);
typedef cudaError_t (*cudaStreamCreate_t)(cudaStream_t * pStream);
typedef cudaError_t (*cudaStreamCreateWithFlags_t)(cudaStream_t* pStream, unsigned int  flags);
typedef cudaError_t (*cudaStreamCreateWithPriority_t)(cudaStream_t* pStream, unsigned int flags, int priority);
typedef cudaError_t (*cudaStreamDestroy_t)(cudaStream_t stream);
typedef cudaError_t (*cudaStreamGetFlags_t)(cudaStream_t hStream, unsigned int* flags);
typedef cudaError_t (*cudaStreamGetPriority_t)(cudaStream_t hStream, int* priority);
typedef cudaError_t (*cudaStreamQuery_t)(cudaStream_t stream);
typedef cudaError_t (*cudaStreamSynchronize_t)(cudaStream_t stream);
typedef cudaError_t (*cudaStreamWaitEvent_t)(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
typedef cudaError_t (*cudaEventCreate_t)(cudaEvent_t * event);
typedef cudaError_t (*cudaEventCreateWithFlags_t)(cudaEvent_t * event, int flags);
typedef cudaError_t (*cudaEventDestroy_t)(cudaEvent_t event);
typedef cudaError_t (*cudaEventElapsedTime_t)(float * ms, cudaEvent_t start, cudaEvent_t end);
typedef cudaError_t (*cudaEventQuery_t)(cudaEvent_t event);
typedef cudaError_t (*cudaEventRecord_t)(cudaEvent_t event, cudaStream_t stream);
typedef cudaError_t (*cudaEventSynchronize_t)(cudaEvent_t event);
typedef cudaError_t (*cudaConfigureCall_t)(dim3,dim3,size_t,cudaStream_t);
typedef cudaError_t (*cudaFuncGetAttributes_t)(struct cudaFuncAttributes * attr, const char * func);
typedef cudaError_t (*cudaFuncSetAttribute_t)(const void* func, cudaFuncAttribute attr, int  value);
typedef cudaError_t (*cudaLaunch_t)(const char* entry);
typedef cudaError_t (*cudaFuncSetCacheConfig_t)(const void* func, cudaFuncCache cacheConfig);
typedef cudaError_t (*cudaFuncSetSharedMemConfig_t)(const void* func, cudaSharedMemConfig config);
typedef cudaError_t (*cudaGetParameterBuffer_t)(size_t alignment, size_t size);
typedef cudaError_t (*cudaGetParameterBufferV2_t)(void* func, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize);
typedef cudaError_t (*cudaLaunchCooperativeKernel_t)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
typedef cudaError_t (*cudaLaunchCooperativeKernelMultiDevice_t)(cudaLaunchParams* launchParamsList, unsigned int numDevices, unsigned int flags);
typedef cudaError_t (*cudaLaunchKernel_t)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
typedef cudaError_t (*cudaSetDoubleForDevice_t)(double *d);
typedef cudaError_t (*cudaSetDoubleForHost_t)(double *d);
typedef cudaError_t (*cudaSetupArgument_t)(const void *, size_t, size_t);
typedef cudaError_t (*cudaFree_t)(void * devPtr);
typedef cudaError_t (*cudaFreeArray_t)(struct cudaArray * array);
typedef cudaError_t (*cudaFreeHost_t)(void * ptr);
typedef cudaError_t (*cudaGetSymbolAddress_t)(void ** devPtr, const char * symbol);
typedef cudaError_t (*cudaGetSymbolSize_t)(size_t * size, const char * symbol);
typedef cudaError_t (*cudaHostAlloc_t)(void ** ptr, size_t size, unsigned int flags);
typedef cudaError_t (*cudaHostGetDevicePointer_t)(void ** pDevice, void * pHost, unsigned int flags);
typedef cudaError_t (*cudaHostGetFlags_t)(unsigned int * pFlags, void * pHost);
typedef cudaError_t (*cudaMalloc_t)(void ** devPtr, size_t size);
typedef cudaError_t (*cudaMalloc3D_t)(struct cudaPitchedPtr * pitchedDevPtr, struct cudaExtent extent);
typedef cudaError_t (*cudaMalloc3DArray_t)(struct cudaArray ** arrayPtr, const struct cudaChannelFormatDesc * desc, struct cudaExtent extent);
typedef cudaError_t (*cudaMallocArray_t)(struct cudaArray ** arrayPtr, const struct cudaChannelFormatDesc * desc, size_t width, size_t height);
typedef cudaError_t (*cudaMallocHost_t)(void ** ptr,size_t size);
typedef cudaError_t (*cudaMallocPitch_t)(void ** devPtr, size_t * pitch, size_t width, size_t height);
typedef cudaError_t (*cudaMemcpy_t)(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind);
typedef cudaError_t (*cudaMemcpy2D_t)(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
typedef cudaError_t (*cudaMemcpy2DArrayToArray_t)(struct cudaArray * dst,
    size_t  wOffsetDst,
    size_t  hOffsetDst,
    const struct cudaArray * src,
    size_t  wOffsetSrc,
    size_t  hOffsetSrc,
    size_t  width,
    size_t  height,
    enum cudaMemcpyKind kind);
typedef cudaError_t (*cudaMemcpy2DAsync_t)(void * dst,
    size_t  dpitch,
    const void * src,
    size_t  spitch,
    size_t  width,
    size_t  height,
    enum cudaMemcpyKind kind,
    cudaStream_t stream);
typedef cudaError_t (*cudaMemcpy2DFromArray_t)(void * dst,
    size_t  dpitch,
    const struct cudaArray * src,
    size_t  wOffset,
    size_t  hOffset,
    size_t  width,
    size_t  height,
    enum cudaMemcpyKind kind);
typedef cudaError_t (*cudaMemcpy2DFromArrayAsync_t)(void * dst,
    size_t  dpitch,
    const struct cudaArray * src,
    size_t  wOffset,
    size_t  hOffset,
    size_t  width,
    size_t  height,
    enum cudaMemcpyKind kind,
    cudaStream_t stream);
typedef cudaError_t (*cudaMemcpy2DToArray_t)(struct cudaArray * dst,
    size_t  wOffset,
    size_t  hOffset,
    const void * src,
    size_t  spitch,
    size_t  width,
    size_t  height,
    enum cudaMemcpyKind kind);
typedef cudaError_t (*cudaMemcpy2DToArrayAsync_t)(struct cudaArray * dst,
    size_t  wOffset,
    size_t  hOffset,
    const void * src,
    size_t  spitch,
    size_t  width,
    size_t  height,
    enum cudaMemcpyKind kind,
    cudaStream_t stream);
typedef cudaError_t (*cudaMemcpy3D_t)(const struct cudaMemcpy3DParms * p);
typedef cudaError_t (*cudaMemcpy3DAsync_t)(const struct cudaMemcpy3DParms * p, cudaStream_t stream);
typedef cudaError_t (*cudaMemcpyArrayToArray_t)(struct cudaArray * dst,
    size_t  wOffsetDst,
    size_t  hOffsetDst,
    const struct cudaArray * src,
    size_t  wOffsetSrc,
    size_t  hOffsetSrc,
    size_t  count,
    enum cudaMemcpyKind kind);
typedef cudaError_t (*cudaMemcpyAsync_t)(void * dst,
    const void * src,
    size_t  count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream);
typedef cudaError_t (*cudaMemcpyFromArray_t)(void * dst,
    const struct cudaArray * src,
    size_t wOffset,
    size_t hOffset,
    size_t count,
    enum cudaMemcpyKind kind);
typedef cudaError_t (*cudaMemcpyFromArrayAsync_t)(void * dst,
    const struct cudaArray * src,
    size_t  wOffset,
    size_t  hOffset,
    size_t  count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream);
typedef cudaError_t (*cudaMemcpyFromSymbol_t)(void * dst,
    const char * symbol,
    size_t  count,
    size_t  offset,
    enum cudaMemcpyKind kind);
typedef cudaError_t (*cudaMemcpyFromSymbolAsync_t)(void * dst,
    const char * symbol,
    size_t  count,
    size_t  offset,
    enum cudaMemcpyKind kind,
    cudaStream_t stream);
typedef cudaError_t (*cudaMemcpyToArray_t)(struct cudaArray * dst,
    size_t  wOffset,
    size_t  hOffset,
    const void * src,
    size_t  count,
    enum cudaMemcpyKind kind);
typedef cudaError_t (*cudaMemcpyToArrayAsync_t)(struct cudaArray * dst,
    size_t  wOffset,
    size_t  hOffset,
    const void * src,
    size_t  count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream);
typedef cudaError_t (*cudaMemcpyToSymbol_t)(const char * symbol,
    const void * src,
    size_t  count,
    size_t  offset,
    enum cudaMemcpyKind kind);
typedef cudaError_t (*cudaMemcpyToSymbolAsync_t)(const char * symbol,
    const void * src,
    size_t  count,
    size_t  offset,
    enum cudaMemcpyKind kind,
    cudaStream_t stream);
typedef cudaError_t (*cudaMemset_t)(void * devPtr, int value, size_t count);
typedef cudaError_t (*cudaMemset2D_t)(void * devPtr, size_t pitch, int value, size_t width, size_t height);
typedef cudaError_t (*cudaMemset3D_t)(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);
typedef cudaError_t (*cudaDriverGetVersion_t)(int * driverVersion);
typedef cudaError_t (*cudaRuntimeGetVersion_t)(int * runtimeVersion);
typedef cudaError_t (*cudaThreadExit_t)(void);
typedef cudaError_t (*cudaThreadSynchronize_t)(void);

//**********************************************//
//      Native function pointers                //
//**********************************************//
static cudaGetErrorName_t native_cudaGetErrorName = NULL;
static cudaGetErrorString_t native_cudaGetErrorString = NULL;
static cudaGetLastError_t native_cudaGetLastError = NULL;
static cudaPeekAtLastError_t native_cudaPeekAtLastError = NULL;
static cudaChooseDevice_t native_cudaChooseDevice = NULL;
static cudaDeviceGetAttribute_t native_cudaDeviceGetAttribute = NULL;
static cudaDeviceGetByPCIBusId_t native_cudaDeviceGetByPCIBusId  = NULL;
static cudaDeviceGetCacheConfig_t native_cudaDeviceGetCacheConfig = NULL;
static cudaDeviceGetLimit_t native_cudaDeviceGetLimit = NULL;
static cudaDeviceGetNvSciSyncAttributes_t native_cudaDeviceGetNvSciSyncAttributes = NULL;
static cudaDeviceGetP2PAttribute_t native_cudaDeviceGetP2PAttribute= NULL;
static cudaDeviceGetPCIBusId_t native_cudaDeviceGetPCIBusId = NULL;
static cudaDeviceGetSharedMemConfig_t native_cudaDeviceGetSharedMemConfig = NULL;
static cudaDeviceGetStreamPriorityRange_t native_cudaDeviceGetStreamPriorityRange = NULL;
static cudaDeviceSetCacheConfig_t native_cudaDeviceSetCacheConfig = NULL;
static cudaDeviceSetLimit_t native_cudaDeviceSetLimit = NULL;
static cudaDeviceSetSharedMemConfig_t native_cudaDeviceSetSharedMemConfig = NULL;
static cudaDeviceSynchronize_t native_cudaDeviceSynchronize = NULL;
static cudaGetDevice_t native_cudaGetDevice = NULL;
static cudaGetDeviceCount_t native_cudaGetDeviceCount = NULL;
static cudaGetDeviceFlags_t native_cudaGetDeviceFlags = NULL;
static cudaGetDeviceProperties_t native_cudaGetDeviceProperties = NULL;
static cudaIpcCloseMemHandle_t native_cudaIpcCloseMemHandle = NULL;
static cudaIpcGetEventHandle_t native_cudaIpcGetEventHandle = NULL;
static cudaIpcGetMemHandle_t native_cudaIpcGetMemHandle= NULL;
static cudaIpcOpenEventHandle_t native_cudaIpcOpenEventHandle = NULL;
static cudaIpcOpenMemHandle_t native_cudaIpcOpenMemHandle = NULL;
static cudaSetDevice_t native_cudaSetDevice = NULL;
static cudaSetDeviceFlags_t native_cudaSetDeviceFlags = NULL;
static cudaSetValidDevices_t native_cudaSetValidDevices = NULL;
static cudaStreamAttachMemAsync_t native_cudaStreamAttachMemAsync = NULL;
static cudaStreamCreate_t native_cudaStreamCreate = NULL;
static cudaStreamCreateWithFlags_t native_cudaStreamCreateWithFlags = NULL;
static cudaStreamCreateWithPriority_t native_cudaStreamCreateWithPriority = NULL;
static cudaStreamDestroy_t native_cudaStreamDestroy = NULL;
static cudaStreamGetFlags_t native_cudaStreamGetFlags= NULL;
static cudaStreamGetPriority_t native_cudaStreamGetPriority = NULL;
static cudaStreamQuery_t native_cudaStreamQuery = NULL;
static cudaStreamSynchronize_t native_cudaStreamSynchronize = NULL;
static cudaStreamWaitEvent_t native_cudaStreamWaitEvent = NULL;
static cudaEventCreate_t native_cudaEventCreate = NULL;
static cudaEventCreateWithFlags_t native_cudaEventCreateWithFlags = NULL;
static cudaEventDestroy_t native_cudaEventDestroy = NULL;
static cudaEventElapsedTime_t native_cudaEventElapsedTime = NULL;
static cudaEventQuery_t native_cudaEventQuery = NULL;
static cudaEventRecord_t native_cudaEventRecord = NULL;
static cudaEventSynchronize_t native_cudaEventSynchronize = NULL;
static cudaConfigureCall_t native_CudaConfigureCall = NULL;
static cudaFuncGetAttributes_t native_cudaFuncGetAttributes = NULL;
static cudaFuncSetAttribute_t native_cudaFuncSetAttribute = NULL;
static cudaLaunch_t native_cudaLaunch = NULL;
static cudaFuncSetCacheConfig_t native_cudaFuncSetCacheConfig = NULL;
static cudaFuncSetSharedMemConfig_t native_cudaFuncSetSharedMemConfig = NULL;
static cudaGetParameterBuffer_t native_cudaGetParameterBuffer = NULL;
static cudaGetParameterBufferV2_t native_cudaGetParameterBufferV2 = NULL;
static cudaLaunchCooperativeKernel_t native_cudaLaunchCooperativeKernel = NULL;
static cudaLaunchCooperativeKernelMultiDevice_t native_cudaLaunchCooperativeKernelMultiDevice = NULL;
static cudaLaunchKernel_t native_cudaLaunchKernel = NULL;
static cudaSetDoubleForDevice_t native_cudaSetDoubleForDevice = NULL;
static cudaSetDoubleForHost_t native_cudaSetDoubleForHost = NULL;
static cudaSetupArgument_t native_CudaSetupArgument = NULL;
static cudaFree_t native_cudaFree = NULL;
static cudaFreeArray_t native_cudaFreeArray = NULL;
static cudaFreeHost_t native_cudaFreeHost = NULL;
static cudaGetSymbolAddress_t native_cudaGetSymbolAddress = NULL;
static cudaGetSymbolSize_t native_cudaGetSymbolSize = NULL;
static cudaHostAlloc_t native_cudaHostAlloc = NULL;
static cudaHostGetDevicePointer_t native_cudaHostGetDevicePointer = NULL;
static cudaHostGetFlags_t native_cudaHostGetFlags = NULL;
static cudaMalloc_t native_cudaMalloc = NULL;
static cudaMalloc3D_t native_cudaMalloc3D = NULL;
static cudaMalloc3DArray_t native_cudaMalloc3DArray = NULL;
static cudaMallocArray_t native_cudaMallocArray = NULL;
static cudaMallocHost_t native_cudaMallocHost = NULL;
static cudaMallocPitch_t native_cudaMallocPitch = NULL;
static cudaMemcpy_t native_cudaMemcpy = NULL;
static cudaMemcpy2D_t native_cudaMemcpy2D= NULL;
static cudaMemcpy2DArrayToArray_t native_cudaMemcpy2DArrayToArray = NULL;
static cudaMemcpy2DAsync_t native_cudaMemcpy2DAsync = NULL;
static cudaMemcpy2DFromArray_t native_cudaMemcpy2DFromArray = NULL;
static cudaMemcpy2DFromArrayAsync_t native_cudaMemcpy2DFromArrayAsync = NULL;
static cudaMemcpy2DToArray_t native_cudaMemcpy2DToArray= NULL;
static cudaMemcpy2DToArrayAsync_t native_cudaMemcpy2DToArrayAsync = NULL;
static cudaMemcpy3D_t native_cudaMemcpy3D = NULL;
static cudaMemcpy3DAsync_t native_cudaMemcpy3DAsync = NULL;
static cudaMemcpyArrayToArray_t native_cudaMemcpyArrayToArray = NULL;
static cudaMemcpyAsync_t native_cudaMemcpyAsync = NULL;
static cudaMemcpyFromArray_t native_cudaMemcpyFromArray = NULL;
static cudaMemcpyFromArrayAsync_t native_cudaMemcpyFromArrayAsync = NULL;
static cudaMemcpyFromSymbol_t native_cudaMemcpyFromSymbol = NULL;
static cudaMemcpyFromSymbolAsync_t native_cudaMemcpyFromSymbolAsync = NULL;
static cudaMemcpyToArray_t native_cudaMemcpyToArray = NULL;
static cudaMemcpyToArrayAsync_t native_cudaMemcpyToArrayAsync = NULL;
static cudaMemcpyToSymbol_t native_cudaMemcpyToSymbol = NULL;
static cudaMemcpyToSymbolAsync_t native_cudaMemcpyToSymbolAsync = NULL;
static cudaMemset_t native_cudaMemset = NULL;
static cudaMemset2D_t native_cudaMemset2D = NULL;
static cudaMemset3D_t native_cudaMemset3D = NULL;
static cudaDriverGetVersion_t native_cudaDriverGetVersion = NULL;
static cudaRuntimeGetVersion_t native_cudaRuntimeGetVersion = NULL;
static cudaThreadExit_t native_cudaThreadExit = NULL;
static cudaThreadSynchronize_t native_cudaThreadSynchronize = NULL;

//**********************************************//
//      Native endpoint pointer initialization  //
//**********************************************//
///   init native endpoints   ///
__attribute__((constructor))
static void initializae_NativeCudaApi() {
    assert((native_cudaGetErrorName = (cudaGetErrorName_t)dlsym(RTLD_NEXT,"cudaGetErrorName")) != NULL);
    assert((native_cudaGetErrorString = (cudaGetErrorString_t)dlsym(RTLD_NEXT,"cudaGetErrorString")) != NULL);
    assert((native_cudaGetLastError = (cudaGetLastError_t)dlsym(RTLD_NEXT,"cudaGetLastError")) != NULL);
    assert((native_cudaPeekAtLastError = (cudaPeekAtLastError_t)dlsym(RTLD_NEXT,"cudaPeekAtLastError")) != NULL);
    assert((native_cudaChooseDevice = (cudaChooseDevice_t)dlsym(RTLD_NEXT,"cudaChooseDevice")) != NULL);
    assert((native_cudaDeviceGetAttribute = (cudaDeviceGetAttribute_t)dlsym(RTLD_NEXT,"cudaDeviceGetAttribute")) != NULL);
    assert((native_cudaDeviceGetByPCIBusId  = (cudaDeviceGetByPCIBusId_t)dlsym(RTLD_NEXT,"cudaDeviceGetByPCIBusId ")) != NULL);
    assert((native_cudaDeviceGetCacheConfig = (cudaDeviceGetCacheConfig_t)dlsym(RTLD_NEXT,"cudaDeviceGetCacheConfig")) != NULL);
    assert((native_cudaDeviceGetLimit = (cudaDeviceGetLimit_t)dlsym(RTLD_NEXT,"cudaDeviceGetLimit")) != NULL);
    assert((native_cudaDeviceGetNvSciSyncAttributes= (cudaDeviceGetNvSciSyncAttributes_t)dlsym(RTLD_NEXT,"cudaDeviceGetNvSciSyncAttributes")) != NULL);
    assert((native_cudaDeviceGetP2PAttribute = (cudaDeviceGetP2PAttribute_t)dlsym(RTLD_NEXT,"cudaDeviceGetP2PAttribute")) != NULL);
    assert((native_cudaDeviceGetPCIBusId = (cudaDeviceGetPCIBusId_t)dlsym(RTLD_NEXT,"cudaDeviceGetPCIBusId")) != NULL);
    assert((native_cudaDeviceGetSharedMemConfig = (cudaDeviceGetSharedMemConfig_t)dlsym(RTLD_NEXT,"cudaDeviceGetSharedMemConfig")) != NULL);
    assert((native_cudaDeviceGetStreamPriorityRange = (cudaDeviceGetStreamPriorityRange_t)dlsym(RTLD_NEXT,"cudaDeviceGetStreamPriorityRange")) != NULL);
    assert((native_cudaDeviceSetCacheConfig = (cudaDeviceSetCacheConfig_t)dlsym(RTLD_NEXT,"cudaDeviceSetCacheConfig")) != NULL);
    assert((native_cudaDeviceSetLimit = (cudaDeviceSetLimit_t)dlsym(RTLD_NEXT,"cudaDeviceSetLimit")) != NULL);
    assert((native_cudaDeviceSetSharedMemConfig = (cudaDeviceSetSharedMemConfig_t)dlsym(RTLD_NEXT,"cudaDeviceSetSharedMemConfig")) != NULL);
    assert((native_cudaDeviceSynchronize = (cudaDeviceSynchronize_t)dlsym(RTLD_NEXT,"cudaDeviceSynchronize")) != NULL);
    assert((native_cudaGetDevice = (cudaGetDevice_t)dlsym(RTLD_NEXT,"cudaGetDevice")) != NULL);
    assert((native_cudaGetDeviceCount = (cudaGetDeviceCount_t)dlsym(RTLD_NEXT,"cudaGetDeviceCount")) != NULL);
    assert((native_cudaGetDeviceFlags = (cudaGetDeviceFlags_t)dlsym(RTLD_NEXT,"cudaGetDeviceFlags")) != NULL);
    assert((native_cudaGetDeviceProperties = (cudaGetDeviceProperties_t)dlsym(RTLD_NEXT,"cudaGetDeviceProperties")) != NULL);
    assert((native_cudaIpcCloseMemHandle= (cudaIpcCloseMemHandle_t)dlsym(RTLD_NEXT,"cudaIpcCloseMemHandle")) != NULL);
    assert((native_cudaIpcGetEventHandle = (cudaIpcGetEventHandle_t)dlsym(RTLD_NEXT,"cudaIpcGetEventHandle")) != NULL);
    assert((native_cudaIpcGetMemHandle = (cudaIpcGetMemHandle_t)dlsym(RTLD_NEXT,"cudaIpcGetMemHandle")) != NULL);
    assert((native_cudaIpcOpenEventHandle = (cudaIpcOpenEventHandle_t)dlsym(RTLD_NEXT,"cudaIpcOpenEventHandle")) != NULL);
    assert((native_cudaIpcOpenMemHandle = (cudaIpcOpenMemHandle_t)dlsym(RTLD_NEXT,"cudaIpcOpenMemHandle")) != NULL);
    assert((native_cudaSetDevice = (cudaSetDevice_t)dlsym(RTLD_NEXT,"cudaSetDevice")) != NULL);
    assert((native_cudaSetDeviceFlags = (cudaSetDeviceFlags_t)dlsym(RTLD_NEXT,"cudaSetDeviceFlags")) != NULL);
    assert((native_cudaSetValidDevices = (cudaSetValidDevices_t)dlsym(RTLD_NEXT,"cudaSetValidDevices")) != NULL);
    assert((native_cudaStreamAttachMemAsync = (cudaStreamAttachMemAsync_t)dlsym(RTLD_NEXT,"cudaStreamAttachMemAsync")) != NULL);
    assert((native_cudaStreamCreate = (cudaStreamCreate_t)dlsym(RTLD_NEXT,"cudaStreamCreate")) != NULL);
    assert((native_cudaStreamCreateWithFlags = (cudaStreamCreateWithFlags_t)dlsym(RTLD_NEXT,"cudaStreamCreateWithFlags")) != NULL);
    assert((native_cudaStreamCreateWithPriority = (cudaStreamCreateWithPriority_t)dlsym(RTLD_NEXT,"cudaStreamCreateWithPriority")) != NULL);
    assert((native_cudaStreamDestroy = (cudaStreamDestroy_t)dlsym(RTLD_NEXT,"cudaStreamDestroy")) != NULL);
    assert((native_cudaStreamGetFlags = (cudaStreamGetFlags_t)dlsym(RTLD_NEXT,"cudaStreamGetFlags")) != NULL);
    assert((native_cudaStreamGetPriority = (cudaStreamGetPriority_t)dlsym(RTLD_NEXT,"cudaStreamGetPriority")) != NULL);
    assert((native_cudaStreamQuery = (cudaStreamQuery_t)dlsym(RTLD_NEXT,"cudaStreamQuery")) != NULL);
    assert((native_cudaStreamSynchronize = (cudaStreamSynchronize_t)dlsym(RTLD_NEXT,"cudaStreamSynchronize")) != NULL);
    assert((native_cudaStreamWaitEvent = (cudaStreamWaitEvent_t)dlsym(RTLD_NEXT,"cudaStreamWaitEvent")) != NULL);
    assert((native_cudaEventCreate = (cudaEventCreate_t)dlsym(RTLD_NEXT,"cudaEventCreate")) != NULL);
    assert((native_cudaEventCreateWithFlags = (cudaEventCreateWithFlags_t)dlsym(RTLD_NEXT,"cudaEventCreateWithFlags")) != NULL);
    assert((native_cudaEventDestroy = (cudaEventDestroy_t)dlsym(RTLD_NEXT,"cudaEventDestroy")) != NULL);
    assert((native_cudaEventElapsedTime = (cudaEventElapsedTime_t)dlsym(RTLD_NEXT,"cudaEventElapsedTime")) != NULL);
    assert((native_cudaEventQuery = (cudaEventQuery_t)dlsym(RTLD_NEXT,"cudaEventQuery")) != NULL);
    assert((native_cudaEventRecord = (cudaEventRecord_t)dlsym(RTLD_NEXT,"cudaEventRecord")) != NULL);
    assert((native_cudaEventSynchronize = (cudaEventSynchronize_t)dlsym(RTLD_NEXT,"cudaEventSynchronize")) != NULL);
    assert((native_CudaConfigureCall = (cudaConfigureCall_t)dlsym(RTLD_NEXT,"cudaConfigureCall")) != NULL);
    assert((native_cudaFuncGetAttributes = (cudaFuncGetAttributes_t)dlsym(RTLD_NEXT,"cudaFuncGetAttributes")) != NULL);
    assert((native_cudaFuncSetAttribute = (cudaFuncSetAttribute_t)dlsym(RTLD_NEXT,"cudaFuncSetAttribute")) != NULL);
    assert((native_cudaLaunch = (cudaLaunch_t)dlsym(RTLD_NEXT,"cudaLaunch")) != NULL);
    assert((native_cudaFuncSetCacheConfig = (cudaFuncSetCacheConfig_t)dlsym(RTLD_NEXT,"cudaFuncSetCacheConfig")) != NULL);
    assert((native_cudaFuncSetSharedMemConfig = (cudaFuncSetSharedMemConfig_t)dlsym(RTLD_NEXT,"cudaFuncSetSharedMemConfig")) != NULL);
    assert((native_cudaGetParameterBuffer = (cudaGetParameterBuffer_t)dlsym(RTLD_NEXT,"cudaGetParameterBuffer")) != NULL);
    assert((native_cudaGetParameterBufferV2 = (cudaGetParameterBufferV2_t)dlsym(RTLD_NEXT,"cudaGetParameterBufferV2")) != NULL);
    assert((native_cudaLaunchCooperativeKernel = (cudaLaunchCooperativeKernel_t)dlsym(RTLD_NEXT,"cudaLaunchCooperativeKernel")) != NULL);
    assert((native_cudaLaunchCooperativeKernelMultiDevice = (cudaLaunchCooperativeKernelMultiDevice_t)dlsym(RTLD_NEXT,"cudaLaunchCooperativeKernelMultiDevice")) != NULL);
    assert((native_cudaLaunchKernel = (cudaLaunchKernel_t)dlsym(RTLD_NEXT,"cudaLaunchKernel")) != NULL);
    assert((native_cudaSetDoubleForDevice = (cudaSetDoubleForDevice_t)dlsym(RTLD_NEXT,"cudaSetDoubleForDevice")) != NULL);
    assert((native_cudaSetDoubleForHost = (cudaSetDoubleForHost_t)dlsym(RTLD_NEXT,"cudaSetDoubleForHost")) != NULL);
    assert((native_CudaSetupArgument = (cudaSetupArgument_t)dlsym(RTLD_NEXT,"cudaSetupArgument")) != NULL);
    assert((native_cudaFree = (cudaFree_t)dlsym(RTLD_NEXT,"cudaFree")) != NULL);
    assert((native_cudaFreeArray = (cudaFreeArray_t)dlsym(RTLD_NEXT,"cudaFreeArray")) != NULL);
    assert((native_cudaFreeHost = (cudaFreeHost_t)dlsym(RTLD_NEXT,"cudaFreeHost")) != NULL);
    assert((native_cudaGetSymbolAddress = (cudaGetSymbolAddress_t)dlsym(RTLD_NEXT,"cudaGetSymbolAddress")) != NULL);
    assert((native_cudaGetSymbolSize = (cudaGetSymbolSize_t)dlsym(RTLD_NEXT,"cudaGetSymbolSize")) != NULL);
    assert((native_cudaHostAlloc = (cudaHostAlloc_t)dlsym(RTLD_NEXT,"cudaHostAlloc")) != NULL);
    assert((native_cudaHostGetDevicePointer = (cudaHostGetDevicePointer_t)dlsym(RTLD_NEXT,"cudaHostGetDevicePointer")) != NULL);
    assert((native_cudaHostGetFlags = (cudaHostGetFlags_t)dlsym(RTLD_NEXT,"cudaHostGetFlags")) != NULL);
    assert((native_cudaMalloc = (cudaMalloc_t)dlsym(RTLD_NEXT,"cudaMalloc")) != NULL);
    assert((native_cudaMalloc3D = (cudaMalloc3D_t)dlsym(RTLD_NEXT,"cudaMalloc3D")) != NULL);
    assert((native_cudaMalloc3DArray = (cudaMalloc3DArray_t)dlsym(RTLD_NEXT,"cudaMalloc3DArray")) != NULL);
    assert((native_cudaMallocArray = (cudaMallocArray_t)dlsym(RTLD_NEXT,"cudaMallocArray")) != NULL);
    assert((native_cudaMallocHost = (cudaMallocHost_t)dlsym(RTLD_NEXT,"cudaMallocHost")) != NULL);
    assert((native_cudaMallocPitch = (cudaMallocPitch_t)dlsym(RTLD_NEXT,"cudaMallocPitch")) != NULL);
    assert((native_cudaMemcpy = (cudaMemcpy_t)dlsym(RTLD_NEXT,"cudaMemcpy")) != NULL);
    assert((native_cudaMemcpy2D = (cudaMemcpy2D_t)dlsym(RTLD_NEXT,"cudaMemcpy2D")) != NULL);
    assert((native_cudaMemcpy2DArrayToArray = (cudaMemcpy2DArrayToArray_t)dlsym(RTLD_NEXT,"cudaMemcpy2DArrayToArray")) != NULL);
    assert((native_cudaMemcpy2DAsync = (cudaMemcpy2DAsync_t)dlsym(RTLD_NEXT,"cudaMemcpy2DAsync")) != NULL);
    assert((native_cudaMemcpy2DFromArray = (cudaMemcpy2DFromArray_t)dlsym(RTLD_NEXT,"cudaMemcpy2DFromArray")) != NULL);
    assert((native_cudaMemcpy2DFromArrayAsync = (cudaMemcpy2DFromArrayAsync_t)dlsym(RTLD_NEXT,"cudaMemcpy2DFromArrayAsync")) != NULL);
    assert((native_cudaMemcpy2DToArray = (cudaMemcpy2DToArray_t)dlsym(RTLD_NEXT,"cudaMemcpy2DToArray")) != NULL);
    assert((native_cudaMemcpy2DToArrayAsync = (cudaMemcpy2DToArrayAsync_t)dlsym(RTLD_NEXT,"cudaMemcpy2DToArrayAsync")) != NULL);
    assert((native_cudaMemcpy3D = (cudaMemcpy3D_t)dlsym(RTLD_NEXT,"cudaMemcpy3D")) != NULL);
    assert((native_cudaMemcpy3DAsync = (cudaMemcpy3DAsync_t)dlsym(RTLD_NEXT,"cudaMemcpy3DAsync")) != NULL);
    assert((native_cudaMemcpyArrayToArray = (cudaMemcpyArrayToArray_t)dlsym(RTLD_NEXT,"cudaMemcpyArrayToArray")) != NULL);
    assert((native_cudaMemcpyAsync = (cudaMemcpyAsync_t)dlsym(RTLD_NEXT,"cudaMemcpyAsync")) != NULL);
    assert((native_cudaMemcpyFromArray = (cudaMemcpyFromArray_t)dlsym(RTLD_NEXT,"cudaMemcpyFromArray")) != NULL);
    assert((native_cudaMemcpyFromArrayAsync = (cudaMemcpyFromArrayAsync_t)dlsym(RTLD_NEXT,"cudaMemcpyFromArrayAsync")) != NULL);
    assert((native_cudaMemcpyFromSymbol = (cudaMemcpyFromSymbol_t)dlsym(RTLD_NEXT,"cudaMemcpyFromSymbol")) != NULL);
    assert((native_cudaMemcpyFromSymbolAsync = (cudaMemcpyFromSymbolAsync_t)dlsym(RTLD_NEXT,"cudaMemcpyFromSymbolAsync")) != NULL);
    assert((native_cudaMemcpyToArray = (cudaMemcpyToArray_t)dlsym(RTLD_NEXT,"cudaMemcpyToArray")) != NULL);
    assert((native_cudaMemcpyToArrayAsync = (cudaMemcpyToArrayAsync_t)dlsym(RTLD_NEXT,"cudaMemcpyToArrayAsync")) != NULL);
    assert((native_cudaMemcpyToSymbol = (cudaMemcpyToSymbol_t)dlsym(RTLD_NEXT,"cudaMemcpyToSymbol")) != NULL);
    assert((native_cudaMemcpyToSymbolAsync = (cudaMemcpyToSymbolAsync_t)dlsym(RTLD_NEXT,"cudaMemcpyToSymbolAsync")) != NULL);
    assert((native_cudaMemset = (cudaMemset_t)dlsym(RTLD_NEXT,"cudaMemset")) != NULL);
    assert((native_cudaMemset2D = (cudaMemset2D_t)dlsym(RTLD_NEXT,"cudaMemset2D")) != NULL);
    assert((native_cudaMemset3D = (cudaMemset3D_t)dlsym(RTLD_NEXT,"cudaMemset3D")) != NULL);
    assert((native_cudaDriverGetVersion = (cudaDriverGetVersion_t)dlsym(RTLD_NEXT,"cudaDriverGetVersion")) != NULL);
    assert((native_cudaRuntimeGetVersion = (cudaRuntimeGetVersion_t)dlsym(RTLD_NEXT,"cudaRuntimeGetVersion")) != NULL);
    assert((native_cudaThreadExit = (cudaThreadExit_t)dlsym(RTLD_NEXT,"cudaThreadExit")) != NULL);
    assert((native_cudaThreadSynchronize = (cudaThreadSynchronize_t)dlsym(RTLD_NEXT,"cudaThreadSynchronize")) != NULL);
}

////////////////////////////
//   CALLS INTERCEPTION   //
////////////////////////////

//*******************************************//
//      CUDA Runtime API Error Handling      //
//*******************************************//
///   cudaGetErrorName   ///
extern "C" const char* cudaGetErrorName(cudaError_t error) {
    CUDA_DEBUG_MSG("\n>> cudaGetErrorName interception\n");
    return native_cudaGetErrorName(error);
}

///   cudaGetErrorString   ///
extern "C" const char* cudaGetErrorString(cudaError_t error) {
    CUDA_DEBUG_MSG("\n>> cudaGetErrorString interception\n");
    return native_cudaGetErrorString(error);
}

///   cudaGetLastError   ///
extern "C" cudaError_t cudaGetLastError(void) {
    CUDA_DEBUG_MSG("\n>> cudaGetLastError interception\n");
    return native_cudaGetLastError();
}

///   cudaGetLastError   ///
extern "C" cudaError_t cudaPeekAtLastError(void) {
    CUDA_DEBUG_MSG("\n>> cudaPeekAtLastError interception\n");
    return native_cudaPeekAtLastError();
}

//**********************************************//
//      CUDA Runtime API Device Management      //
//**********************************************//
///   cudaChooseDevice   ///
extern "C" cudaError_t cudaChooseDevice(int * device, const struct cudaDeviceProp * prop) {
    CUDA_DEBUG_MSG("\n>>cudaChooseDevice interception \n");
    return native_cudaChooseDevice(device,prop);
}

///   cudaDeviceGetAttribute   ///
extern "C" cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
    CUDA_DEBUG_MSG("\n>>cudaDeviceGetAttribute interception \n");
    return native_cudaDeviceGetAttribute(value,attr,device);
}

///   cudaDeviceGetByPCIBusId    ///
extern "C" cudaError_t cudaDeviceGetByPCIBusId  (int* device, const char* pciBusId) {
    CUDA_DEBUG_MSG("\n>>cudaDeviceGetByPCIBusId  interception\n");
    return native_cudaDeviceGetByPCIBusId (device,pciBusId);
}

///   cudaDeviceGetCacheConfig   ///
extern "C" cudaError_t cudaDeviceGetCacheConfig (cudaFuncCache ** pCacheConfig) {
    CUDA_DEBUG_MSG("\n>>cudaDeviceGetCacheConfig interception\n");
    return native_cudaDeviceGetCacheConfig(pCacheConfig);
}

///   cudaDeviceGetLimit   ///
extern "C" cudaError_t cudaDeviceGetLimit (size_t* pValue, cudaLimit limit) {
    CUDA_DEBUG_MSG("\n>>cudaDeviceGetLimit interception\n");
    return native_cudaDeviceGetLimit(pValue,limit);
}

///   cudaDeviceGetNvSciSyncAttributes   ///
extern "C" cudaError_t cudaDeviceGetNvSciSyncAttributes ( void* nvSciSyncAttrList, int device, int flags) {
    CUDA_DEBUG_MSG("\n>>cudaDeviceGetNvSciSyncAttributes interception\n");
    return native_cudaDeviceGetNvSciSyncAttributes(nvSciSyncAttrList,device,flags);
}

///   cudaDeviceGetP2PAttribute   ///
extern "C" cudaError_t cudaDeviceGetP2PAttribute (int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) {
    CUDA_DEBUG_MSG("\n>>cudaDeviceGetP2PAttribute interception\n");
    return native_cudaDeviceGetP2PAttribute(value,attr,srcDevice,dstDevice);
}

///   cudaDeviceGetPCIBusId   ///
extern "C" cudaError_t cudaDeviceGetPCIBusId (char* pciBusId, int len, int device) {
    CUDA_DEBUG_MSG("\n>>cudaDeviceGetPCIBusId interception\n");
    return native_cudaDeviceGetPCIBusId(pciBusId,len,device);
}

///   cudaDeviceGetSharedMemConfig   ///
extern "C" cudaError_t cudaDeviceGetSharedMemConfig (cudaSharedMemConfig ** pConfig ) {
    CUDA_DEBUG_MSG("\n>>cudaDeviceGetSharedMemConfig interception\n");
    return native_cudaDeviceGetSharedMemConfig(pConfig);
}

///   cudaDeviceGetStreamPriorityRange   ///
extern "C" cudaError_t cudaDeviceGetStreamPriorityRange ( int* leastPriority, int* greatestPriority) {
    CUDA_DEBUG_MSG("\n>>cudaDeviceGetStreamPriorityRange interception\n");
    return native_cudaDeviceGetStreamPriorityRange(leastPriority,greatestPriority);
}

///   cudaMalloc3D   ///
extern "C" cudaError_t cudaDeviceSetCacheConfig (cudaFuncCache cacheConfig) {
    CUDA_DEBUG_MSG("\n>>cudaDeviceSetCacheConfig interception\n");
    return native_cudaDeviceSetCacheConfig(cacheConfig);
}

///   cudaDeviceSetLimit   ///
extern "C" cudaError_t cudaDeviceSetLimit (cudaLimit limit, size_t value) {
    CUDA_DEBUG_MSG("\n>>cudaDeviceSetLimit interception\n");
    return native_cudaDeviceSetLimit(limit,value);
}

///   cudaDeviceSetSharedMemConfig   ///
extern "C" cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config) {
    CUDA_DEBUG_MSG("\n>>cudaDeviceSetSharedMemConfig interception\n");
    return native_cudaDeviceSetSharedMemConfig(config);
}

///   cudaDeviceSynchronize   ///
extern "C" cudaError_t cudaDeviceSynchronize (void) {
    CUDA_DEBUG_MSG("\n>>cudaDeviceSynchronize interception\n");
    return native_cudaDeviceSynchronize();
}

///   cudaGetDevice   ///
extern "C" cudaError_t cudaGetDevice(int *device){
    CUDA_DEBUG_MSG("\n>>cudaGetDevice \n");
    return native_cudaGetDevice(device);
}

///   cudaGetDeviceCount   ///
extern "C" cudaError_t cudaGetDeviceCount(int * count){
    CUDA_DEBUG_MSG("\n>>cudaGetDeviceCount interception \n");
    return native_cudaGetDeviceCount(count);
}

///   cudaGetDeviceFlags   ///
extern "C" cudaError_t cudaGetDeviceFlags (unsigned int* flags) {
    CUDA_DEBUG_MSG("\n>>cudaGetDeviceFlags interception\n");
    return native_cudaGetDeviceFlags(flags);
}

///   cudaGetDeviceProperties   ///
extern "C" cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp * prop, int device){
    CUDA_DEBUG_MSG("\n>>cudaGetDeviceProperties interception \n");
    return native_cudaGetDeviceProperties(prop,device);
}

///   cudaIpcCloseMemHandle   ///
extern "C" cudaError_t cudaIpcCloseMemHandle (void* devPtr) {
    CUDA_DEBUG_MSG("\n>>cudaIpcCloseMemHandle interception\n");
    return native_cudaIpcCloseMemHandle(devPtr);
}

///   cudaIpcGetEventHandle   ///
extern "C" cudaError_t cudaIpcGetEventHandle (cudaIpcEventHandle_t* handle, cudaEvent_t event) {
    CUDA_DEBUG_MSG("\n>>cudaIpcGetEventHandle interception\n");
    return native_cudaIpcGetEventHandle(handle,event);
}

///   cudaIpcGetMemHandle   ///
extern "C" cudaError_t cudaIpcGetMemHandle (cudaIpcMemHandle_t* handle, void* devPtr) {
    CUDA_DEBUG_MSG("\n>>cudaIpcGetMemHandle interception\n");
    return native_cudaIpcGetMemHandle(handle,devPtr);
}

///   cudaIpcOpenEventHandle   ///
extern "C" cudaError_t cudaIpcOpenEventHandle (cudaEvent_t* event, cudaIpcEventHandle_t handle) {
    CUDA_DEBUG_MSG("\n>>cudaIpcOpenEventHandle interception\n");
    return native_cudaIpcOpenEventHandle(event,handle);
}

///   cudaIpcOpenMemHandle   ///
extern "C" cudaError_t cudaIpcOpenMemHandle (void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) {
    CUDA_DEBUG_MSG("\n>>cudaIpcOpenMemHandle interception\n");
    return native_cudaIpcOpenMemHandle(devPtr,handle,flags);
}

///   cudaSetDevice   ///
extern "C" cudaError_t cudaSetDevice(int device){
    CUDA_DEBUG_MSG("\n>>cudaSetDevice interception \n");
    return native_cudaSetDevice(device);
}

///   cudaSetDeviceFlags   ///
extern "C" cudaError_t cudaSetDeviceFlags(int flags){
    CUDA_DEBUG_MSG("\n>>cudaSetDeviceFlags interception \n");
    return native_cudaSetDeviceFlags(flags);
}

///   cudaSetValidDevices   ///
extern "C" cudaError_t cudaSetValidDevices(int * device_arr, int len){
    CUDA_DEBUG_MSG("\n>>cudaSetValidDevices interception \n");
    return native_cudaSetValidDevices(device_arr,len);
}

//**********************************************//
//      CUDA Runtime API Stream Management      //
//**********************************************//
///   cudaStreamAttachMemAsync   ///
extern "C" cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags){
    CUDA_DEBUG_MSG("\n>>cudaStreamAttachMemAsync interception \n");
    return native_cudaStreamAttachMemAsync(stream,devPtr,length,flags);
}


///   cudaStreamCreate   ///
extern "C" cudaError_t cudaStreamCreate(cudaStream_t * pStream){
    CUDA_DEBUG_MSG("\n>>cudaStreamCreate interception \n");
    return native_cudaStreamCreate(pStream);
}

///   cudaStreamCreateWithFlags   ///
extern "C" cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int  flags){
    CUDA_DEBUG_MSG("\n>>cudaStreamCreateWithFlags interception \n");
    return native_cudaStreamCreateWithFlags(pStream,flags);
}

///   cudaStreamCreateWithPriority   ///
extern "C" cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority){
    CUDA_DEBUG_MSG("\n>>cudaStreamCreateWithPriority interception \n");
    return native_cudaStreamCreateWithPriority(pStream,flags,priority);
}

///   cudaStreamDestroy   ///
extern "C" cudaError_t cudaStreamDestroy(cudaStream_t stream){
    CUDA_DEBUG_MSG("\n>>cudaStreamDestroy interception \n");
    return native_cudaStreamDestroy(stream);
}

///   cudaStreamGetFlags   ///
extern "C" cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags){
    CUDA_DEBUG_MSG("\n>>cudaStreamGetFlags interception \n");
    return native_cudaStreamGetFlags(hStream,flags);
}

///   cudaStreamGetPriority   ///
extern "C" cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority){
    CUDA_DEBUG_MSG("\n>>cudaStreamGetPriority interception \n");
    return native_cudaStreamGetPriority(hStream,priority);
}

///   cudaStreamQuery   ///
extern "C" cudaError_t cudaStreamQuery(cudaStream_t stream){
    CUDA_DEBUG_MSG("\n>>cudaStreamQuery interception \n");
    return native_cudaStreamQuery(stream);
}

///   cudaStreamSynchronize   ///
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream){
    CUDA_DEBUG_MSG("\n>>cudaStreamSynchronize interception \n");
    return native_cudaStreamSynchronize(stream);
}

///   cudaStreamWaitEvent   ///
extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags){
    CUDA_DEBUG_MSG("\n>>cudaStreamWaitEvent interception \n");
    return native_cudaStreamWaitEvent(stream,event,flags);
}

//*********************************************//
//      CUDA Runtime API Event Management      //
//*********************************************//
///   cudaDriverGetVersion   ///
extern "C" cudaError_t cudaEventCreate (cudaEvent_t * event) {
    CUDA_DEBUG_MSG("\n>>cudaEventCreate interception\n");
    return native_cudaEventCreate(event);
}

///   cudaEventCreateWithFlags   ///
extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, int flags) {
    CUDA_DEBUG_MSG("\n>>cudaEventCreateWithFlags interception\n");
    return native_cudaEventCreateWithFlags(event,flags);
}

///   cudaEventDestroy   ///
extern "C" cudaError_t cudaEventDestroy	(cudaEvent_t event) {
    CUDA_DEBUG_MSG("\n>>cudaEventDestroy interception\n");
    return native_cudaEventDestroy(event);
}

///   cudaEventElapsedTime   ///
extern "C" cudaError_t cudaEventElapsedTime	(float * ms, cudaEvent_t start,cudaEvent_t end) {
    CUDA_DEBUG_MSG("\n>>cudaEventElapsedTime interception\n");
    return native_cudaEventElapsedTime(ms,start,end);
}

///   cudaEventQuery   ///
extern "C" cudaError_t cudaEventQuery (cudaEvent_t event) {
    CUDA_DEBUG_MSG("\n>>cudaEventQuery interception\n");
    return native_cudaEventQuery(event);
}

///   cudaEventRecord   ///
extern "C" cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    CUDA_DEBUG_MSG("\n>>cudaEventRecord interception\n");
    return native_cudaEventRecord(event,stream);
}

///   cudaEventSynchronize   ///
extern "C" cudaError_t cudaEventSynchronize	(cudaEvent_t event) {
    CUDA_DEBUG_MSG("\n>>cudaEventSynchronize interception\n");
    return native_cudaEventSynchronize(event);
}


//**********************************************//
//      CUDA Runtime API Execution Control      //
//**********************************************//
//  cudaConfigureCall  /// 
extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem=0, cudaStream_t stream=0) {
    CUDA_DEBUG_MSG("\n>>cudaConfigureCall interception\n");
    assert(kernelInfo().counter == 0);
    kernelInfo().gridDim = gridDim;
    kernelInfo().blockDim = blockDim;
    //kernelInfo().counter++;   //increase a counter to indicate an expected cudaLaunch to be completed
    return native_CudaConfigureCall(gridDim,blockDim,sharedMem,stream);
}

///   cudaFuncGetAttributes   ///
extern "C" cudaError_t cudaFuncGetAttributes (struct cudaFuncAttributes * attr, const char * func) {
    CUDA_DEBUG_MSG("\n>>cudaFuncGetAttributes interception\n");
    return native_cudaFuncGetAttributes(attr,func);
}

///   cudaFuncSetAttribute   ///
extern "C" cudaError_t cudaFuncSetAttribute (const void* func, cudaFuncAttribute attr, int  value) {
    CUDA_DEBUG_MSG("\n>>cudaFuncSetAttribute interception\n");
    return native_cudaFuncSetAttribute(func,attr,value);
}

///  cudaLaunch ///
extern "C" cudaError_t cudaLaunch( const char* entry){
    CUDA_DEBUG_MSG("\n>>cudaLaunch interception\n");
    return native_cudaLaunch(entry);
}


///   cudaFuncSetCacheConfig   ///
extern "C" cudaError_t cudaFuncSetCacheConfig (const void* func, cudaFuncCache cacheConfig) {
    CUDA_DEBUG_MSG("\n>>cudaFuncSetCacheConfig interception\n");
    return native_cudaFuncSetCacheConfig(func,cacheConfig);
}

///   cudaFuncSetSharedMemConfig   ///
extern "C" cudaError_t cudaFuncSetSharedMemConfig (const void* func, cudaSharedMemConfig config) {
    CUDA_DEBUG_MSG("\n>>cudaFuncSetSharedMemConfig interception\n");
    return native_cudaFuncSetSharedMemConfig(func,config);
}

///   cudaGetParameterBuffer   ///
extern "C" cudaError_t cudaGetParameterBuffer (size_t alignment, size_t size) {
    CUDA_DEBUG_MSG("\n>>cudaGetParameterBuffer interception\n");
    return native_cudaGetParameterBuffer(alignment,size);
}

///   cudaGetParameterBufferV2   ///
extern "C" cudaError_t cudaGetParameterBufferV2	(void* func, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize) {
    CUDA_DEBUG_MSG("\n>>cudaGetParameterBufferV2 interception\n");
    return native_cudaGetParameterBufferV2(func,gridDimension,blockDimension,sharedMemSize);
}

///   cudaLaunchCooperativeKernel   ///
extern "C" cudaError_t cudaLaunchCooperativeKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    CUDA_DEBUG_MSG("\n>>cudaLaunchCooperativeKernel interception\n");
    return native_cudaLaunchCooperativeKernel(func,gridDim,blockDim,args,sharedMem,stream);
}

///   cudaLaunchCooperativeKernelMultiDevice   ///
extern "C" cudaError_t cudaLaunchCooperativeKernelMultiDevice (cudaLaunchParams* launchParamsList, unsigned int numDevices, unsigned int flags) {
    CUDA_DEBUG_MSG("\n>>cudaLaunchCooperativeKernelMultiDevice interception\n");
    return native_cudaLaunchCooperativeKernelMultiDevice(launchParamsList,numDevices,flags);
}

///   cudaLaunchKernel   ///
extern "C" cudaError_t cudaLaunchKernel	(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    CUDA_DEBUG_MSG("\n>>cudaLaunchKernel interception\n");
    return native_cudaLaunchKernel(func,gridDim,blockDim,args,sharedMem,stream);
}

///   cudaSetDoubleForDevice   ///
extern "C" cudaError_t cudaSetDoubleForDevice (double *d) {
    CUDA_DEBUG_MSG("\n>>cudaSetDoubleForDevice interception\n");
    return native_cudaSetDoubleForDevice(d);
}

///   cudaSetDoubleForHost   ///
extern "C" cudaError_t cudaSetDoubleForHost	(double *d) {
    CUDA_DEBUG_MSG("\n>>cudaSetDoubleForHost interception\n");
    return native_cudaSetDoubleForHost(d);
}

//**********************************************//
//      CUDA Runtime API Memory Management      //
//**********************************************//
///   cudaFree   ///
extern "C" cudaError_t cudaFree	(void * devPtr) {
    CUDA_DEBUG_MSG("\n>>cudaFree interception\n");
    return native_cudaFree(devPtr);
}


///   cudaFreeArray   ///
extern "C" cudaError_t cudaFreeArray (struct cudaArray * array) {
    CUDA_DEBUG_MSG("\n>>cudaFreeArray interception\n");
    return native_cudaFreeArray(array);
}


///   cudaFreeHost   ///
extern "C" cudaError_t cudaFreeHost(void * ptr) {
    CUDA_DEBUG_MSG("\n>>cudaFreeHost interception\n");
    return native_cudaFreeHost(ptr);
}


///   cudaGetSymbolAddress   ///
extern "C" cudaError_t cudaGetSymbolAddress	(void ** devPtr, const char * symbol) {
    CUDA_DEBUG_MSG("\n>>cudaGetSymbolAddress interception\n");
    return native_cudaGetSymbolAddress(devPtr,symbol);
}


///   cudaGetSymbolSize   ///
extern "C" cudaError_t cudaGetSymbolSize(size_t * size, const char * symbol) {
    CUDA_DEBUG_MSG("\n>>cudaGetSymbolSize interception\n");
    return native_cudaGetSymbolSize(size,symbol);
}


///   cudaHostAlloc   ///
extern "C" cudaError_t cudaHostAlloc (void ** ptr, size_t size, unsigned int flags) {
    CUDA_DEBUG_MSG("\n>>cudaHostAlloc interception\n");
    return native_cudaHostAlloc(ptr,size,flags);
}


///   cudaHostGetDevicePointer   ///
extern "C" cudaError_t cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned int flags) {
    CUDA_DEBUG_MSG("\n>>cudaHostGetDevicePointer interception\n");
    return native_cudaHostGetDevicePointer(pDevice,pHost,flags);
}


///   cudaHostGetFlags   ///
extern "C" cudaError_t cudaHostGetFlags(unsigned int * pFlags, void * pHost) {
    CUDA_DEBUG_MSG("\n>>cudaHostGetFlags interception\n");
    return native_cudaHostGetFlags(pFlags,pHost);
}


///   cudaMalloc   ///
extern "C" cudaError_t cudaMalloc(void ** devPtr, size_t size) {
    CUDA_DEBUG_MSG("\n>>cudaMalloc interception\n");
    return native_cudaMalloc(devPtr,size);
}


///   cudaMalloc3D   ///
extern "C" cudaError_t cudaMalloc3D (struct cudaPitchedPtr * pitchedDevPtr, struct cudaExtent extent) {
    CUDA_DEBUG_MSG("\n>>cudaMalloc3D interception\n");
    return native_cudaMalloc3D(pitchedDevPtr,extent);
}


///   cudaMalloc3DArray   ///
extern "C" cudaError_t cudaMalloc3DArray (struct cudaArray ** arrayPtr, const struct cudaChannelFormatDesc * desc, struct cudaExtent extent) {
    CUDA_DEBUG_MSG("\n>>cudaMalloc3DArray interception\n");
    return native_cudaMalloc3DArray(arrayPtr,desc,extent);
}


///   cudaMallocArray   ///
extern "C" cudaError_t cudaMallocArray (struct cudaArray ** arrayPtr, const struct cudaChannelFormatDesc * desc, size_t width, size_t height) {
    CUDA_DEBUG_MSG("\n>>cudaMallocArray interception\n");
    return native_cudaMallocArray(arrayPtr,desc,width,height);
}


///   cudaMallocHost   ///
extern "C" cudaError_t cudaMallocHost (void ** ptr,size_t size) {
    CUDA_DEBUG_MSG("\n>>cudaMallocHost interception\n");
    return native_cudaMallocHost(ptr,size);
}


///   cudaMallocPitch   ///
extern "C" cudaError_t cudaMallocPitch (void ** devPtr, size_t * pitch, size_t width, size_t height) {
    CUDA_DEBUG_MSG("\n>>cudaMallocPitch interception\n");
    return native_cudaMallocPitch(devPtr,pitch,width,height);
}


///   cudaMemcpy   ///
extern "C" cudaError_t cudaMemcpy (void * dst, const void * src, size_t count, enum cudaMemcpyKind kind) {
    CUDA_DEBUG_MSG("\n>>cudaMemcpy interception\n");
    return native_cudaMemcpy(dst,src,count,kind);
}


///   cudaMemcpy2D   ///
extern "C" cudaError_t cudaMemcpy2D (void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
    CUDA_DEBUG_MSG("\n>>cudaMemcpy2D interception\n");
    return native_cudaMemcpy2D(dst,dpitch,src,spitch,width,height,kind);
}


///   cudaMemcpy2DArrayToArray   ///
extern "C" cudaError_t cudaMemcpy2DArrayToArray (struct cudaArray * dst,
    size_t 	wOffsetDst,
    size_t 	hOffsetDst,
    const struct cudaArray * src,
    size_t 	wOffsetSrc,
    size_t 	hOffsetSrc,
    size_t 	width,
    size_t 	height,
    enum cudaMemcpyKind kind) {
    CUDA_DEBUG_MSG("\n>>cudaMalloc3D interception\n");
    return native_cudaMemcpy2DArrayToArray(dst,wOffsetDst,hOffsetDst,src,wOffsetSrc,hOffsetSrc,width,height,kind);
}

///   cudaMemcpy2DAsync   ///
extern "C" cudaError_t cudaMemcpy2DAsync (void * dst,
    size_t 	dpitch,
    const void * src,
    size_t 	spitch,
    size_t 	width,
    size_t 	height,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {
    CUDA_DEBUG_MSG("\n>>cudaMemcpy2DAsync interception\n");
    return native_cudaMemcpy2DAsync(dst,dpitch,src,spitch,width,height,kind,stream);
}


///   cudaMemcpy2DFromArray   ///
extern "C" cudaError_t cudaMemcpy2DFromArray (void * dst,
    size_t 	dpitch,
    const struct cudaArray * src,
    size_t 	wOffset,
    size_t 	hOffset,
    size_t 	width,
    size_t 	height,
    enum cudaMemcpyKind kind){
    CUDA_DEBUG_MSG("\n>>cudaMemcpy2DFromArray interception\n");
    return native_cudaMemcpy2DFromArray(dst,dpitch,src,wOffset,hOffset,width,height,kind);
}



///   cudaMemcpy2DFromArrayAsync   ///
extern "C" cudaError_t cudaMemcpy2DFromArrayAsync (void * dst,
    size_t 	dpitch,
    const struct cudaArray * src,
    size_t 	wOffset,
    size_t 	hOffset,
    size_t 	width,
    size_t 	height,
    enum cudaMemcpyKind kind,
    cudaStream_t stream){
    CUDA_DEBUG_MSG("\n>>cudaMemcpy2DFromArrayAsync interception\n");
    return native_cudaMemcpy2DFromArrayAsync(dst,dpitch,src,wOffset,hOffset,width,height,kind,stream);
}

///   cudaMemcpy2DToArray   ///
extern "C" cudaError_t cudaMemcpy2DToArray (struct cudaArray * dst,
    size_t 	wOffset,
    size_t 	hOffset,
    const void * src,
    size_t 	spitch,
    size_t 	width,
    size_t 	height,
    enum cudaMemcpyKind kind) {
    CUDA_DEBUG_MSG("\n>>cudaMemcpy2DToArray interception\n");
    return native_cudaMemcpy2DToArray(dst,wOffset,hOffset,src,spitch,width,height,kind);
}


///   cudaMemcpy2DToArrayAsync   ///
extern "C" cudaError_t cudaMemcpy2DToArrayAsync (struct cudaArray * dst,
    size_t 	wOffset,
    size_t 	hOffset,
    const void * src,
    size_t 	spitch,
    size_t 	width,
    size_t 	height,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {
    CUDA_DEBUG_MSG("\n>>cudaMemcpy2DToArrayAsync interception\n");
    return native_cudaMemcpy2DToArrayAsync(dst,wOffset,hOffset,src,spitch,width,height,kind,stream);
}


///   cudaMemcpy3D   ///
extern "C" cudaError_t cudaMemcpy3D (const struct cudaMemcpy3DParms * p) {
    CUDA_DEBUG_MSG("\n>>cudaMemcpy3D interception\n");
    return native_cudaMemcpy3D(p);
}


///   cudaMemcpy3DAsync   ///
extern "C" cudaError_t cudaMemcpy3DAsync (const struct cudaMemcpy3DParms * p, cudaStream_t stream) {
    CUDA_DEBUG_MSG("\n>>cudaMemcpy3DAsync interception\n");
    return native_cudaMemcpy3DAsync(p,stream);
}

///   cudaMemcpyArrayToArray   ///
extern "C" cudaError_t cudaMemcpyArrayToArray(struct cudaArray * dst,
    size_t 	wOffsetDst,
    size_t 	hOffsetDst,
    const struct cudaArray * src,
    size_t 	wOffsetSrc,
    size_t 	hOffsetSrc,
    size_t 	count,
    enum cudaMemcpyKind kind){
    CUDA_DEBUG_MSG("\n>>cudaMemcpyArrayToArray interception\n");
    return native_cudaMemcpyArrayToArray(dst,wOffsetDst,hOffsetDst,src,wOffsetSrc,hOffsetSrc,count,kind);
}


///   cudaMemcpyAsync   ///
extern "C" cudaError_t cudaMemcpyAsync (void * dst,
    const void * src,
    size_t 	count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {
    CUDA_DEBUG_MSG("\n>>cudaMemcpyAsync interception\n");
    return native_cudaMemcpyAsync(dst,src,count,kind,stream);
}


///   cudaMemcpyFromArray   ///
extern "C" cudaError_t cudaMemcpyFromArray (void * dst,
    const struct cudaArray * src,
    size_t wOffset,
    size_t hOffset,
    size_t count,
    enum cudaMemcpyKind kind){
    CUDA_DEBUG_MSG("\n>>cudaMemcpyFromArray interception\n");
    return native_cudaMemcpyFromArray(dst,src,wOffset,hOffset,count,kind);
}


///   cudaMemcpyFromArrayAsync   ///
extern "C" cudaError_t cudaMemcpyFromArrayAsync (void * dst,
    const struct cudaArray * src,
    size_t 	wOffset,
    size_t 	hOffset,
    size_t 	count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream){
    CUDA_DEBUG_MSG("\n>>cudaMemcpyFromArrayAsync interception\n");
    return native_cudaMemcpyFromArrayAsync(dst,src,wOffset,hOffset,count,kind,stream);
}


///   cudaMemcpyFromSymbol   ///
extern "C" cudaError_t cudaMemcpyFromSymbol (void * dst,
    const char * symbol,
    size_t 	count,
    size_t 	offset,
    enum cudaMemcpyKind kind) {
    CUDA_DEBUG_MSG("\n>>cudaMemcpyFromSymbol interception\n");
    return native_cudaMemcpyFromSymbol(dst,symbol,count,offset,kind);
}

///   cudaMemcpyFromSymbolAsync   ///
extern "C" cudaError_t cudaMemcpyFromSymbolAsync (void * dst,
    const char * symbol,
    size_t 	count,
    size_t 	offset,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {
    CUDA_DEBUG_MSG("\n>>cudaMemcpyFromSymbolAsync interception\n");
    return native_cudaMemcpyFromSymbolAsync(dst,symbol,count,offset,kind,stream);
}

///   cudaMemcpyToArray   ///
extern "C" cudaError_t cudaMemcpyToArray (struct cudaArray * dst,
    size_t 	wOffset,
    size_t 	hOffset,
    const void * src,
    size_t 	count,
    enum cudaMemcpyKind kind) {
    CUDA_DEBUG_MSG("\n>>cudaMemcpyToArray interception\n");
    return native_cudaMemcpyToArray(dst,wOffset,hOffset,src,count,kind);
}

///   cudaMemcpyToArrayAsync   ///
extern "C" cudaError_t cudaMemcpyToArrayAsync (struct cudaArray * dst,
    size_t 	wOffset,
    size_t 	hOffset,
    const void * src,
    size_t 	count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {
    CUDA_DEBUG_MSG("\n>>cudaMemcpyToArrayAsync interception\n");
    return native_cudaMemcpyToArrayAsync(dst,wOffset,hOffset,src,count,kind,stream);
}

///   cudaMemcpyToSymbol   ///
extern "C" cudaError_t cudaMemcpyToSymbol (const char * symbol,
    const void * src,
    size_t 	count,
    size_t 	offset,
    enum cudaMemcpyKind kind) {
    CUDA_DEBUG_MSG("\n>>cudaMemcpyToSymbol interception\n");
    return native_cudaMemcpyToSymbol(symbol,src,count,offset,kind);
}

///   cudaMemcpyToSymbolAsync   ///
extern "C" cudaError_t cudaMemcpyToSymbolAsync (const char * symbol,
    const void * src,
    size_t 	count,
    size_t 	offset,
    enum cudaMemcpyKind kind,
    cudaStream_t stream) {
    CUDA_DEBUG_MSG("\n>>cudaMemcpyToSymbolAsync interception\n");
    return native_cudaMemcpyToSymbolAsync(symbol,src,count,offset,kind,stream);
}

///   cudaMemset   ///
extern "C" cudaError_t cudaMemset(void * devPtr, int value, size_t count) {
    CUDA_DEBUG_MSG("\n>>cudaMemset interception\n");
    return native_cudaMemset(devPtr,value,count);
}

///   cudaMemset2D   ///
extern "C" cudaError_t cudaMemset2D (void * devPtr,
    size_t  pitch,
    int     value,
    size_t 	width,
    size_t 	height) {
    CUDA_DEBUG_MSG("\n>>cudaMemset2D interception\n");
    return native_cudaMemset2D(devPtr,pitch,value,width,height);
}

///   cudaMemset3D   ///
extern "C" cudaError_t cudaMemset3D (struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent) {
    CUDA_DEBUG_MSG("\n>>cudaMemset3D interception\n");
    return native_cudaMemset3D(pitchedDevPtr,value,extent);
}


//***********************************************//
//      CUDA Runtime API Version Management      //
//***********************************************//
///   cudaDriverGetVersion   ///
extern "C" cudaError_t cudaDriverGetVersion	(int * driverVersion) {
    CUDA_DEBUG_MSG("\ncudaDriverGetVersion interception\n");
    return native_cudaDriverGetVersion(driverVersion);
}

///   cudaDriverGetVersion   ///
extern "C" cudaError_t cudaRuntimeGetVersion(int * runtimeVersion) {
    CUDA_DEBUG_MSG("\ncudaRuntimeGetVersion interception\n");
    return native_cudaRuntimeGetVersion(runtimeVersion);
}


//**********************************************//
//      CUDA Runtime API Thread Management      //
//**********************************************//
///   cudaThreadExit   ///
extern "C" cudaError_t cudaThreadExit(void) {
    CUDA_DEBUG_MSG("\n>>cudaThreadExit interception\n");
    return native_cudaThreadExit();
}

///   cudaThreadExit   ///
extern "C" cudaError_t cudaThreadSynchronize(void) {
    CUDA_DEBUG_MSG("\n>>cudaThreadSynchronize interception\n");
    return native_cudaThreadSynchronize();
}
