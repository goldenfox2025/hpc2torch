
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <iostream>

// -------------------- (A) 通用 gather kernel (非 axis=0) -------------------- //
template<typename T>
__global__ void gather_kernel(
    const T* __restrict__ input,
    const int* __restrict__ index,
    T* __restrict__ output,
    const int* input_shape, int input_rank,
    const int* index_shape, int index_rank,
    const int* out_shape, int out_rank,
    int outer,
    int axis,
    int start_offset 
) {
    __shared__ int s_out_strides[32];
    __shared__ int s_idx_strides[32];
    __shared__ int s_in_strides[32];

    if (threadIdx.x == 0) {
        s_out_strides[out_rank - 1] = 1;
        for (int d = out_rank - 2; d >= 0; d--) {
            s_out_strides[d] = s_out_strides[d + 1] * out_shape[d + 1];
        }
        if (index_rank > 0) {
            s_idx_strides[index_rank - 1] = 1;
            for (int d = index_rank - 2; d >= 0; d--) {
                s_idx_strides[d] = s_idx_strides[d + 1] * index_shape[d + 1];
            }
        }
        s_in_strides[input_rank - 1] = 1;
        for (int d = input_rank - 2; d >= 0; d--) {
            s_in_strides[d] = s_in_strides[d + 1] * input_shape[d + 1];
        }
    }
    __syncthreads();

    int tid = start_offset + blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer) return;

    int A = axis;
    int K = index_rank;
    int tmp = tid;
    int input_offset_first = 0;
    int idx_linear = 0;
    int input_offset_last = 0;
    for (int d = 0; d < out_rank; d++) {
        int idx_d = tmp / s_out_strides[d];
        tmp = tmp % s_out_strides[d];
        if (d < A)
            input_offset_first += idx_d * s_in_strides[d];
        else if (d < A + K)
            idx_linear += idx_d * s_idx_strides[d - A];
        else
            input_offset_last += idx_d * s_in_strides[d - K + 1];
    }
    int gathered = __ldg(index + idx_linear);
    int in_linear = input_offset_first + gathered * s_in_strides[A] + input_offset_last;
    output[tid] = __ldg(input + in_linear);
}


// -------------------- (B) axis=0 向量化内核 -------------------- //
// 模板参数：
//   T 为标量类型（例如 float 或 __half），
//   VecT 为向量类型（例如 float4 或 half2），
//   VEC_SIZE 为加载个数（例如 4 或 2）。
template<typename T, typename VecT, int VEC_SIZE>
__global__ void super_gather(
    const T* __restrict__ input,
    const int* __restrict__ index,
    T* __restrict__ output,
    const int* input_shape, int input_rank,
    const int* index_shape, int index_rank, 
    int total   
) {
    __shared__ int s_sliceSize;
    if (threadIdx.x == 0) {
        int sSize = 1;
        for (int d = 1; d < input_rank; d++) {
            sSize *= input_shape[d];
        }
        s_sliceSize = sSize;
    }
    __syncthreads();
    int sliceSize = s_sliceSize;
    int total_vec = total / VEC_SIZE;
    int remainder = total % VEC_SIZE;
    int globalVecId = blockIdx.x * blockDim.x + threadIdx.x;
    bool isRemainderThread = false;
    int loadCount = VEC_SIZE;
    if (globalVecId == total_vec && remainder > 0) {
        isRemainderThread = true;
        loadCount = remainder;
    }
    if (!isRemainderThread && globalVecId < total_vec) {
        int base = globalVecId * VEC_SIZE;  
        int sliceId = base / sliceSize;
        int offsetInSlice = base % sliceSize;
        int gatherIndex = __ldg(index + sliceId);
        int inPos = gatherIndex * sliceSize + offsetInSlice;
        if ((offsetInSlice + VEC_SIZE <= sliceSize)) {
            const VecT* inputVec = reinterpret_cast<const VecT*>(input);
            VecT* outputVec = reinterpret_cast<VecT*>(output);
            int vecIndex = inPos / VEC_SIZE;  
            outputVec[globalVecId] = __ldg(inputVec + vecIndex);
            return; 
        }
        else {
            loadCount = VEC_SIZE;
        }
    }
    else if (!isRemainderThread) {
        return;
    }
    int baseIndex = (!isRemainderThread) ? (globalVecId * VEC_SIZE) : (total_vec * VEC_SIZE);
    for (int i = 0; i < loadCount; i++) {
        int outIdx = baseIndex + i;
        if (outIdx >= total) break;
        int sliceId = outIdx / sliceSize;
        int offsetInSlice = outIdx % sliceSize;
        int gatherIndex = __ldg(index + sliceId);
        int inPos = gatherIndex * sliceSize + offsetInSlice;
        output[outIdx] = __ldg(input + inPos);
    }
}


// -------------------- (C) Kernel Launcher 入口 -------------------- //

void gather_cuda_kernel_launcher(
    const void* input,
    const void* index,
    void* output,
    int dtype_code,   // 用 onnx.TensorProto.* 的枚举值表示
    const int* input_shape, int input_rank, 
    const int* index_shape, int index_rank,
    const int* out_shape, int out_rank,
    int outer,        // 总输出标量数
    int axis
) {
    if (axis == 0) {
        // fp32 和 fp16 使用向量化读取
        // 这是优化版本
        if (dtype_code == 1 /* onnx.TensorProto.FLOAT */) {
            int total = outer;
            dim3 block(1024);
            int total_vec = total / 4;
            int extra = (total % 4) ? 1 : 0;
            int totalThreads = total_vec + extra;
            dim3 grid((totalThreads + block.x - 1) / block.x);
            super_gather<float, float4, 4>
                <<<grid, block>>>( static_cast<const float*>(input),
                                    static_cast<const int*>(index),
                                    static_cast<float*>(output),
                                    input_shape, input_rank,
                                    index_shape, index_rank,
                                    total );
        }
        else if (dtype_code == 10 /* onnx.TensorProto.FLOAT16 */) {
            int total = outer;
            dim3 block(128);
            int total_vec = total / 2;
            int extra = (total % 2) ? 1 : 0;
            int totalThreads = total_vec + extra;
            dim3 grid((totalThreads + block.x - 1) / block.x);
            super_gather<__half, half2, 2>
                <<<grid, block>>>( static_cast<const __half*>(input),
                                    static_cast<const int*>(index),
                                    static_cast<__half*>(output),
                                    input_shape, input_rank,
                                    index_shape, index_rank,
                                    total );
        }
        else if (dtype_code == 2 /* onnx.TensorProto.UINT8 */ ||
                 dtype_code == 3 /* onnx.TensorProto.INT8 */   ||
                 dtype_code == 4 /* onnx.TensorProto.UINT16 */  ||
                 dtype_code == 5 /* onnx.TensorProto.INT16 */   ||
                 dtype_code == 6 /* onnx.TensorProto.INT32 */   ||
                 dtype_code == 7 /* onnx.TensorProto.INT64 */   ||
                 dtype_code == 11 /* onnx.TensorProto.DOUBLE */  ||
                 dtype_code == 12 /* onnx.TensorProto.UINT32 */  ||
                 dtype_code == 13 /* onnx.TensorProto.UINT64 */ ) {
            dim3 block(256);
            dim3 grid((outer + block.x - 1) / block.x);
            // 根据 dtype_code 将模板参数 T 替换为对应的类型
            // 例如，对于 UINT8：unsigned char, INT8: char, UINT16: unsigned short, INT16: short,
            // INT32: int, INT64: long long, DOUBLE: double, UINT32: unsigned int, UINT64: unsigned long long.
            if (dtype_code == 2) {  // UINT8
                gather_kernel<unsigned char>
                    <<<grid, block>>>( static_cast<const unsigned char*>(input),
                                         static_cast<const int*>(index),
                                         static_cast<unsigned char*>(output),
                                         input_shape, input_rank,
                                         index_shape, index_rank,
                                         out_shape, out_rank,
                                         outer,
                                         axis,
                                         0 );
            } else if (dtype_code == 3) { // INT8
                gather_kernel<char>
                    <<<grid, block>>>( static_cast<const char*>(input),
                                         static_cast<const int*>(index),
                                         static_cast<char*>(output),
                                         input_shape, input_rank,
                                         index_shape, index_rank,
                                         out_shape, out_rank,
                                         outer,
                                         axis,
                                         0 );
            } else if (dtype_code == 4) { // UINT16
                gather_kernel<unsigned short>
                    <<<grid, block>>>( static_cast<const unsigned short*>(input),
                                         static_cast<const int*>(index),
                                         static_cast<unsigned short*>(output),
                                         input_shape, input_rank,
                                         index_shape, index_rank,
                                         out_shape, out_rank,
                                         outer,
                                         axis,
                                         0 );
            } else if (dtype_code == 5) { // INT16
                gather_kernel<short>
                    <<<grid, block>>>( static_cast<const short*>(input),
                                         static_cast<const int*>(index),
                                         static_cast<short*>(output),
                                         input_shape, input_rank,
                                         index_shape, index_rank,
                                         out_shape, out_rank,
                                         outer,
                                         axis,
                                         0 );
            } else if (dtype_code == 6) { // INT32
                gather_kernel<int>
                    <<<grid, block>>>( static_cast<const int*>(input),
                                         static_cast<const int*>(index),
                                         static_cast<int*>(output),
                                         input_shape, input_rank,
                                         index_shape, index_rank,
                                         out_shape, out_rank,
                                         outer,
                                         axis,
                                         0 );
            } else if (dtype_code == 7) { // INT64
                gather_kernel<long long>
                    <<<grid, block>>>( static_cast<const long long*>(input),
                                         static_cast<const int*>(index),
                                         static_cast<long long*>(output),
                                         input_shape, input_rank,
                                         index_shape, index_rank,
                                         out_shape, out_rank,
                                         outer,
                                         axis,
                                         0 );
            } else if (dtype_code == 11) { // DOUBLE
                gather_kernel<double>
                    <<<grid, block>>>( static_cast<const double*>(input),
                                         static_cast<const int*>(index),
                                         static_cast<double*>(output),
                                         input_shape, input_rank,
                                         index_shape, index_rank,
                                         out_shape, out_rank,
                                         outer,
                                         axis,
                                         0 );
            } else if (dtype_code == 12) { // UINT32
                gather_kernel<unsigned int>
                    <<<grid, block>>>( static_cast<const unsigned int*>(input),
                                         static_cast<const int*>(index),
                                         static_cast<unsigned int*>(output),
                                         input_shape, input_rank,
                                         index_shape, index_rank,
                                         out_shape, out_rank,
                                         outer,
                                         axis,
                                         0 );
            } else if (dtype_code == 13) { // UINT64
                gather_kernel<unsigned long long>
                    <<<grid, block>>>( static_cast<const unsigned long long*>(input),
                                         static_cast<const int*>(index),
                                         static_cast<unsigned long long*>(output),
                                         input_shape, input_rank,
                                         index_shape, index_rank,
                                         out_shape, out_rank,
                                         outer,
                                         axis,
                                         0 );
            }
        }
        else {
            std::cerr << "[Error] Unsupported or unimplemented dtype_code: " << dtype_code << std::endl;
            return;
        }
    }
    else {  // axis != 0
        dim3 block(256);
        dim3 grid((outer + block.x - 1) / block.x);
        if (dtype_code == 10 /* onnx.TensorProto.FLOAT16 */) {
            gather_kernel<__half>
                <<<grid, block>>>( static_cast<const __half*>(input),
                                     static_cast<const int*>(index),
                                     static_cast<__half*>(output),
                                     input_shape, input_rank,
                                     index_shape, index_rank,
                                     out_shape, out_rank,
                                     outer,
                                     axis,
                                     0 );
        }
        else if (dtype_code == 1 /* onnx.TensorProto.FLOAT */) {
            gather_kernel<float>
                <<<grid, block>>>( static_cast<const float*>(input),
                                     static_cast<const int*>(index),
                                     static_cast<float*>(output),
                                     input_shape, input_rank,
                                     index_shape, index_rank,
                                     out_shape, out_rank,
                                     outer,
                                     axis,
                                     0 );
        }
        else if (dtype_code == 2 /* UINT8 */) {
            gather_kernel<unsigned char>
                <<<grid, block>>>( static_cast<const unsigned char*>(input),
                                     static_cast<const int*>(index),
                                     static_cast<unsigned char*>(output),
                                     input_shape, input_rank,
                                     index_shape, index_rank,
                                     out_shape, out_rank,
                                     outer,
                                     axis,
                                     0 );
        }
        else if (dtype_code == 3 /* INT8 */) {
            gather_kernel<char>
                <<<grid, block>>>( static_cast<const char*>(input),
                                     static_cast<const int*>(index),
                                     static_cast<char*>(output),
                                     input_shape, input_rank,
                                     index_shape, index_rank,
                                     out_shape, out_rank,
                                     outer,
                                     axis,
                                     0 );
        }
        else if (dtype_code == 4 /* UINT16 */) {
            gather_kernel<unsigned short>
                <<<grid, block>>>( static_cast<const unsigned short*>(input),
                                     static_cast<const int*>(index),
                                     static_cast<unsigned short*>(output),
                                     input_shape, input_rank,
                                     index_shape, index_rank,
                                     out_shape, out_rank,
                                     outer,
                                     axis,
                                     0 );
        }
        else if (dtype_code == 5 /* INT16 */) {
            gather_kernel<short>
                <<<grid, block>>>( static_cast<const short*>(input),
                                     static_cast<const int*>(index),
                                     static_cast<short*>(output),
                                     input_shape, input_rank,
                                     index_shape, index_rank,
                                     out_shape, out_rank,
                                     outer,
                                     axis,
                                     0 );
        }
        else if (dtype_code == 6 /* INT32 */) {
            gather_kernel<int>
                <<<grid, block>>>( static_cast<const int*>(input),
                                     static_cast<const int*>(index),
                                     static_cast<int*>(output),
                                     input_shape, input_rank,
                                     index_shape, index_rank,
                                     out_shape, out_rank,
                                     outer,
                                     axis,
                                     0 );
        }
        else if (dtype_code == 7 /* INT64 */) {
            gather_kernel<long long>
                <<<grid, block>>>( static_cast<const long long*>(input),
                                     static_cast<const int*>(index),
                                     static_cast<long long*>(output),
                                     input_shape, input_rank,
                                     index_shape, index_rank,
                                     out_shape, out_rank,
                                     outer,
                                     axis,
                                     0 );
        }
        else if (dtype_code == 11 /* DOUBLE */) {
            gather_kernel<double>
                <<<grid, block>>>( static_cast<const double*>(input),
                                     static_cast<const int*>(index),
                                     static_cast<double*>(output),
                                     input_shape, input_rank,
                                     index_shape, index_rank,
                                     out_shape, out_rank,
                                     outer,
                                     axis,
                                     0 );
        }
        else if (dtype_code == 12 /* UINT32 */) {
            gather_kernel<unsigned int>
                <<<grid, block>>>( static_cast<const unsigned int*>(input),
                                     static_cast<const int*>(index),
                                     static_cast<unsigned int*>(output),
                                     input_shape, input_rank,
                                     index_shape, index_rank,
                                     out_shape, out_rank,
                                     outer,
                                     axis,
                                     0 );
        }
        else if (dtype_code == 13 /* UINT64 */) {
            gather_kernel<unsigned long long>
                <<<grid, block>>>( static_cast<const unsigned long long*>(input),
                                     static_cast<const int*>(index),
                                     static_cast<unsigned long long*>(output),
                                     input_shape, input_rank,
                                     index_shape, index_rank,
                                     out_shape, out_rank,
                                     outer,
                                     axis,
                                     0 );
        }
        else {
            std::cerr << "[Error] Unsupported dtype_code: " << dtype_code << std::endl;
            return;
        }
    }
    
}


extern "C" {
void gather_cuda_wrapper(
    void* input,
    void* index,
    void* output,
    int dtype_code,
    const void* input_shape, int input_rank,
    const void* index_shape, int index_rank,
    const void* out_shape, int out_rank,
    int outer,
    int axis = 0
) {
    gather_cuda_kernel_launcher(
        input, index, output,
        dtype_code,
        static_cast<const int*>(input_shape), input_rank,
        static_cast<const int*>(index_shape), index_rank,
        static_cast<const int*>(out_shape), out_rank,
        outer,
        axis
    );
}
} 
