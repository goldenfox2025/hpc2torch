import torch
import ctypes
import numpy as np
from functools import partial
import argparse

import performance
# 添加上一层目录到模块搜索路径
import sys
import os
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)


lib.gather_cuda_wrapper.argtypes = [
    ctypes.c_void_p,  # input数据指针
    ctypes.c_void_p,  # index数据指针
    ctypes.c_void_p,  # output数据指针
    ctypes.c_int,     # dtype_code (0 = float32, 1 = float16)
    ctypes.c_void_p,  # input_shape（设备内存指针）
    ctypes.c_int,     # input_rank
    ctypes.c_void_p,  # index_shape（设备内存指针）
    ctypes.c_int,     # index_rank
    ctypes.c_void_p,  # out_shape（设备内存指针）
    ctypes.c_int,     # out_rank
    ctypes.c_int,     # outer  —— 输出张量总元素数（预先计算好）
    ctypes.c_int      # axis
]
lib.gather_cuda_wrapper.restype = None



def gather(rank, axis, inputTensor, indexTensor):
    indices = [slice(None)] * rank
    indices[axis] = indexTensor
    outTensor = inputTensor[tuple(indices)]
    return outTensor

def custom_gather(inputTensor: torch.Tensor,
                  indexTensor: torch.Tensor,
                  out_shape: list,
                  in_shape_tensor: torch.Tensor,
                  idx_shape_tensor: torch.Tensor,
                  out_shape_tensor: torch.Tensor,
                  outer: int,
                  rank: int,
                  idx_rank: int,
                  out_rank: int,
                  axis: int,
                  dtype_code: int):
    # 参数有效性检查（可根据需要保留）
    if not (0 <= axis < inputTensor.dim()):
        raise ValueError(f"Invalid axis {axis}, must be in [0, {inputTensor.dim()-1}]")

    # 直接使用预计算的输出形状创建张量
    outTensor = torch.empty(
        out_shape,
        device=inputTensor.device,
        dtype=inputTensor.dtype
    )

    # 调用CUDA算子（所有参数已预处理）
    lib.gather_cuda_wrapper(
        ctypes.c_void_p(inputTensor.data_ptr()),
        ctypes.c_void_p(indexTensor.data_ptr()),
        ctypes.c_void_p(outTensor.data_ptr()),
        ctypes.c_int(dtype_code),
        ctypes.c_void_p(in_shape_tensor.data_ptr()),
        ctypes.c_int(rank),
        ctypes.c_void_p(idx_shape_tensor.data_ptr()),
        ctypes.c_int(idx_rank),
        ctypes.c_void_p(out_shape_tensor.data_ptr()),
        ctypes.c_int(out_rank),
        ctypes.c_int(outer),
        ctypes.c_int(axis)
    )
    return outTensor

def test(inputShape, indexShape, axis, test_dtype, device):
    print(f"Testing Softmax on {device} with x_shape:{inputShape}, indice_shape:{indexShape}, axis:{axis}, dtype:{test_dtype}")

    # 生成输入数据
    inputTensor = torch.rand(inputShape, device=device, dtype=test_dtype)
    index = np.random.randint(0, inputShape[axis], indexShape).astype(np.int32)
    indexTensor = torch.from_numpy(index).to(device=device, dtype=torch.int32).contiguous()

    # 预处理所有参数
    rank = len(inputShape)
    idx_rank = len(indexShape)
    in_shape = list(inputTensor.shape)
    idx_shape = list(indexTensor.shape)
    out_shape = in_shape[:axis] + idx_shape + in_shape[axis+1:]
    out_rank = len(out_shape)
    outer = int(np.prod(out_shape))

    # 创建形状张量（需确保设备一致性）
    in_shape_tensor = torch.tensor(in_shape, dtype=torch.int32, device=device)
    idx_shape_tensor = torch.tensor(idx_shape, dtype=torch.int32, device=device)
    out_shape_tensor = torch.tensor(out_shape, dtype=torch.int32, device=device)

    # 确定数据类型编码
    dtype_code = 1 if test_dtype == torch.float32 else 10

    # 执行原始gather函数
    torch_out = gather(rank, axis, inputTensor, indexTensor)
    
    # 执行优化后的custom_gather
    custom_out = custom_gather(
        inputTensor=inputTensor,
        indexTensor=indexTensor,
        out_shape=out_shape,
        in_shape_tensor=in_shape_tensor,
        idx_shape_tensor=idx_shape_tensor,
        out_shape_tensor=out_shape_tensor,
        outer=outer,
        rank=rank,
        idx_rank=idx_rank,
        out_rank=out_rank,
        axis=axis,
        dtype_code=dtype_code
    )

    # 性能测试
    if test_dtype in [torch.float32, torch.float16] and device == "cuda":
        torch_time = performance.CudaProfile((gather, (rank, axis, inputTensor, indexTensor)))
        custom_time = performance.CudaProfile((
            custom_gather,
            (inputTensor, indexTensor, out_shape, in_shape_tensor, idx_shape_tensor,
             out_shape_tensor, outer, rank, idx_rank, out_rank, axis, dtype_code)
        ))
        performance.logBenchmark(torch_time, custom_time)

    # 精度验证
    torch_np = torch_out.cpu().numpy().flatten()
    custom_np = custom_out.cpu().numpy().flatten()
    abs_err = np.max(np.abs(torch_np - custom_np))
    rel_err = abs_err / (np.max(np.abs(custom_np)) + 1e-8)
    print(f"Absolute Error: {abs_err:.4e}")
    print(f"Relative Error: {rel_err:.4e}\n")
parser = argparse.ArgumentParser(description="Test softmax on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    
test_cases = [
        # inputShape , indexShape, axis, test_dtype, device
        ((3, 2), (2, 2), 0, torch.float32, "cuda"),
        ((3, 2), (1, 2), 1, torch.float32, "cuda"),
        ((50257, 768), (16, 1024), 0, torch.float32, "cuda"),

                                

        ((3, 2), (2, 2), 0, torch.float16, "cuda"),
        ((3, 2), (1, 2), 1, torch.float16, "cuda"),
        ((50257, 768), (16, 1024), 0, torch.float16, "cuda"),

         
]
filtered_test_cases = [
    (inputShape , indexShape, axis, test_dtype, device)
    for inputShape , indexShape, axis, test_dtype, device in test_cases
    if device == args.device
]
if args.device == 'mlu':
    import torch_mlu
# 执行过滤后的测试用例
for inputShape , indexShape, axis, test_dtype, device in filtered_test_cases:
    test(inputShape , indexShape, axis, test_dtype, device)