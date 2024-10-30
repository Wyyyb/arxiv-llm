#!/bin/bash


source /gpfs/public/research/miniconda3/bin/activate
conda activate cite_rag

# 创建并执行Python脚本
python3 << EOF
import torch
import time

# 确保CUDA可用
if not torch.cuda.is_available():
    print("CUDA is not available")
    exit(1)

# 分配100MB显存 (100 * 1024 * 1024 字节)
# 使用float32类型(4字节)，所以需要除以4来计算元素数量
size = int(100 * 1024 * 1024 / 4)
tensor = torch.zeros(size, device='cuda')

# 持续10小时 (36000秒)
print(f"Allocated {100}MB GPU memory")
print("Will hold for 10 hours")
time.sleep(36000)

# 释放内存
del tensor
torch.cuda.empty_cache()
EOF