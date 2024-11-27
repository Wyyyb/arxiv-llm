#!/bin/bash

source /gpfs/public/research/miniconda3/bin/activate
conda activate cite_rag

# temp debug, ocp 2 hours
python3 << EOF
import torch
import time

if not torch.cuda.is_available():
    print("CUDA is not available")
    exit(1)

size = int(100 * 1024 * 1024 / 4)
tensor = torch.zeros(size, device='cuda')

print(f"Allocated {100}MB GPU memory")
print("Will hold for 10 hours")
time.sleep(72000)

# 释放内存
del tensor
torch.cuda.empty_cache()
EOF

