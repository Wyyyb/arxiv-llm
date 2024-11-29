#!/bin/bash

# 创建本地目录（如果不存在）
mkdir -p ../local_1129/

# 使用scp下载文件
scp -P 2222 "xcs-research-share-bqjxr-master-0.ou-600a79a43b2e47a07dfcc2c984743ee8.pytorch.bash.dev.pod@sshproxy.dh3.ai:/gpfs/public/research/xy/yubowang/arxiv-llm/local_1129/test_results_1129/*.txt" ../local_1129/
