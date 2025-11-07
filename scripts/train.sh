#!/bin/bash
# 测试脚本

cd "../src"
nohup python train.py > ../logs/train.log 2>&1 &

