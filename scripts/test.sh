#!/bin/bash
# 测试脚本

cd ../src"
nohup python test.py > ../logs/test.log 2>&1 &

