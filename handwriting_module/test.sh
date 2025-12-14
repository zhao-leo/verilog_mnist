#!/bin/bash

# 编译所有Verilog文件
echo "编译Verilog文件..."
iverilog -o handwriting.vvp \
    handwriting_test.v \
    handwriting.v \
    ram.v \
    mnist_model.v

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo "编译成功！"
    echo "运行仿真..."
    vvp handwriting.vvp
else
    echo "编译失败！"
    exit 1
fi
