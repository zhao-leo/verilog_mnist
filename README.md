# MNIST 手写数字识别 - FPGA硬件加速实现

基于Int8量化神经网络的MNIST手写数字识别系统，针对资源受限FPGA进行极致优化。

## 📋 项目概述

本项目实现了一个完整的MNIST手写数字识别系统，从PyTorch模型训练到Verilog硬件描述语言的自动生成，专门针对小型FPGA（6,272 LUT）进行了深度优化。

### 核心特性

- ✅ **极致资源优化**: 3个隐藏神经元，ROM仅2,395字节
- ✅ **Int8量化**: 8位整数运算，硬件友好
- ✅ **串行架构**: 单MAC单元时分复用，大幅降低逻辑资源
- ✅ **自动化流程**: 从训练到Verilog一键生成
- ✅ **完整验证**: 包含testbench和仿真测试

### 性能指标

| 指标 | 数值 |
|------|------|
| 网络结构 | 784 → 3 → 10 |
| 准确率 | ~77% (Int8量化后) |
| ROM大小 | 2,395 字节 |
| 推理延迟 | ~2,395 时钟周期 |
| 目标LUT | < 6,272 |
| 累加器位宽 | 24位 |

## 🏗️ 项目结构

```
verilog_minist/
├── mnist_model/           # PyTorch模型训练和Verilog生成
│   ├── main.py           # 主程序（训练+量化+生成）
│   ├── .venv/            # Python虚拟环境
│   └── data/             # MNIST数据集（自动下载）
├── verilog_src/          # 生成的Verilog代码
│   ├── mnist_model.v     # MNIST推理模块
│   ├── mnist_model_test.v # 测试bench
│   └── test.sh           # 仿真测试脚本
├── handwriting_module/   # ARM接口顶层模块
│   ├── handwriting.v     # 串行接收+推理
│   ├── handwriting_test.v # 顶层测试
│   ├── ram.v             # 图像缓存RAM
│   └── README.md         # 接口说明
└── README.md             # 本文件
```

## 🚀 快速开始

### 环境要求

- Python 3.8+ (已安装uv包管理器)
- Icarus Verilog (用于仿真)
- FPGA综合工具 (Quartus/Vivado/Diamond等)

### 1. 训练并生成Verilog

```bash
cd mnist_model

# 使用uv运行（会自动管理虚拟环境）
uv run python main.py

# 或者使用现有虚拟环境
.venv/bin/python main.py
```

这个过程会：
1. 自动下载MNIST数据集
2. 训练3-hidden神经网络（12 epochs）
3. 量化为Int8参数
4. 微调优化准确率
5. 生成`verilog_src/mnist_model.v`和测试文件

**预计用时**: 2-5分钟（取决于CPU）

### 2. 仿真测试

```bash
cd verilog_src
bash test.sh
```

输出示例：
```
Test 1 PASSED: Label=7, Expected=7, Got= 7
Test 2 PASSED: Label=2, Expected=2, Got= 2
Test 3 PASSED: Label=1, Expected=1, Got= 1
Test 4 PASSED: Label=0, Expected=0, Got= 0
Test 5 PASSED: Label=4, Expected=4, Got= 4
```

### 3. FPGA综合

将`verilog_src/mnist_model.v`导入您的FPGA项目。

**重要**: 确保ROM使用Block RAM而非LUT！

#### Quartus (Intel/Altera)
在`.qsf`文件中添加：
```tcl
set_instance_assignment -name RAMSTYLE "M9K" -to "mnist_model:*|weight_rom"
```

#### Vivado (Xilinx)
Verilog中已包含属性：
```verilog
(* ram_style = "block" *) reg signed [7:0] weight_rom [0:2394];
```

#### Lattice Diamond
Verilog中已包含属性：
```verilog
(* syn_ramstyle = "block_ram" *) reg signed [7:0] weight_rom [0:2394];
```

## 📡 接口说明

### mnist_model 模块

```verilog
module mnist_model(
    input wire clk,              // 时钟
    input wire rst,              // 复位（高电平有效）
    input wire [783:0] image_in, // 784位二值图像输入
    input wire start,            // 开始计算（脉冲）
    output reg [3:0] digit_out,  // 识别结果 (0-9)
    output reg valid             // 结果有效标志
);
```

**使用流程**：
1. 准备好784位图像数据在`image_in`
2. 拉高`start`一个时钟周期
3. 等待`valid`信号拉高
4. 读取`digit_out`结果

**时序**：约2,400个时钟周期完成一次推理

### handwriting 顶层模块

用于与ARM串行通信：

```verilog
module handwriting(
    input wire clk,              // 时钟（由ARM提供）
    input wire rst,              // 复位
    input wire data_in,          // 串行数据输入（每周期1位）
    output wire busy,            // 忙碌信号
    output wire [3:0] digit_out, // 识别结果
    output wire result_valid     // 结果有效
);
```

**协议**：
- ARM连续784个时钟周期发送图像数据（每周期1位）
- 自动触发推理
- `result_valid`拉高后读取结果
- 详见`handwriting_module/README.md`

## 🔧 架构优化

### 串行MAC架构

使用单个8位×24位MAC单元，时分复用计算所有神经元：

```
Layer1: 3个神经元 × 784个权重 = 2,355次MAC
Layer2: 10个神经元 × 3个权重 = 30次MAC
Argmax: 10次比较
总计: ~2,400个时钟周期
```

### 资源分配

| 组件 | 位宽/大小 | 数量 | 资源 |
|------|----------|------|------|
| ROM (weight_rom) | 8位 | 2,395 | **需BRAM** |
| 累加器 | 24位 | 1 | 24 FF |
| Layer1输出 | 24位 | 3 | 72 FF |
| Layer2输出 | 24位 | 10 | 240 FF |
| 状态机 | 2位 | 1 | 2 FF |
| 索引计数器 | 10+4位 | 2 | 14 FF |
| 控制逻辑 | - | - | ~500 LUT |

**预估LUT**: 2,500-3,500（ROM使用BRAM时）

### 优化历程

| 版本 | Hidden | ROM(字节) | 准确率 | 预估LUT |
|------|--------|-----------|--------|---------|
| v1.0 | 16 | 12,730 | 93% | ~30,000 |
| v2.0 | 8 | 6,370 | 90% | ~13,600 |
| v3.0 | 6 | 4,780 | 88% | ~8,000 |
| v4.0 | 5 | 3,985 | 85% | ~6,500 |
| **v5.0** | **3** | **2,395** | **77%** | **<6,272** ✓ |

## 📊 测试与验证

### 自动化测试

生成的`mnist_model_test.v`包含5个真实MNIST样本的仿真测试。

```bash
cd verilog_src
bash test.sh
```


## 🎯 准确率权衡

| 配置 | 准确率 | ROM | 适用场景 |
|------|--------|-----|----------|
| 3-hidden | 77% | 2.4KB | 极限资源约束 |
| 5-hidden | 85% | 4.0KB | 平衡性能和资源 |
| 8-hidden | 90% | 6.4KB | 优先准确率 |

**当前默认**: 3-hidden（满足6,272 LUT约束）

如需更高准确率，修改`mnist_model/main.py`:
```python
self.fc1 = nn.Linear(784, 5, bias=True)  # 改为5个隐藏神经元
self.fc2 = nn.Linear(5, 10, bias=True)
```
