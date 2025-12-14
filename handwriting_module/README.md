# 手写数字识别模块使用说明

## ⚡ 性能与资源特性

**架构**: 串行计算（优化逻辑门使用）  
**逻辑资源**: ~5K LUT（相比全并行架构节省95%）  
**ROM大小**: 12,730 字节（存储Int8权重和偏置）  
**计算延迟**: ~12,720 时钟周期  
**响应时间**: 1.3ms @ 10MHz 时钟  
**准确率**: ~93% (MNIST测试集)  

> 💡 **设计理念**: 使用单个MAC单元串行计算，用时间换空间，大幅减少逻辑门数量，适合资源受限的FPGA平台。详见 [../ARCHITECTURE.md](../ARCHITECTURE.md)

## 模块概述

本模块为手写数字识别的顶层接口，用于ARM与FPGA之间的通信。模块接收串行图像数据（784位），通过内置的MNIST神经网络进行识别，并返回识别结果（0-9）。

## 文件说明

- `handwriting.v` - 顶层模块，负责状态机控制和模块集成
- `ram.v` - 数据缓存模块，将串行输入转换为并行数据
- `mnist_model.v` - 神经网络推理模块（需要从../verilog_src/复制）

## 模块接口

```verilog
module handwriting(
    input wire clk,              // 时钟输入（由ARM提供）
    input wire rst,              // 复位信号（高电平有效）
    input wire data_in,          // 串行数据输入（1位）
    output wire busy,            // 忙碌信号
    output wire [3:0] digit_out, // 识别结果（0-9）
    output wire result_valid     // 结果有效信号
);
```

### 接口说明

| 信号名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| clk | 输入 | 1 | 时钟信号，由ARM提供 |
| rst | 输入 | 1 | 复位信号，高电平有效 |
| data_in | 输入 | 1 | 串行数据输入，每个时钟周期传输1位 |
| busy | 输出 | 1 | 忙碌信号，高电平表示正在处理 |
| digit_out | 输出 | 4 | 识别结果，范围0-9 |
| result_valid | 输出 | 1 | 结果有效信号，高电平表示digit_out有效 |

## ARM调用流程

### 1. 初始化阶段

```c
// 1. 拉高复位信号
set_rst_pin(HIGH);
delay_us(10);
set_rst_pin(LOW);
delay_us(10);

// 2. 检查busy信号，确保模块处于空闲状态
while(read_busy_pin() == HIGH);
```

### 2. 数据传输阶段

**重要：必须严格遵守以下时序规范，确保数据在时钟上升沿之前稳定！**

```c
// 准备28x28的二值图像数据（0或1）
uint8_t image[784];  // 0表示背景，1表示笔迹

// 开始发送数据
for(int i = 0; i < 784; i++) {
    // 步骤1: 确保时钟为低电平
    set_clk_pin(LOW);
    delay_ns(50);  // 等待时钟稳定
    
    // 步骤2: 在时钟低电平期间设置数据
    set_data_pin(image[i]);
    delay_ns(50);  // 数据建立时间（setup time），确保数据在上升沿前稳定
    
    // 步骤3: 产生时钟上升沿，FPGA在此采样稳定的数据
    set_clk_pin(HIGH);
    delay_ns(100); // 时钟高电平保持时间
}

// 此时busy应该已经拉高
```

### 3. 等待计算完成

```c
// 继续提供时钟信号，等待计算完成
int timeout = 0;
while(read_busy_pin() == HIGH && timeout < 10000) {
    // 继续提供时钟（数据已发送完，data_in可保持不变）
    set_clk_pin(LOW);
    delay_ns(100);
    set_clk_pin(HIGH);
    delay_ns(100);
    timeout++;
}

if(timeout >= 10000) {
    // 超时处理
    printf("Error: Timeout\n");
    return -1;
}
```

### 4. 读取结果

```c
// busy变低后，检查result_valid信号
if(read_result_valid_pin() == HIGH) {
    // 读取4位识别结果
    uint8_t result = read_digit_out_pins();  // 0-9
    printf("识别结果: %d\n", result);
} else {
    printf("Error: Result invalid\n");
}
```

### 5. 停止时钟

```c
// 读取完成后停止时钟供应
set_clk_pin(LOW);
```

## 完整C代码示例

```c
#include <stdint.h>
#include <stdio.h>

// GPIO操作函数（需要根据具体硬件平台实现）
void set_clk_pin(uint8_t level);
void set_rst_pin(uint8_t level);
void set_data_pin(uint8_t level);
uint8_t read_busy_pin(void);
uint8_t read_result_valid_pin(void);
uint8_t read_digit_out_pins(void);
void delay_ns(uint32_t ns);
void delay_us(uint32_t us);

#define HIGH 1
#define LOW 0

int recognize_digit(uint8_t *image) {
    int i, timeout;
    uint8_t result;
    
    // 1. 复位模块
    set_rst_pin(HIGH);
    delay_us(10);
    set_rst_pin(LOW);
    delay_us(10);
    
    // 2. 检查初始状态
    if(read_busy_pin() == HIGH) {
        printf("Error: Module busy at start\n");
        return -1;
    }
    
    // 3. 发送784位图像数据（严格遵守时序）
    for(i = 0; i < 784; i++) {
        // 先拉低时钟
        set_clk_pin(LOW);
        delay_ns(50);
        
        // 在时钟低电平期间设置数据
        set_data_pin(image[i]);
        delay_ns(50);  // 数据建立时间
        
        // 产生上升沿采样
        set_clk_pin(HIGH);
        delay_ns(100);
    }
    
    // 4. 继续提供时钟，等待计算完成
    timeout = 0;
    while(read_busy_pin() == HIGH && timeout < 10000) {
        set_clk_pin(LOW);
        delay_ns(100);
        set_clk_pin(HIGH);
        delay_ns(100);
        timeout++;
    }
    
    if(timeout >= 10000) {
        printf("Error: Timeout waiting for computation\n");
        return -1;
    }
    
    // 5. 读取结果
    if(read_result_valid_pin() == HIGH) {
        result = read_digit_out_pins();
        
        // 6. 停止时钟
        set_clk_pin(LOW);
        
        return result;
    } else {
        printf("Error: Result invalid\n");
        set_clk_pin(LOW);
        return -1;
    }
}

// 使用示例
int main() {
    uint8_t image[784];
    int result;
    
    // 准备图像数据（示例：全0）
    for(int i = 0; i < 784; i++) {
        image[i] = 0;
    }
    
    // 在中间绘制一个简单的数字1
    for(int row = 5; row < 23; row++) {
        image[row * 28 + 14] = 1;
    }
    
    // 进行识别
    result = recognize_digit(image);
    
    if(result >= 0) {
        printf("识别结果: %d\n", result);
    }
    
    return 0;
}
```

## 时序说明

### 完整工作流程时序

```
ARM端:    _____|‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|_____
(时钟)         发送784位数据 + 等待计算        停止

rst:      ‾‾|___|‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

busy:     _____|‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|________

data_in:  -----<D0><D1><D2>...<D783>----------

valid:    __________________________________|‾‾‾‾|__

digit_out: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX<RESULT>
```

### 单个数据位的详细时序（关键！）

**每个数据位必须遵守以下时序要求：**

```
时钟:     ______|‾‾‾‾‾‾‾‾‾‾|______|‾‾‾‾‾‾‾‾‾‾|______
              ↑                  ↑
              采样点1              采样点2

data_in:  ----<════D0════>------<════D1════>------
              ↑         ↑
              |         保持时间(hold time)
              建立时间(setup time)

说明:
1. 时钟为低电平时，设置data_in为目标值
2. 给予至少50ns的建立时间，让数据稳定
3. 时钟上升沿到来时，FPGA采样稳定的数据
4. 数据在上升沿后还需保持一段时间（保持时间）
```

**错误示例（数据与时钟边沿同时变化）：**
```
时钟:     ______|‾‾‾‾‾‾‾‾‾‾|______|‾‾‾‾‾‾‾‾‾‾|______
              ↑                  ↑
data_in:  ----X<══D0══>--------X<══D1══>----------
              ↑ 危险！           ↑ 危险！
              数据和时钟同时变化会导致采样错误
```

### 关键时序参数

- **数据传输时间**: 784个时钟周期（每个数据位一个时钟）
- **计算时间**: 约12,720个时钟周期（串行MAC架构）
  - Layer1: 16 × 785 = 12,560 cycles
  - Layer2: 10 × 16 = 160 cycles
- **总时间**: 约13,500个时钟周期（传输+计算）
- **建议时钟频率**: 1-50 MHz
- **时钟周期**: 20ns - 1μs（取决于时钟频率）
- **数据建立时间**: ≥50ns（推荐）
- **数据保持时间**: ≥10ns（推荐）
- **每次识别总耗时**: 
  - @ 10 MHz: ~1.35 ms
  - @ 50 MHz: ~270 μs

## 注意事项

1. **数据格式**: 
   - 图像数据必须是28×28的二值图像
   - 按照从左到右、从上到下的顺序传输
   - bit[0]对应第一行第一列，bit[783]对应最后一行最后一列
   - 1表示笔迹，0表示背景

2. **时钟和数据时序要求（非常重要！）**:
   - 时钟由ARM提供
   - 建议频率1-10 MHz
   - **数据必须在时钟低电平期间设置，并在上升沿之前稳定**
   - **数据建立时间：至少50ns**（从设置数据到时钟上升沿）
   - **数据保持时间：至少10ns**（时钟上升沿后数据保持稳定）
   - 禁止在时钟边沿同时改变数据，否则会导致采样错误
   - 必须在整个数据传输和计算过程中保持时钟稳定
   - 计算完成后才能停止时钟

3. **复位时序**:
   - 复位信号至少保持10个时钟周期
   - 复位后需要等待busy信号变低

4. **时序调试**:
   - 如果识别结果始终错误，首先检查时钟和数据的时序关系
   - 使用示波器或逻辑分析仪确认：
     - 数据在时钟低电平期间稳定
     - 数据在上升沿之前至少50ns就已设置好
     - 数据和时钟边沿不会同时变化
   - 测试建议：先用低频时钟（如1MHz）验证功能，再提高频率

5. **错误处理**:
   - 添加超时检测机制（建议10000个时钟周期）
   - 检查result_valid信号确保结果有效
   - 传输错误时重新复位并重试

6. **功耗优化**:
   - 在IDLE状态时停止时钟供应可降低功耗
   - 每次识别完成后及时停止时钟

## GPIO引脚分配建议

| 信号 | ARM GPIO | FPGA引脚 | 方向 |
|------|----------|----------|------|
| clk | GPIO0 | PIN_A1 | ARM→FPGA |
| rst | GPIO1 | PIN_A2 | ARM→FPGA |
| data_in | GPIO2 | PIN_A3 | ARM→FPGA |
| busy | GPIO3 | PIN_B1 | FPGA→ARM |
| result_valid | GPIO4 | PIN_B2 | FPGA→ARM |
| digit_out[0] | GPIO5 | PIN_B3 | FPGA→ARM |
| digit_out[1] | GPIO6 | PIN_B4 | FPGA→ARM |
| digit_out[2] | GPIO7 | PIN_B5 | FPGA→ARM |
| digit_out[3] | GPIO8 | PIN_B6 | FPGA→ARM |

## 集成说明

将本模块集成到FPGA项目中需要：

1. 复制 `mnist_model.v` 从 `../verilog_src/` 到当前目录
2. 在顶层项目中实例化 `handwriting` 模块
3. 配置GPIO引脚约束
4. 编译并下载到FPGA

## 性能指标

- **准确率**: ~93% (基于MNIST测试集)
- **延迟**: 1.35ms @ 10MHz, 270μs @ 50MHz
- **资源占用**: 
  - 逻辑资源: ~5K LUT（串行架构，大幅优化）
  - ROM: 12,730字节（存储Int8权重和偏置）
  - RAM: ~1.6KB（中间结果缓存）
  - 16个隐藏层神经元
  - 适合中小型FPGA（包括Cyclone II、Spartan-3等）
- **功耗**: 低（单MAC单元，翻转率低）

## 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| busy一直为高 | 数据未传输完整 | 确保发送了784位数据 |
| result_valid为低 | 计算未完成 | 继续提供时钟直到busy变低 |
| 识别结果总是错误 | **时序问题：数据与时钟边沿对齐** | **在时钟低电平期间设置数据，给予足够建立时间** |
| 识别结果偶尔错误 | 数据建立/保持时间不足 | 增加delay_ns的延迟时间，降低时钟频率 |
| 图像数据错位 | 时钟和数据同时变化 | 严格按照时序要求：先设置数据，再产生时钟上升沿 |
| 超时 | 时钟频率过低 | 提高时钟频率或增加超时阈值 |
| 模块无响应 | 未正确复位 | 重新执行复位流程 |

## 版本信息

- **版本**: 1.0
- **更新日期**: 2024
- **兼容性**: 需要与mnist_model.v配合使用
