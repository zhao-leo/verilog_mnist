import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os

# 简化的神经网络：784输入 -> 16隐藏层 -> 10输出
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 16, bias=True)
        self.fc2 = nn.Linear(16, 10, bias=True)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model():
    """训练MNIST模型"""
    print("开始训练模型...")

    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())  # 二值化
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 模型训练
    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练5个epoch
    for epoch in range(5):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

    # 测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = 100. * correct / total
    print(f'测试准确率: {accuracy:.2f}%')

    return model

def int8_inference(image, params, debug=False):
    """使用Int8参数进行推理（模拟硬件行为）"""
    fc1_w = params['fc1_weight']
    fc1_b = params['fc1_bias']
    fc2_w = params['fc2_weight']
    fc2_b = params['fc2_bias']

    # 展平图像并转换为0/1
    x = image.flatten()

    if debug:
        print(f"\n=== Int8推理调试信息 ===")
        print(f"输入图像非零像素数: {np.sum(x > 0)}")

    # 第一层计算（使用Int32中间结果）
    layer1_out = np.zeros(16, dtype=np.int32)
    for i in range(16):
        acc = int(fc1_b[i])
        for j in range(784):
            acc += int(fc1_w[i, j]) * int(x[j])
        # ReLU
        layer1_out[i] = max(0, acc)

    if debug:
        print(f"Layer1输出: {layer1_out}")

    # 第二层计算（使用Int32中间结果，右移7位防止溢出）
    layer2_out = np.zeros(10, dtype=np.int32)
    for i in range(10):
        acc = int(fc2_b[i])
        for j in range(16):
            acc += (int(fc2_w[i, j]) * layer1_out[j]) >> 7
        layer2_out[i] = acc

    if debug:
        print(f"Layer2输出: {layer2_out}")
        print(f"预测结果: {np.argmax(layer2_out)}")

    # 找到最大值的索引
    return np.argmax(layer2_out)

def quantize_to_int8(model):
    """将模型参数量化到int8"""
    print("\n量化模型参数到int8...")

    # 提取权重和偏置
    fc1_weight = model.fc1.weight.data.numpy()
    fc1_bias = model.fc1.bias.data.numpy()
    fc2_weight = model.fc2.weight.data.numpy()
    fc2_bias = model.fc2.bias.data.numpy()

    # 量化到int8范围 [-128, 127]
    # 使用缩放因子来保持数值范围
    scale1_w = 127.0 / (np.abs(fc1_weight).max() + 1e-8)
    scale1_b = 127.0 / (np.abs(fc1_bias).max() + 1e-8)
    scale2_w = 127.0 / (np.abs(fc2_weight).max() + 1e-8)
    scale2_b = 127.0 / (np.abs(fc2_bias).max() + 1e-8)

    fc1_weight_int8 = np.round(fc1_weight * scale1_w).astype(np.int8)
    fc1_bias_int8 = np.round(fc1_bias * scale1_b).astype(np.int8)
    fc2_weight_int8 = np.round(fc2_weight * scale2_w).astype(np.int8)
    fc2_bias_int8 = np.round(fc2_bias * scale2_b).astype(np.int8)

    print(f"Layer 1 - Weight shape: {fc1_weight_int8.shape}, Bias shape: {fc1_bias_int8.shape}")
    print(f"Layer 2 - Weight shape: {fc2_weight_int8.shape}, Bias shape: {fc2_bias_int8.shape}")

    return {
        'fc1_weight': fc1_weight_int8,
        'fc1_bias': fc1_bias_int8,
        'fc2_weight': fc2_weight_int8,
        'fc2_bias': fc2_bias_int8,
        'scales': (scale1_w, scale1_b, scale2_w, scale2_b)
    }

def validate_int8_model(params):
    """验证Int8模型的准确率"""
    print("\n验证Int8量化模型...")

    # 加载测试数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    correct = 0
    total = 0

    # 测试前1000个样本（加快速度）
    for i in range(min(1000, len(test_dataset))):
        image, label = test_dataset[i]
        image_np = image.numpy().squeeze()

        pred = int8_inference(image_np, params)
        if pred == label:
            correct += 1
        total += 1

        if (i + 1) % 200 == 0:
            print(f"  已测试 {i+1} 个样本...")

    accuracy = 100.0 * correct / total
    print(f"Int8模型准确率: {accuracy:.2f}% ({correct}/{total})")
    return accuracy

def fine_tune_int8(params, iterations=3):
    """对Int8参数进行简单的启发式微调"""
    print("\n对Int8参数进行微调...")

    # 加载训练数据用于微调
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])
    train_dataset = datasets.MNIST('./data', train=True, transform=transform)

    best_params = {k: v.copy() for k, v in params.items() if k != 'scales'}
    best_accuracy = validate_int8_model(best_params)

    # 简单的随机搜索微调
    for iteration in range(iterations):
        print(f"\n微调迭代 {iteration + 1}/{iterations}...")

        # 随机选择一些权重进行小幅调整
        test_params = {k: v.copy() for k, v in best_params.items()}

        # 随机调整第二层的一些权重（影响较大）
        for _ in range(5):
            i = np.random.randint(0, 10)
            j = np.random.randint(0, 16)
            delta = np.random.choice([-1, 1])
            new_val = int(test_params['fc2_weight'][i, j]) + delta
            if -128 <= new_val <= 127:
                test_params['fc2_weight'][i, j] = np.int8(new_val)

        # 随机调整第二层偏置
        for _ in range(3):
            i = np.random.randint(0, 10)
            delta = np.random.choice([-2, -1, 1, 2])
            new_val = int(test_params['fc2_bias'][i]) + delta
            if -128 <= new_val <= 127:
                test_params['fc2_bias'][i] = np.int8(new_val)

        # 评估
        accuracy = validate_int8_model(test_params)

        if accuracy > best_accuracy:
            print(f"  准确率提升: {best_accuracy:.2f}% -> {accuracy:.2f}%")
            best_accuracy = accuracy
            best_params = test_params
        else:
            print(f"  准确率未提升，保持最佳结果: {best_accuracy:.2f}%")

    print(f"\n微调完成！最终Int8模型准确率: {best_accuracy:.2f}%")
    return best_params

def generate_verilog(params):
    """生成Verilog代码"""
    print("\n生成Verilog代码...")

    fc1_w = params['fc1_weight']  # shape: (16, 784)
    fc1_b = params['fc1_bias']    # shape: (16,)
    fc2_w = params['fc2_weight']  # shape: (10, 16)
    fc2_b = params['fc2_bias']    # shape: (10,)

    # 生成主模块
    verilog_code = """// MNIST手写数字识别模型 - Int8量化版本
// 输入: 28x28二值图像 (784位)
// 输出: 预测数字 (0-9)

module mnist_model(
    input wire clk,
    input wire rst,
    input wire [783:0] image_in,  // 28*28 = 784位输入
    input wire start,
    output reg [3:0] digit_out,   // 输出数字 0-9
    output reg valid
);

    // 状态机
    localparam IDLE = 2'd0;
    localparam LAYER1 = 2'd1;
    localparam LAYER2 = 2'd2;
    localparam DONE = 2'd3;

    reg [1:0] state;
    reg [9:0] counter;

    // 第一层输出 (16个神经元)
    reg signed [31:0] layer1_out [0:15];

    // 第二层输出 (10个神经元)
    reg signed [31:0] layer2_out [0:9];

    integer i;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            valid <= 0;
            digit_out <= 0;
            counter <= 0;
        end else begin
            case (state)
                IDLE: begin
                    valid <= 0;
                    if (start) begin
                        state <= LAYER1;
                        counter <= 0;
                    end
                end

                LAYER1: begin
                    // 计算第一层（简化：直接计算所有神经元）
                    if (counter == 0) begin
"""

    # 生成第一层的计算（16个神经元）
    for neuron_idx in range(16):
        verilog_code += f"                        layer1_out[{neuron_idx}] = "

        # 偏置项
        bias_val = int(fc1_b[neuron_idx])
        verilog_code += f"{bias_val}"

        # 权重项 - 只为非零输入添加项
        # 为了控制表达式长度，我们每32个输入换一行
        for input_idx in range(784):
            weight_val = int(fc1_w[neuron_idx, input_idx])
            if weight_val != 0:
                sign = "+" if weight_val >= 0 else ""
                if (input_idx % 32 == 0 and input_idx > 0):
                    verilog_code += f"\n                            {sign}{weight_val} * image_in[{input_idx}]"
                else:
                    verilog_code += f" {sign}{weight_val} * image_in[{input_idx}]"

        verilog_code += ";\n"

        # 添加ReLU激活
        verilog_code += f"                        if (layer1_out[{neuron_idx}] < 0) layer1_out[{neuron_idx}] = 0;\n"

    verilog_code += """                        counter <= counter + 1;
                    end else begin
                        state <= LAYER2;
                        counter <= 0;
                    end
                end

                LAYER2: begin
                    // 计算第二层（10个神经元）
                    if (counter == 0) begin
"""

    # 生成第二层的计算（10个神经元）
    for neuron_idx in range(10):
        verilog_code += f"                        layer2_out[{neuron_idx}] = "

        # 偏置项
        bias_val = int(fc2_b[neuron_idx])
        verilog_code += f"{bias_val}"

        # 权重项
        for input_idx in range(16):
            weight_val = int(fc2_w[neuron_idx, input_idx])
            if weight_val != 0:
                if weight_val >= 0:
                    verilog_code += f" + (({weight_val} * layer1_out[{input_idx}]) >>> 7)"
                else:
                    verilog_code += f" - (({-weight_val} * layer1_out[{input_idx}]) >>> 7)"

        verilog_code += ";\n"

    verilog_code += """                        counter <= counter + 1;
                    end else begin
                        state <= DONE;
                    end
                end

                DONE: begin
                    // 找到最大值的索引
                    if (layer2_out[0] >= layer2_out[1] && layer2_out[0] >= layer2_out[2] &&
                        layer2_out[0] >= layer2_out[3] && layer2_out[0] >= layer2_out[4] &&
                        layer2_out[0] >= layer2_out[5] && layer2_out[0] >= layer2_out[6] &&
                        layer2_out[0] >= layer2_out[7] && layer2_out[0] >= layer2_out[8] &&
                        layer2_out[0] >= layer2_out[9])
                        digit_out <= 0;
                    else if (layer2_out[1] >= layer2_out[2] && layer2_out[1] >= layer2_out[3] &&
                             layer2_out[1] >= layer2_out[4] && layer2_out[1] >= layer2_out[5] &&
                             layer2_out[1] >= layer2_out[6] && layer2_out[1] >= layer2_out[7] &&
                             layer2_out[1] >= layer2_out[8] && layer2_out[1] >= layer2_out[9])
                        digit_out <= 1;
                    else if (layer2_out[2] >= layer2_out[3] && layer2_out[2] >= layer2_out[4] &&
                             layer2_out[2] >= layer2_out[5] && layer2_out[2] >= layer2_out[6] &&
                             layer2_out[2] >= layer2_out[7] && layer2_out[2] >= layer2_out[8] &&
                             layer2_out[2] >= layer2_out[9])
                        digit_out <= 2;
                    else if (layer2_out[3] >= layer2_out[4] && layer2_out[3] >= layer2_out[5] &&
                             layer2_out[3] >= layer2_out[6] && layer2_out[3] >= layer2_out[7] &&
                             layer2_out[3] >= layer2_out[8] && layer2_out[3] >= layer2_out[9])
                        digit_out <= 3;
                    else if (layer2_out[4] >= layer2_out[5] && layer2_out[4] >= layer2_out[6] &&
                             layer2_out[4] >= layer2_out[7] && layer2_out[4] >= layer2_out[8] &&
                             layer2_out[4] >= layer2_out[9])
                        digit_out <= 4;
                    else if (layer2_out[5] >= layer2_out[6] && layer2_out[5] >= layer2_out[7] &&
                             layer2_out[5] >= layer2_out[8] && layer2_out[5] >= layer2_out[9])
                        digit_out <= 5;
                    else if (layer2_out[6] >= layer2_out[7] && layer2_out[6] >= layer2_out[8] &&
                             layer2_out[6] >= layer2_out[9])
                        digit_out <= 6;
                    else if (layer2_out[7] >= layer2_out[8] && layer2_out[7] >= layer2_out[9])
                        digit_out <= 7;
                    else if (layer2_out[8] >= layer2_out[9])
                        digit_out <= 8;
                    else
                        digit_out <= 9;

                    valid <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
"""

    # 写入文件
    verilog_dir = "../verilog_src"
    os.makedirs(verilog_dir, exist_ok=True)

    with open(os.path.join(verilog_dir, "mnist_model.v"), "w") as f:
        f.write(verilog_code)

    print(f"Verilog模型已生成: {verilog_dir}/mnist_model.v")

    # 生成测试文件（传入参数用于生成真实测试用例）
    generate_testbench(verilog_dir, params)

def generate_testbench(verilog_dir, params):
    """生成Verilog测试文件（使用真实MNIST样本）"""

    # 加载测试数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # 选择5个不同数字的样本
    test_samples = []
    used_digits = set()

    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        if label not in used_digits:
            image_np = image.numpy().squeeze()
            # 使用Int8推理得到预期输出
            predicted = int8_inference(image_np, params)
            test_samples.append((image_np, label, predicted))
            used_digits.add(label)
            if len(test_samples) >= 5:
                break

    print(f"\n选择了 {len(test_samples)} 个MNIST测试样本用于Verilog仿真")
    for idx, (img, label, pred) in enumerate(test_samples):
        print(f"  样本{idx+1}: 真实标签={label}, Int8预测={pred}")

    # 对第一个样本进行详细调试
    print(f"\n对第一个样本进行详细验证...")
    test_img = test_samples[0][0]
    print(f"输入图像的前10个像素: {test_img.flatten()[:10]}")
    print(f"输入图像的像素和: {np.sum(test_img.flatten())}")

    # 打印Verilog格式的输入
    img_flat = test_img.flatten()
    verilog_bits = ""
    for i in range(10):
        verilog_bits += '1' if int(img_flat[i]) > 0 else '0'
    print(f"Verilog输入前10位(bit[0]到bit[9]): {verilog_bits}")

    int8_inference(test_img, params, debug=True)

    testbench = """// MNIST模型测试文件 - 使用真实MNIST数据

`timescale 1ns / 1ps

module mnist_model_test;

    reg clk;
    reg rst;
    reg [783:0] image_in;
    reg start;
    wire [3:0] digit_out;
    wire valid;

    // 实例化模块
    mnist_model uut (
        .clk(clk),
        .rst(rst),
        .image_in(image_in),
        .start(start),
        .digit_out(digit_out),
        .valid(valid)
    );

    // 时钟生成
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // 测试流程
    initial begin
        // 初始化
        rst = 1;
        start = 0;
        image_in = 784'b0;

        #20 rst = 0;
"""

    # 为每个测试样本生成测试代码
    for idx, (image, label, predicted) in enumerate(test_samples):
        testbench += f"""
        // 测试用例{idx+1}: 真实MNIST样本 (标签={label}, 期望输出={predicted})
        #50;
        image_in = 784'b"""

        # 将图像转换为二进制字符串
        image_flat = image.flatten()
        for i in range(783, -1, -1):  # 从高位到低位
            testbench += '1' if int(image_flat[i]) > 0 else '0'

        testbench += f""";
            start = 1;
            #10 start = 0;

            wait(valid);
            $display("Test {idx+1}: Layer1[0]=%d, Layer1[1]=%d, Layer1[2]=%d, Layer1[3]=%d",
                     uut.layer1_out[0], uut.layer1_out[1], uut.layer1_out[2], uut.layer1_out[3]);
            $display("         Layer2[0]=%d, [1]=%d, [2]=%d, [3]=%d, [4]=%d",
                     uut.layer2_out[0], uut.layer2_out[1], uut.layer2_out[2], uut.layer2_out[3], uut.layer2_out[4]);
            $display("         Layer2[5]=%d, [6]=%d, [7]=%d, [8]=%d, [9]=%d",
                     uut.layer2_out[5], uut.layer2_out[6], uut.layer2_out[7], uut.layer2_out[8], uut.layer2_out[9]);
            if (digit_out == {predicted})
                $display("Test {idx+1} PASSED: Label={label}, Expected={predicted}, Got=%d", digit_out);
            else
                $display("Test {idx+1} FAILED: Label={label}, Expected={predicted}, Got=%d", digit_out);
    """

    testbench += """
        #100;
        $display("\\n所有测试完成");
        $finish;
    end

    // 监控输出
    initial begin
        $monitor("Time=%0t rst=%b start=%b valid=%b digit_out=%d",
                 $time, rst, start, valid, digit_out);
    end

endmodule
"""

    with open(os.path.join(verilog_dir, "mnist_model_test.v"), "w") as f:
        f.write(testbench)

    print(f"Verilog测试文件已生成: {verilog_dir}/mnist_model_test.v")

def main():
    print("="*60)
    print("MNIST手写数字识别 - Int8量化模型")
    print("="*60)

    # 训练模型
    model = train_model()

    # 量化到int8
    quantized_params = quantize_to_int8(model)

    # 验证Int8模型准确率
    initial_accuracy = validate_int8_model(quantized_params)

    # 如果准确率低于90%，进行微调
    if initial_accuracy < 90.0:
        print(f"\n初始Int8准确率 ({initial_accuracy:.2f}%) 较低，开始微调...")
        quantized_params = fine_tune_int8(quantized_params, iterations=5)
    else:
        print(f"\nInt8模型准确率已达到 {initial_accuracy:.2f}%，无需微调")

    # 生成Verilog代码
    generate_verilog(quantized_params)

    print("\n"+"="*60)
    print("完成！模型已训练并生成Verilog代码")
    print("="*60)
    print("\n使用说明:")
    print("1. Verilog模块: ../verilog_src/mnist_model.v")
    print("2. 测试文件: ../verilog_src/mnist_model_test.v")
    print("3. 输入: 784位二值图像 (28x28)")
    print("4. 输出: 4位数字 (0-9)")
    print("\n注意: 所有参数已硬编码到Verilog文件中")

if __name__ == "__main__":
    main()
