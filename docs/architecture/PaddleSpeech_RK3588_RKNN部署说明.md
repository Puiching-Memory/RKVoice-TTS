# PaddleSpeech 在 A588 RK3588 开发板上的 TTS 部署说明

## 1. 已确认的板端环境

通过实际连板读取到的信息如下：

- 板型：A588
- SoC：RK3588
- 系统：Ubuntu 22.04.5 LTS
- 架构：aarch64
- 内核：Linux 6.1.118
- CPU：8 核
- 内存：7.7 GiB
- Python：3.10.12
- 根分区可用空间：约 48 GiB

板端当前与 RKNN 相关的运行条件：

- 已存在 `/usr/lib/librknnrt.so`
- 已存在 `/usr/bin/rknn_server`
- NPU 驱动版本：`RKNPU driver: v0.9.8`

板端当前缺少或受限项：

- 没有 `pip3`
- 没有 `cmake`
- 不能解析外网 DNS，无法直接从板端下载 GitHub 或 PaddleSpeech 模型文件

结论：

1. 板端运行 RKNN Runtime 的基础条件已经具备。
2. 模型转换必须在 PC 端完成，再把产物拷到板端。
3. 板端更适合跑最终推理程序，不适合直接在板端搭完整 Python 训练/转换环境。

## 2. 关键结论

### 2.1 PaddleSpeech 官方 ARM Linux Demo 不是 RKNN 路线

PaddleSpeech 官方的 ARM Linux TTS Demo 是：

- `demos/TTSArmLinux`

这个 Demo 的特征是：

- 使用 Paddle Lite
- 官方下载的是 `fs2cnn_mbmelgan_cpu_v1.3.0.tar.gz`
- 官方 README 明确是 ARM Linux C++ Demo
- 默认是中文合成

这条路线可以在板端跑通 PaddleSpeech TTS，但默认不是 RKNN NPU 推理。

### 2.2 PaddleSpeech 直接转 RKNN 目前没有现成官方可落地方案

已核实到的公开信息：

- PaddleSpeech issue #4054 中，维护者回复过“npu 是支持的，但没有针对 RKNN 模型测试过”，并建议尝试 `examples/csmsc/tts2` 里的 `run_npu.sh`。
- 但 `run_npu.sh` 本身只是走 Paddle 的 NPU 脚本入口，使用的是 `FLAGS_selected_npus` 和 Python 推理脚本，不是 RKNN Toolkit2 的转换加部署链路。
- PaddleSpeech issue #3367 中，有用户表示“所有声学模型都没法成功转成 RKNN”。
- PaddleSpeech issue #3368 中，社区讨论明确提到：TTS 转 RKNN 的核心问题是动态输入/动态 shape，RKNN 更适合固定输入长度。

结论：

1. “PaddleSpeech 官方现成 Demo 直接上 RK3588 NPU”这件事，目前没有现成、已验证的官方路径。
2. 如果必须利用 RK3588 NPU，需要做二次工程化改造，而不是直接执行 PaddleSpeech 的现成 Demo。

### 2.3 RKNN 官方确实有 TTS 示例，但不是 PaddleSpeech

Rockchip 官方 `rknn_model_zoo` 已提供：

- `examples/mms_tts`

这个示例说明了两件事：

1. RK3588 上做 TTS 的 NPU 推理是可行的。
2. 可行前提通常是固定输入长度，并且把动态部分拆开。

`mms_tts` 官方方案的特点：

- 文本输入长度固定为 `MAX_LENGTH`，默认示例是 200
- 模型拆成 `encoder` 和 `decoder`
- 中间的时长展开、attention 构造等步骤放在 CPU 上做
- 最终把固定 shape 的子图送给 RKNN

这正是你在 RK3588 上想让 PaddleSpeech 用上 NPU 时，最应该参考的结构。

## 3. 推荐部署路线

如果目标是“既要 PaddleSpeech，又要尽可能利用 RK3588 NPU”，建议分两步走。

### 路线 A：先跑通 PaddleSpeech 官方 ARM Linux 版本

用途：

- 先验证中文前端、音频输出、整机链路
- 先确认业务文本在板端能稳定合成
- 这一步主要解决“能不能用”，不是“极致性能”

优点：

- 最接近 PaddleSpeech 官方支持路径
- 风险最低
- 便于先打通业务流程

缺点：

- 默认是 Paddle Lite CPU 路线，不是 RKNN NPU

### 路线 B：再把 TTS 推理子图改造成 RKNN 路线

用途：

- 真正使用 RK3588 NPU
- 把耗时最大的推理子图从 CPU 挪到 NPU

建议结构：

- CPU：分句、文本归一化、G2P、词典、多音字处理
- NPU：声学模型主干
- CPU：时长展开、padding、裁剪、拼接
- NPU：vocoder 主干
- CPU：结果裁剪、WAV 落盘、播放

为什么要这样拆：

- RKNN 更适合固定 shape 子图
- TTS 里最麻烦的是可变长度文本和可变长度 mel/wav
- 把动态逻辑留在 CPU，固定 shape 的矩阵计算留给 NPU，成功率最高

## 4. 模型选择建议

### 4.1 不建议第一版就上 VITS

原因：

- 图更复杂
- 动态 shape 更多
- ONNX 导出和 RKNN 转换风险更高

### 4.2 第一版优先选 SpeedySpeech 或轻量 FastSpeech2

理由：

- PaddleSpeech 自身对 `examples/csmsc/tts2` 提供了较完整的导出、静态图和 ONNX 相关资源
- SpeedySpeech 的结构比 VITS 更容易切分
- 先把声学模型跑上 NPU，再处理 vocoder，更符合工程推进顺序

### 4.3 Vocoder 先从 MB-MelGAN 入手

原因：

- 相比 HiFiGAN，通常更轻量
- 官方 ARM Demo 也是以 `fs2cnn_mbmelgan_cpu_v1.3.0` 为现成模型包
- 更适合作为 NPU 改造的第一目标

## 5. 具体实施方案

## 5.1 第一阶段：PaddleSpeech 官方 ARM Linux 版本落地

### PC 端准备

由于板端没有 DNS，以下文件建议都在 PC 端下载后再复制到板端：

1. PaddleSpeech 源码
2. `demos/TTSArmLinux` 所需文件
3. Paddle Lite 预编译库
4. 现成模型包
5. 中文前端字典

参考资源：

- PaddleSpeech 仓库：`https://github.com/PaddlePaddle/PaddleSpeech`
- ARM Linux Demo：`demos/TTSArmLinux`

官方 Demo 用到的关键文件：

- `inference_lite_lib.armlinux.armv8.gcc.with_extra.with_cv.tar.gz`
- `fs2cnn_mbmelgan_cpu_v1.3.0.tar.gz`

### 板端建议目录

建议在板端统一放到：

```bash
/root/tts/
```

目录建议结构：

```text
/root/tts/
  PaddleSpeech/
  assets/
```

### 板端运行思路

如果你能补齐 `cmake`，可以直接在板端编译：

```bash
cd /root/tts/PaddleSpeech/demos/TTSArmLinux
./build.sh
./run.sh --sentence "短波电台指令测试"
```

如果暂时补不齐 `cmake`：

- 在 PC 端交叉编译或直接准备好 Demo 二进制
- 把 `build/`、`libs/`、`models/`、`dict/` 一并拷到板端
- 直接执行 `./run.sh`

### 这一阶段的目标

- 先把 PaddleSpeech 中文 TTS 在 A588 上跑通
- 验证音质、词典、业务文本和输出文件链路
- 不在这一阶段强行上 RKNN

## 5.2 第二阶段：PaddleSpeech 改造成 RKNN NPU 推理

这是重点。

### 原则一：转换必须在 PC 端完成

RKNN-Toolkit2 的官方定位就是：

- 在 PC 端把模型转成 `.rknn`
- 在板端用 RKNN Runtime 或 RKNN Toolkit Lite2 执行

不要在板端尝试做模型转换。

### 原则二：PC 端不要用当前工作区这个 Python 3.14 环境

RKNN-Toolkit2 官方支持的 Python 版本是：

- 3.6 到 3.12

而当前工作区虚拟环境是：

- Python 3.14.3

所以转换环境要单独建，比如：

- Ubuntu 22.04 + Python 3.10
- 或 Conda Python 3.10/3.11

### 原则三：先固定长度，再谈 RKNN

参考 RKNN 官方 `mms_tts` 的做法，PaddleSpeech 若要上 RKNN，应采用固定长度策略。

建议你在第一版中直接定义：

- `MAX_PHONE_LEN=100` 或 `200`
- `MAX_MEL_LEN` 按声学模型的最大展开长度预估
- 超长句子先分句
- 超长 token 直接截断或拆分

### 推荐改造方式

建议仿照 RKNN `mms_tts`，把 PaddleSpeech TTS 拆成下面三段：

1. CPU 前端

- 文本清洗
- 中文分句
- 数字、符号归一化
- phone / tone id 生成
- padding 到固定长度

2. NPU 声学模型

- 优先尝试 `SpeedySpeech` 或轻量 `FastSpeech2`
- 目标输出固定 shape 的 mel 特征

3. NPU 或 CPU vocoder

- 第一版先尝试 `MB-MelGAN`
- 如果 vocoder 转 RKNN 失败，先保留 CPU vocoder 也可以

这样你至少可以先做到：

- 文本前端是 PaddleSpeech
- 声学模型吃 NPU
- vocoder 后续再继续优化

这是最稳的推进方式。

## 5.3 RKNN 转换参考流程

### PC 端准备转换环境

建议环境：

```bash
python3.10 -m venv .venv-rknn
source .venv-rknn/bin/activate
pip install --upgrade pip
```

然后按 RKNN-Toolkit2 官方安装包安装对应版本。

### 准备 PaddleSpeech 导出模型

优先从 PaddleSpeech 已有静态图或 ONNX 资源下手，尤其是：

- `examples/csmsc/tts2`
- 其中可下载到 `speedyspeech_csmsc_onnx_0.2.0.zip`

注意：

- 原始 ONNX 不代表一定能直接转 RKNN
- 很可能还要手改导出逻辑，去掉动态 shape 和不支持的算子

### RKNN 转换策略

转换脚本的核心思路是：

1. 先把模型切成固定 shape 子图
2. 用 `rknn.load_onnx()` 载入 ONNX
3. `rknn.build()` 生成 RKNN
4. `rknn.export_rknn()` 导出

如果直接整图失败，不要继续死磕整图，优先：

- 拆 encoder / decoder
- 把动态部分搬回 CPU
- 把随机、cumsum、range、按索引取值这类动态逻辑显式改写

这也是 RKNN 官方 `mms_tts` 示例的实际做法。

## 6. 为什么不建议直接依赖 PaddleSpeech 的 run_npu.sh

虽然 PaddleSpeech `examples/csmsc/tts2` 里确实有：

- `run_npu.sh`

但从脚本内容看，它是：

- `FLAGS_selected_npus=...`
- 调用 `train_npu.sh`、`synthesize_npu.sh`、`inference_npu.sh`
- 本质仍是 Paddle 自己的 Python 推理训练脚本

它没有体现以下 RKNN 路线的关键要素：

- `.rknn` 模型
- `rknn.load_rknn`
- `librknnrt.so`
- RKNN Runtime / Lite2

所以这不能等价理解为“官方已经给了 RK3588 RKNN 版 TTS Demo”。

## 7. 针对当前这块板子的最优建议

基于当前实测环境，建议优先级如下。

### 建议 1：先把 PaddleSpeech ARM Linux CPU 版跑通

原因：

- 风险最低
- 先确认文本前端、词典和音频输出链路
- 先解决业务可用性

### 建议 2：NPU 路线不要从“整套 PaddleSpeech 直接转 RKNN”开始

更合理的做法是：

- 先参考 RKNN 官方 `mms_tts` 示例，跑通 “TTS on NPU” 的完整样板
- 再按同样结构替换成 PaddleSpeech 的固定长度子图

这样做的好处是：

- 先验证板端 NPU TTS 通路
- 再替换模型
- 每一步都可测、可回退

### 建议 3：板端最终部署尽量用 C++

原因：

- 板端已经有 `librknnrt.so`
- 板端缺少 `pip3`
- 厂商文档本身也更偏向板端 C++ 推理

推荐最终形态：

- PC 端：PaddleSpeech 导出 + RKNN 转换
- 板端：C++ 调 RKNN Runtime
- CPU 前端和控制逻辑可用 C++ 或极简 Python 包装

## 8. 最小可落地实施顺序

建议按下面顺序推进：

1. 在 PC 端下载 PaddleSpeech `TTSArmLinux` 所需源码和模型包。
2. 把 Demo 先在板端以 CPU 方式跑起来。
3. 在 PC 端建立 Python 3.10/3.11 的 RKNN 转换环境。
4. 参考 RKNN `mms_tts` 的固定长度思路，先做 PaddleSpeech 声学模型的固定 shape 化。
5. 先只把声学模型转 RKNN，vocoder 先保留 CPU。
6. 声学模型 NPU 成功后，再尝试把 MB-MelGAN 转 RKNN。
7. 最终把整套推理封装成板端 C++ 程序。

## 9. 当前阶段的明确判断

如果你的要求是：

- “必须是 PaddleSpeech”
- “必须立刻上 RK3588 NPU”

那这不是现成部署题，而是一个带模型切分和固定 shape 改造的二次开发题。

如果你的要求是：

- “先在这块板上把 PaddleSpeech 跑起来”

那官方 `TTSArmLinux` 就是最快路径。

如果你的要求是：

- “必须优先利用 NPU”

那应先用 RKNN 官方 `mms_tts` 验证 NPU TTS 通路，再按相同方法改造 PaddleSpeech。

## 10. 板端验证 NPU 是否真的在工作

运行 RKNN 程序前后，可以在板端查看：

```bash
cat /sys/kernel/debug/rknpu/load
```

如果模型真的跑在 NPU 上，推理过程中 Core0/Core1/Core2 的负载会明显变化。
