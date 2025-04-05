# hifisampler

一个基于 [pc-nsf-hifigan](https://github.com/openvpi/vocoders)的新的 utau 重采样器。

## 为什么叫 hifisampler?

hifisampler 是由[straycatresampler](https://github.com/UtaUtaUtau/straycat) 修改而来，用 pc-nsf-hifigan 替换了原来的 world

## pc-nsf-hifigan 和其它传统 vocoder 有什么不同?

pc-nsf-hifigan 采用神经网络对输入的特征进行上采样，音质比传统的 vocoder 更加清晰。
pc-nsf-hifigan 是传统 nsf-hifigan 的改进，支持输入与 mel 不匹配的 f0，因此可以用于 utau 的重采样。

## 如何使用?

0. 下载 [release](https://github.com/mtfotto/hifimisampler/releases) 解压，进入文件夹，如果有嵌套的话继续打开直到显示有 hifiserver.py 文件。

1. 安装 miniconda ，安装好后在刚刚打开的文件夹右击，点在终端中打开，输入

   ```bash
   conda create -n hifisampler python=3.8 -y
   ```

   创建完成后，输入

   ```bash
   conda activate hifisampler
   ```

   即可进入虚拟环境。虚拟环境只需创建一次，以后直接进入有 hifiserver.py 的文件夹打开终端输入 activate 命令即可  
   进入成功后会发现终端工作目录前面有 ( hifisamper ) 标志  
   如果报错说 conda 不是内部外部命令，是环境变量没设置好，搜索： conda 设置环境变量，按要求操作即可
2. 安装依赖，输入

   ``` bash
   pip install numpy scipy resampy onnxruntime soundfile pyloudnorm
   ```

3. 在 [torch 官网] (<https://pytorch.org/>) 下载 cuda 版本的 pytorch ( 如果你确定只使用 onnx 版，那么可以下载 cpu 版的 pytorch )
   具体安装方法：进入后往下滑，看到 INSTALL PYTORCH 以及一个表格，PyTorch Build 选 Stable , Your OS 选你的操作系统 , Package 选 pip , Language 选 python , Compute Platform 如果要下载 gpu 版就选带 cuda 的，下载 cpu 版选 cpu ，然后复制 Run this Command 右边表格里的命令到终端运行
4. 在 config.yaml 中填入对应路径信息，如果是首次使用，则在 config.default.yaml 中修改，首次运行时会自动生成 config.yaml 文件。 (需要将 config.default.yaml ，hifiserver.py ， hifisampler.exe 以及 launch_server.py 四个文件放在同一目录下。建议解压后保持原文件结构不变)
5. 每次使用前运行 'hifiserver.py'。
   在终端输入

   ```bash
   conda activate hifisampler
   python hifiserver.py
   ```

   _若已设置好配置文件则每次使用时这一步可跳过_
6. 将 utau 的重采样器设置为 `hifisampler.exe`

## 已实现的 flags

- **g:** 调整性别/共振峰。
  - 范围: `-600` 到 `600` | 默认: `0`
- **B:** 控制气息/噪波成分的量。
  - 范围: `0` 到 `500` | 默认: `100`
- **V:** 控制发声/谐波成分的量。
  - 范围: `0` 到 `150` | 默认: `100`
- **P:** 以 -16 LUFS 为基准进行音符粒度的响度标准化。需要在 config.yaml 中设置 `loudness_norm` 为 `True`。
  - 范围: `0` 到 `100` | 默认: `100`
- **G:** 强制重新生成特征缓存（忽略已有缓存）。
  - 无需数值
- **Me:** 为长音启用 Mel 频谱循环模式。
  - 无需数值

## 感谢

- [yjzxkxdn](https://github.com/yjzxkxdn)
- [openvpi](https://github.com/openvpi) for the pc-nsf-hifigan
- [MinaminoTenki](https://github.com/Lanhuace-Wan)
- [Linkzerosss](https://github.com/Linkzerosss)
