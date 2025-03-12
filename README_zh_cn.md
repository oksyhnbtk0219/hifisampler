# hifisampler
 一个基于 [pc-nsf-hifigan](https://github.com/openvpi/vocoders)的新的utau重采样器。
## 为什么叫hifisampler?
hifisampler是由[straycatresampler](https://github.com/UtaUtaUtau/straycat) 修改而来，用pc-nsf-hifigan替换了原来的world
## pc-nsf-hifigan和其它传统vocoder有什么不同?
pc-nsf-hifigan采用神经网络对输入的特征进行上采样，音质比传统的vocoder更加清晰。
pc-nsf-hifigan是传统nsf-hifigan的改进，支持输入与mel不匹配的f0，因此可以用于utau的重采样。
## 如何使用? 
1. 安装python3.10并运行下面的指令（墙裂建议使用conda以方便管理环境）
```
pip install numpy scipy resampy pyworld torch onnxruntime praat-parselmouth soundfile pyloudnorm
```
2. 在torch官网下载cuda版本的pytorch (如果你确定只使用onnx版，那么可以下载cpu版的pytorch)
3. 下载 [release](https://github.com/mtfotto/hifimisampler/releases) 解压后运行 'hifisampler.py'.
4. 将utau的重采样器设置为 `hifisampler.exe`.
# 感谢：
- [yjzxkxdn](https://github.com/yjzxkxdn)
- [openvpi](https://github.com/openvpi) for the pc-nsf-hifigan
