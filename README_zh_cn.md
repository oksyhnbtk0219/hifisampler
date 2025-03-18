# hifisampler
 一个基于 [pc-nsf-hifigan](https://github.com/openvpi/vocoders)的新的utau重采样器。
## 为什么叫hifisampler?
hifisampler是由[straycatresampler](https://github.com/UtaUtaUtau/straycat) 修改而来，用pc-nsf-hifigan替换了原来的world
## pc-nsf-hifigan和其它传统vocoder有什么不同?
pc-nsf-hifigan采用神经网络对输入的特征进行上采样，音质比传统的vocoder更加清晰。
pc-nsf-hifigan是传统nsf-hifigan的改进，支持输入与mel不匹配的f0，因此可以用于utau的重采样。
## 如何使用? 
0.下载 [release](https://github.com/mtfotto/hifimisampler/releases) 解压，进入文件夹，如果有嵌套的话继续打开直到显示有hifiserver.py文件。         
1.安装miniconda，安装好后在刚刚打开的文件夹右击，点在终端中打开，输入      
```
conda create -n hifisampler python=3.8 -y
```
创建完成后，输入     
```
conda activate hifisampler
```
即可进入虚拟环境。  虚拟环境只需创建一次，以后直接进入有hifiserver.py的文件夹打开终端输入activate命令即可      
进入成功后会发现终端工作目录前面有(hifisamper)标志      
如果报错说conda不是内部外部命令，是环境变量没设置好，搜索：conda设置环境变量，按要求操作即可      
2.安装依赖，输入     
```
pip install numpy scipy resampy onnxruntime soundfile pyloudnorm
```
3. 在[torch官网](https://pytorch.org/)下载cuda版本的pytorch (如果你确定只使用onnx版，那么可以下载cpu版的pytorch)
具体安装方法：进入后往下滑，看到INSTALL PYTORCH以及一个表格，PyTorch Build选Stable、Your OS选你的操作系统、Package选pip、Language选python、Compute Platform如果要下载gpu版就选带cuda的，下载cpu版选cpu ，然后复制Run this Command右边表格里的命令到终端运行
4. 在config.toml中填入对应路径信息 (目前需要将config.toml, hifiserver.py以及hifisampler.exe三个文件放在同一目录下。建议解压后保持原文件结构不变)
5. 运行 'hifiserver.py'. 运行方法：在终端输入 python hifiserver.py
6. 将utau的重采样器设置为 `hifisampler.exe`.
# 感谢：
- [yjzxkxdn](https://github.com/yjzxkxdn)
- [openvpi](https://github.com/openvpi) for the pc-nsf-hifigan
