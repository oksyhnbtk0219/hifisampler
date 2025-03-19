# hifisampler
A new UTAU resampler based on [pc-nsf-hifigan](https://github.com/openvpi/vocoders) for virtual singer.
### For Jinriki please use our [Hachimisampler](https://github.com/openhachimi/hachimisampler)
## Why is it called hifisampler?
Hifisampler was modified from [straycatresampler](https://github.com/UtaUtaUtau/straycat), replacing the original WORLD with pc-nsf-hifigan.
## What makes pc-nsf-hifigan different from traditional vocoders?
Pc-nsf-hifigan employs neural networks to upsample the input features, offering clearer audio quality than traditional vocoders. It is an improvement over the traditional nsf-hifigan, supporting f0 inputs that do not match mel, making it suitable for UTAU resampling.
## How to use? 
1. Install Python 3.10 and run the following commands (it's strongly recommended to use conda for easier environment management):
```
pip install numpy scipy resampy onnxruntime soundfile pyloudnorm
```
2. Download the CUDA version of PyTorch from the Torch website (If you're certain about only using the ONNX version, then downloading the CPU version of PyTorch is fine).
3. Fill out the config.toml. (config.toml, hifisampler.exe, hifiserver.py and launch_server.py should be in the same directory, for now. It is suggested to keep the original file hierarchy as in the compressed file)
4. Download the [release](https://github.com/openhachimi/hifisampler/releases), unzip it, and run 'hifiserver.py'.
5. Set UTAU's resampler to `hifisampler.exe`.
# Acknowledgments:
- [yjzxkxdn](https://github.com/yjzxkxdn)
- [openvpi](https://github.com/openvpi) for the pc-nsf-hifigan
- [MinaminoTenki](https://github.com/Lanhuace-Wan)
