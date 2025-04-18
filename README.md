# hifisampler

A new UTAU resampler based on [pc-nsf-hifigan](https://github.com/openvpi/vocoders) for virtual singer.

**For Jinriki please use our [Hachimisampler](https://github.com/openhachimi/hachimisampler).**

## Why is it called hifisampler?

Hifisampler was modified from [straycatresampler](https://github.com/UtaUtaUtau/straycat), replacing the original WORLD with pc-nsf-hifigan.

## What makes pc-nsf-hifigan different from traditional vocoders?

Pc-nsf-hifigan employs neural networks to upsample the input features, offering clearer audio quality than traditional vocoders. It is an improvement over the traditional nsf-hifigan, supporting f0 inputs that do not match mel, making it suitable for UTAU resampling.

## How to use?

1. Install Python 3.10 and run the following commands (it's strongly recommended to use conda for easier environment management):

    ```bash
    pip install -r requirements.txt
    ```

2. Download the CUDA version of PyTorch from the Torch website (If you're certain about only using the ONNX version, then downloading the CPU version of PyTorch is fine).
3. Fill the corresponding path information in `config.yaml`. If it's your first time using the program, modify the settings in `config.default.yaml`. The `config.yaml` file will be automatically generated upon the first run. (You need to place `config.default.yaml`, `hifiserver.py`, `hifisampler.exe`, and `launch_server.py` in the same directory. It is recommended to keep the original file structure unchanged after unpacking.)
4. Download the [release](https://github.com/openhachimi/hifisampler/releases), unzip it, and run 'hifiserver.py'.
5. Set UTAU's resampler to `hifisampler.exe`.

## Implemented flags

* **g:** Adjust gender/formants.  
  * Range: `-600` to `600` | Default: `0`
* **Hb:** Adjust breath/noise.  
  * Range: `0` to `500` | Default: `100`
* **Hv:** Adjust voice/harmonic.  
  * Range: `0` to `150` | Default: `100`
* **P:** Normalize loudness at the note level, targeting -16 LUFS. Enable this by setting `wave_norm` to `true` in your `config.yaml` file.
  * Range: `0` to `100` | Default: `100`
* **t:** Shift the pitch by a specific amount, in cents. 1 cent = 1/100 of a semitone.
  * Range: `-1200` to `1200` | Default: `0`
* **Ht:** Adjust tension.  
  * Range: `-100` to `100` | Default: `0`
* **A** Modulating the amplitude based on pitch variations, which helps creating a more realistic vibrato.
  * Range: `-100` to `100` | Default: `0`
* **G:** Force to regenerate feature cache (Ignoring existed cache).  
  * No value needed
* **Me:** Enable Mel spectrum loop mode.  
  * No value needed

_Note: The flags `B` and `V` were renamed to `Hb` and `Hv` respectively because they conflict with other UTAU flags but have different definitions._

## Acknowledgments

* [yjzxkxdn](https://github.com/yjzxkxdn)
* [openvpi](https://github.com/openvpi) for the pc-nsf-hifigan
* [MinaminoTenki](https://github.com/Lanhuace-Wan)
* [Linkzerosss](https://github.com/Linkzerosss)
* [MUTED64](https://github.com/MUTED64)
* [mili-tan](https://github.com/mili-tan)
