# DSB-SC, AM, SSB, FM 调制解调仿真

本项目用于完成《信号分析与处理》模拟调制解调实验，覆盖：

- DSB-SC 调制、相干解调、AWGN 高/低信噪比对比
- AM 调制、包络检波、AWGN 高/低信噪比对比
- SSB 上边带调制、相干解调、AWGN 高/低信噪比对比
- FM 调制、鉴频解调、AWGN 高/低信噪比对比

## 运行

安装依赖：

```powershell
pip install -r requirements.txt
```

直接运行：

```powershell
python am_dsb_ssb_fm_simulation.py
```

如果 `myvioce/` 目录下存在 WAV 文件，脚本会优先自动读取该录音作为基带信号 `m(t)`。当前默认会读取：

```text
myvioce/myvoice.wav
```

如果该目录下没有 WAV 文件，脚本才会退回到默认合成音频基带。

使用自己的 WAV 音频作为个性化基带：

```powershell
python am_dsb_ssb_fm_simulation.py --audio .\your_audio.wav
```

## 输出

运行后结果保存在 `outputs/`：

- `summary.md`：参数、FM 带宽估算、输入输出信噪比表格和简要说明
- `snr_results.csv`：信噪比计算结果
- `*_summary.png`：每种调制方式的基带波形、已调波形、频谱和高/低信噪比解调输出
- `baseband_message.wav`：基带音频
- `*_demod.wav`：各方法在高/低信噪比下的解调音频

## 默认参数

- 采样率：48 kHz
- 载波频率：8 kHz
- 基带带宽：3 kHz
- AM 调制度：0.65
- FM 最大频偏：1.8 kHz
- 高信噪比：30 dB
- 低信噪比：8 dB
