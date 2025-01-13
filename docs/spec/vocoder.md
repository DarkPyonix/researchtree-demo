## Vocoder

> HiFi-GAN V1 turns the 80-bin mel spectrogram into a 22.05 kHz waveform. It is trained separately from the acoustic model.

Config: `configs/vocoder.yaml`. Code: `tinyspeech/models/hifigan.py`, `train_vocoder.py`.

### Generator

> Transposed convolutions upsample by 8, 8, 2 and 2 (256 times in total), each followed by residual blocks with kernels 3, 7 and 11. 512 initial channels, 13.9M parameters.

### Discriminators

> A multi-period discriminator (periods 2, 3, 5, 7 and 11) and a multi-scale discriminator (3 scales).

### Vocoder training

> 500k steps on mels computed from the recordings, then 50k steps on mels predicted by the acoustic model.

- Loss: least-squares GAN loss, plus mel L1 with weight 45 and feature matching with weight 2.
- AdamW, learning rate 2e-4, betas 0.8 and 0.99, learning rate multiplied by 0.999 every epoch.
- Batch size 16, segments of 8,192 samples.
- The fine-tuning stage uses mels predicted with the recorded durations, so each mel lines up with its waveform.
