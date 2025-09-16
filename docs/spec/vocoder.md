## Vocoder

> An iSTFTNet generator turns the 80-bin mel spectrogram into a 22.05 kHz waveform: two upsampling stages, then an inverse STFT. It is trained separately from the acoustic model.

Config: `configs/vocoder.yaml`. Code: `tinyspeech/models/istftnet.py` (generator), `tinyspeech/models/hifigan.py` (residual blocks and discriminators), `train_vocoder.py`.

### Generator

> Transposed convolutions upsample by 8 and 8, each followed by HiFi-GAN residual blocks (kernels 3, 7 and 11). The last layer predicts the magnitude and phase of a 16-point STFT with hop 4, and an inverse STFT gives the waveform. 512 initial channels, 8.2M parameters.

- Magnitude is the exponential of the network output; phase is pi times its sine, so it stays in [-pi, pi].
- Evidence: experiment/istftnet (#19), compared with experiment/hifigan-v2 (#18).

### Discriminators

> A multi-period discriminator (periods 2, 3, 5, 7 and 11) and a multi-resolution spectrogram discriminator (STFT sizes 512, 1024 and 2048).

- The resolution discriminator runs 2D convolutions over the magnitude spectrogram at each STFT size (hops 50, 120 and 240).
- Evidence: experiment/istft-mrd (#23).

### Vocoder training

> 500k steps on mels computed from the recordings, then 50k steps on mels predicted by the acoustic model.

- Loss: least-squares GAN loss, plus mel L1 with weight 45 and feature matching with weight 2.
- AdamW, learning rate 2e-4, betas 0.8 and 0.99, learning rate multiplied by 0.999 every epoch.
- Batch size 16, segments of 8,192 samples.
- The fine-tuning stage uses mels predicted with the recorded durations, so each mel lines up with its waveform.
