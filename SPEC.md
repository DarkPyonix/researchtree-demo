# TinySpeech design

> The current design of TinySpeech, decisions only. The measurements behind each decision are in the pull request linked from its section. Goals and claims are in INTENT.md.

## Text frontend

> Raw English text is lowercased, split into words and punctuation, and converted to ARPAbet phonemes with CMUdict.

- Normalization: lowercase, curly quotes replaced with straight quotes, runs of whitespace collapsed to one space.
- Numbers are not expanded: each digit is read on its own ("1963" is read "one nine six three").
- Each word is looked up in CMUdict 0.7b, keeping stress markers. A word missing from CMUdict is spelled out letter by letter.
- The punctuation marks `, . ? ! ; :` become their own tokens so the model can place pauses.
- Every phoneme keeps the index of the word it came from (`word_ids`).
- Code: `tinyspeech/text/frontend.py`.

## Acoustic model

> A non-autoregressive FastSpeech2-style model maps phoneme ids to an 80-bin log-mel spectrogram in one forward pass.

The parts run in this order: encoder, duration predictor, length regulator, pitch and energy, decoder, postnet. Sizes are in `configs/acoustic.yaml`. Code: `tinyspeech/models/fastspeech2.py` and `tinyspeech/models/variance.py`.

### Encoder

> Four feed-forward Transformer blocks over phoneme embeddings: hidden size 256, 2 attention heads, convolution kernel 9.

### Duration predictor
<!-- id: duration-predictor -->

> Predicts how many mel frames each phoneme lasts. It is trained on durations from Montreal Forced Aligner.

- Two 1D convolutions (kernel 3) and a linear layer. It predicts log(frames + 1) and is trained with mean squared error.
- Training targets come from Montreal Forced Aligner 2.0 with the `english_us_arpa` acoustic model, converted to frame counts at hop size 256. Clips that the aligner fails on are left out of training.
- At inference the prediction is rounded to whole frames, at least 1 frame per phoneme.

### Length regulator

> Repeats each phoneme's hidden vector as many times as its duration in frames.

### Variance adaptor
<!-- id: variance-adaptor -->

> Adds predicted energy to the hidden states, one value per mel frame. Pitch is not predicted; the decoder infers it from the phonemes and energy.

- Pitch: log-F0 from WORLD (DIO and StoneMask), interpolated through unvoiced frames and normalized with the corpus mean and standard deviation.
- Energy: L2 norm of each STFT frame, normalized the same way.
- One predictor each, with the same shape as the duration predictor, running after the length regulator.
- Each value is quantized into 256 bins and the bin's embedding is added to the hidden states.
- Training uses the measured pitch and energy. Inference uses the predictions.

### Decoder

> Six feed-forward Transformer blocks of the same size as the encoder, then a linear layer to 80 mel bins.

### Postnet

> Five 1D convolutions (512 channels, kernel 5) predict a correction that is added to the decoder's mel output.

<!-- include: docs/spec/vocoder.md -->

## Training

> The acoustic model trains for 200k steps at batch size 32 on one 24 GB GPU, which takes about 3 days.

- Loss: L1 on the mel before and after the postnet, plus mean squared error on log-duration, pitch and energy.
- Optimizer: Adam (betas 0.9 and 0.98) with the Noam schedule: 4,000 warmup steps, peak learning rate 1e-3.
- Data: LJSpeech 1.1, 13,100 clips. 100 clips are held out for validation and 100 for test, chosen with a fixed seed.
- The vocoder is trained separately (see Vocoder training).

## Evaluation

> Every experiment reports the same five metrics on the same held-out sentences, so any two experiments can be compared.

| Metric | What it measures | Better |
|---|---|---|
| `mel_l1` | L1 distance to the recorded mel on the 100 validation clips, using the recorded durations, pitch and energy | lower |
| `mcd` | Mel cepstral distortion to the recording on the 100 test sentences, in dB, 13 coefficients, frames aligned with DTW | lower |
| `utmos` | UTMOS22 predicted mean opinion score (1 to 5) on the 100 test sentences | higher |
| `cer` | Character error rate, in percent, of Whisper large-v3 transcripts of the synthesized test sentences against the normalized text | lower |
| `rtf` | Synthesis time divided by audio duration, acoustic model and vocoder, batch size 1, 4 threads of an Intel Xeon 8375C | lower |

- Test sentences are synthesized from the raw text in column 2 of `metadata.csv`, with numbers and abbreviations as written, the way a user would type them.
- A difference smaller than the spread between three seeds of the same model counts as noise: 0.004 for `mel_l1`, 0.03 for `utmos` and 0.3 points for `cer`.
- Code: `evaluate.py`. The training script writes the final metrics into the experiment's pull request with `rt.log()`.
