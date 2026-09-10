# TinySpeech design

> The current design of TinySpeech, decisions only. The measurements behind each decision are in the pull request linked from its section. Goals and claims are in INTENT.md.

## Text frontend

> Raw English text is lowercased, split into words and punctuation, and converted to ARPAbet phonemes with CMUdict.

- Normalization: lowercase, curly quotes replaced with straight quotes, runs of whitespace collapsed to one space.
- Numbers are spelled out before phonemization: money ("$5.50" is read "five dollars, fifty cents"), ordinals ("3rd"), years from 1100 to 2099 ("1963" is read "nineteen sixty-three"), decimals and whole numbers with or without thousands separators. Code: `tinyspeech/text/numbers.py`.
- Abbreviations followed by a period are expanded first: 18 titles and words such as "mr.", "dr." and "col.". "st." is always read "saint". Code: `tinyspeech/text/abbreviations.py`.
- Evidence: experiment/number-normalizer (#25), experiment/abbrev-expansion (#29).
- Each word is looked up in CMUdict 0.7b, keeping stress markers. A word missing from CMUdict is spelled out letter by letter.
- The punctuation marks `, . ? ! ; :` become their own tokens so the model can place pauses.
- Every phoneme keeps the index of the word it came from (`word_ids`).
- Code: `tinyspeech/text/frontend.py`.

## Acoustic model

> A non-autoregressive FastSpeech2-style model maps phoneme ids to an 80-bin log-mel spectrogram in one forward pass.

The parts run in this order: encoder, phrase context, duration predictor, pitch and energy, length regulator, decoder, postnet. Sizes are in `configs/acoustic.yaml`. Code: `tinyspeech/models/fastspeech2.py` and `tinyspeech/models/variance.py`.

### Encoder

> Four feed-forward Transformer blocks over phoneme embeddings: hidden size 256, 2 attention heads, convolution kernel 9.

### Phrase context

> Each phoneme also gets the vector of its word from a frozen BERT-base model that reads the whole sentence, so the model can tell which words the phrase stresses.

- BERT-base (uncased, 110M parameters, frozen) reads the sentence's words. A word's vector is the mean of its sub-word vectors from the last layer.
- A linear layer projects each 768-dimensional word vector to 256, and the result is added to the encoder output of every phoneme of that word. Word gaps and punctuation get nothing.
- Code: `tinyspeech/models/context.py`. Config: `context.model` in `configs/acoustic.yaml`.
- Evidence: experiment/prosody-bert (#26).

### Durations and alignment
<!-- id: duration-predictor -->

> Predicts how many mel frames each phoneme lasts. The training targets come from an alignment the model learns itself, so no external aligner is used.

- Duration predictor: two 1D convolutions (kernel 3) and a linear layer. It predicts log(frames + 1) and is trained with mean squared error.
- Aligner: the encoder output and the mel frames are each encoded with 1D convolutions. The soft alignment is a softmax over their negative squared distances. During the first 8,000 steps it is multiplied by a beta-binomial prior that favors the diagonal.
- Monotonic alignment search turns the soft alignment into a hard one. A phoneme's duration target is the number of frames the hard alignment gives it.
- Aligner losses: forward-sum (a CTC loss over the soft alignment) and a binarization loss that pulls the soft alignment toward the hard one.
- Pitch and energy targets are averaged per phoneme with these durations, during training.
- Every clip in the training split is used.
- At inference the predicted duration is rounded to whole frames, at least 1 frame per phoneme.
- Code: `tinyspeech/models/aligner.py`.
- Evidence: experiment/learned-alignment (#9), compared with experiment/mfa-g2p-lexicon (#10).

### Length regulator

> Repeats each phoneme's hidden vector as many times as its duration in frames.

### Variance adaptor
<!-- id: variance-adaptor -->

> Adds predicted pitch and energy to the hidden states, one value per phoneme, before the length regulator.

- Pitch: log-F0 from WORLD (DIO and StoneMask), interpolated through unvoiced frames, normalized with the corpus mean and standard deviation, then averaged over the frames of each phoneme.
- Energy: L2 norm of each STFT frame, normalized and averaged per phoneme the same way.
- One predictor each, with the same shape as the duration predictor, running on the encoder output.
- Each value is quantized into 256 bins and the bin's embedding is added to the phoneme's hidden vector, so all frames of a phoneme share it.
- Training uses the measured pitch and energy. Inference uses the predictions.
- Evidence: experiment/pitch-phoneme-level (#2), compared with experiment/pitch-cwt (#1) and experiment/energy-only (#3).

### Decoder

> Four Conformer blocks (hidden size 256, 2 attention heads, depthwise convolution kernel 31), then a linear layer to 80 mel bins.

- Each block: half-step feed-forward, self-attention, convolution module, half-step feed-forward, layer norm.
- Convolution module: pointwise convolution with GLU, depthwise convolution, batch norm, SiLU, pointwise convolution.
- Code: `tinyspeech/models/conformer.py`.
- Evidence: experiment/conformer-decoder (#11), experiment/conformer-4-layers (#15).

### Postnet

> Five 1D convolutions (512 channels, kernel 5) predict a correction that is added to the decoder's mel output.

<!-- include: docs/spec/vocoder.md -->

## Inference controls

> A user can shift the pitch of a whole sentence with one factor, `pitch_scale`, between 0.8 and 1.2, and change its speed with another, `rate`, between 0.8 and 1.25.

- `pitch_scale` multiplies the predicted F0 of every phoneme. Because pitch is stored as normalized log-F0, this adds log(pitch_scale) divided by the corpus standard deviation of log-F0 (0.183).
- The shifted value is clamped to the 2nd to 98th percentile of the speaker's normalized phoneme pitch ([-2.1, 2.6]), so no phoneme leaves the speaker's range.
- The measured pitch is never scaled, so training is unaffected.
- `rate` divides every predicted duration before rounding; 1.25 is 25% faster. Pauses at punctuation scale the same way.
- Command line: `python synthesize.py "text" --pitch-scale 1.1`.
- Evidence: experiment/pitch-control-range (#6), experiment/pitch-scale-clamp (#8).

## Training

> The acoustic model trains for 200k steps at batch size 32 on one 24 GB GPU, which takes about 3 days.

- Loss: L1 on the mel before and after the postnet, mean squared error on log-duration, pitch and energy, and the two aligner losses (see Durations and alignment).
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
