# Intent

> Why TinySpeech exists, what we claim, and what we will not do. The current design is in SPEC.md. The evidence is in the experiment pull requests.

## Problem

Neural text-to-speech voices that sound natural are usually slow on a CPU, and the fast ones sound flat. We want one English voice that sounds natural, runs faster than real time on a laptop CPU, and lets a user raise or lower its pitch.

## Goals

- **Naturalness:** UTMOS 4.0 or higher on our 100 test sentences. The LJSpeech recordings themselves score 4.31 with the same predictor.
- **Intelligibility:** character error rate (CER) below 2% when Whisper large-v3 transcribes the synthesized test sentences.
- **Speed:** real-time factor (RTF) below 0.25 on 4 CPU threads, acoustic model and vocoder together. An RTF of 0.25 means one second of audio takes 0.25 seconds to make.
- **Control:** a user can shift the pitch of a whole sentence up or down.

## Claims

Each claim is one sentence that an experiment can support or refute. An experiment lists the claims it tests in its PR's YAML block, for example `claims: [N1]`.

- **N1:** Durations from an external forced aligner (Montreal Forced Aligner) are good enough that the model does not need to learn alignment itself.
- **N2:** Explicit pitch and energy predictors let a user shift sentence pitch by ±20% while UTMOS drops by less than 0.1.
- **N3:** A smaller HiFi-GAN (the V2 size, 0.9M parameters) keeps UTMOS within 0.1 of the V1 size (13.9M parameters) while running at least 3 times faster on a CPU.
- **N4:** For the same model size, phoneme input gives a lower CER than character input.

## Constraints

- Training: one 24 GB GPU. One acoustic model run must finish within 3 days.
- Data: public data only, LJSpeech 1.1 (one female speaker, 24 hours).
- Inference: plain PyTorch on CPU, no custom kernels.

## Non-goals

- More than one speaker, or voice cloning.
- Languages other than English.
- Streaming synthesis. We synthesize whole sentences.
- Emotion or speaking-style transfer.

## Open decisions

- **D1:** Should pitch be predicted once per mel frame or once per phoneme? The answer decides how pitch control (N2) is built.
- **D2:** Which vocoder ships: HiFi-GAN V1, a smaller HiFi-GAN, or a different design? The answer decides whether we can meet the speed goal (N3).
