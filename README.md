# TinySpeech: a ResearchTree demo

This repository is a demo for [ResearchTree](https://github.com/DarkPyonix/researchtree), which turns Git branches and pull requests into a research tree. It holds a year of made-up research on **TinySpeech**, a small English text-to-speech system: a FastSpeech2-style acoustic model plus a GAN vocoder, trained on LJSpeech.

**Open the tree:** https://darkpyonix.github.io/researchtree/?repo=DarkPyonix/researchtree-demo

> [!NOTE]
> The project is invented for the demo. The code is readable and shaped like a real TTS codebase, but the training runs never happened: every metric, W&B link and date in the pull requests is illustrative.

## How to read this repository

| Where | What you find |
|---|---|
| `research` branch | The root of the tree: the initial implementation plus every adopted experiment. |
| Tags `research/v1` … | Research versions. Each tag marks the commit where adopted experiments were merged into `research`. |
| `experiment/<name>` branches | One experiment each. An experiment branched from another experiment is its child in the tree. |
| Pull requests | One per experiment: the lab note. The YAML block at the top holds the hypothesis, the change, the metrics and the dates; below it are the setup, a results table and the conclusion. |
| PR state | Open means running, merged means adopted, closed without merging means rejected. |
| [INTENT.md](INTENT.md) | Why the project exists: goals, claims (N1, N2, …), constraints, non-goals and open decisions. |
| [SPEC.md](SPEC.md) | The current design, decisions only. Every section starts with a one-line summary and links to the experiment that decided it. Part of it lives in [docs/spec/vocoder.md](docs/spec/vocoder.md), pulled in with an include line. |

### A good first tour

1. Read [INTENT.md](INTENT.md) for the goals and claims, then the summary line of each section in [SPEC.md](SPEC.md).
2. Open the tree and click a version stone. Each version shows which experiments were merged into it.
3. From one version, several experiments branch out at once, each testing a competing idea. Open the siblings side by side: the adopted one and the rejected ones all have a conclusion with numbers.
4. Look for claims that changed. A claim in INTENT.md is rewritten in the same pull request that refuted it.
5. Rejected experiments that proposed a design change still show their SPEC.md edit in their pull request. The edit never reached `research`.
6. The newest experiments are still running: open pull requests, some of them drafts.

Finished experiment branches are deleted when a new version is cut. Rejected branches are first archived into `research` with their changes reverted, so no commit is lost. The pull requests stay, so the tree does not change.

## Working with the code

```bash
uv sync
python scripts/extract_features.py          # mels, durations, pitch and energy from LJSpeech
python train.py --run-name my-experiment    # acoustic model; writes final metrics into the PR
python train_vocoder.py --stage gt          # vocoder on recorded mels, then --stage finetune
python synthesize.py "Hello there." -o hello.wav
```

Agents working in this repository use the ResearchTree skill in [.claude/skills/researchtree/SKILL.md](.claude/skills/researchtree/SKILL.md).
