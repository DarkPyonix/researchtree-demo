---
name: researchtree
description: Run and record ML research experiments with ResearchTree, where every Git branch is an experiment and every pull request is its lab note. Use when starting, logging, concluding, or reviewing experiments in a repository that has a `research` root branch and `experiment/*` branches; when writing or editing an experiment PR body, a conclusion, a commit message, or the project's intent and spec documents; when reading past results to decide what to try next; or when cutting a new research version with `researchtree release`.
---

# ResearchTree

ResearchTree keeps a research project's full experiment history in Git and GitHub, with no database:

- **One branch = one experiment.** Each `experiment/<name>` branch holds one small, focused change that tests one hypothesis.
- **One pull request = one lab note.** The PR body records the hypothesis, the change, the metrics and the conclusion.
- **Branching = the tree.** An experiment branched from another experiment is its child.
- **PR state = the verdict.** Open means running, merged means adopted, closed without merging means rejected.

The viewer (https://darkpyonix.github.io/researchtree/, the VS Code extension `darkpyonix.researchtree`, or `researchtree serve`) draws these PRs as a tree. The `researchtree` Python package lets training scripts write into the PR and lets you (the agent) read the whole tree back as typed Python objects.

Follow the conventions below exactly. The viewer and the tools depend on them, and a wrong base branch or a broken YAML block puts an experiment in the wrong place or hides its data.

## Rule zero: write so a person can read it

**Everything you write here is read by people first**: PR bodies, conclusions, spec and intent documents, commit messages, comments. The whole point of ResearchTree is that a teammate (or the same researcher a month later) can open the tree and understand what was tried and why. A record nobody can read is worse than no record, because people stop reading the documents and then agents stop reading them too.

Agents tend to write compressed, clever sentences that only make sense with their private context. Do not do that. Before you write, picture a teammate who joined the project today and has not seen your conversation.

Rules:

1. **Say what happened in plain words.** Name the thing and the change: which function, which file, which setting, what it did before, what it does now. No metaphors, no personification ("the rollout stops paying for a reading", "the weight follows its module to the card").
2. **Lead with the conclusion, then the evidence.** First sentence: the result or the decision. Then the numbers. Then how you got them.
3. **One idea per sentence. Short sentences.** If a sentence needs a second read, split it.
4. **Define every label the first time.** Private shorthand like "plan C", "v7", "R1-e", "the gate", "Haan" means nothing to the reader. Write "the LoRA run on the top four layers (v7)" the first time, then "v7".
5. **Numbers need units and a comparison.** Not "loss 5.29" but "word loss 5.29, up from 5.04 on the original model (+0.25)". Say what counts as good or bad.
6. **No symbol chains as prose.** Arrows, slashes and plus signs (`A → B / C + D`) are fine in tables and code, not as a substitute for a sentence.
7. **Use the repository's language.** If the existing PR bodies and docs are in Korean, write in Korean. Match the terminology already used; do not rename things.
8. **Keep it as short as it can be while still complete.** Cut filler, hedging stacks and nested parentheses. Keep every fact a reader needs.
9. **Reread before you post.** Read it as the new teammate. If they would have to ask "what does this mean?", rewrite it.

Commit messages follow the same rules. Bad and good examples:

| Bad (do not write like this) | Good |
|---|---|
| `Fix: The rollout stops paying for a reading its step never uses` | `Fix: Skip the unused reward computation in the rollout step` |
| `Fix: The quantized weight is a buffer, so it follows its module to the card` | `Fix: Register the 4-bit weight as a buffer so .to(device) moves it` |
| `Feat: Every codebook holds its own role band, and the two roles start half apart` | `Feat: Give each codebook a separate role band, initialized half a band apart` |
| A trailer like `Haan: 5ad70be` with no explanation | Leave it out, or say what it is: `Ported from commit 5ad70be in the old repository` |

A conclusion section, bad and good:

> Bad: *R1-e confirmed R1-d; N1 reproduced at toy scale, en-listen collapse orthogonal to N2.*
>
> Good: *Adding a non-negativity penalty makes listening emerge. In the toy model, listening accuracy rose from 0.25 to 0.955 while speaking stayed at 1.000 (chance is 0.0625). The cost is English listening, which fell from 0.876 to 0.125, so one shared rotation cannot serve both languages. This does not affect the turn-taking claim (N2), which is about timing, not English recognition.*

## Intent, spec and evidence

Keep three kinds of writing apart. Mixing them is what makes research documents grow until nobody reads them.

| Kind | Answers | Where | Changes |
|---|---|---|---|
| **Intent** | Why are we doing this? What do we claim? What will we not do? | `INTENT.md` at the root of `research` | Rarely, on purpose |
| **Spec** | What is the system right now? | `SPEC.md` at the root of `research` (one file; split only when a section outgrows a screen) | When an experiment changes the design |
| **Evidence** | What did we try, what happened, what did we learn? | The experiment's PR body | Every experiment |

Plans and to-do lists are not documents: each open item is a draft experiment PR.

Some repositories keep the same content under other names (for example `PROJECT.md` and `MOTIVATION.md` for intent, `PHASE.md` and `IMPLEMENTATION.md` for spec). Use what the repository has; do not create a parallel set of files unless the user asks for it. The paths are set in `.researchtree.yml` at the root of `research`, which the viewer, the CLI and the Python API all read:

```yaml
spec: docs/PHASE.md
intent: docs/INTENT.md
```

Read them from the terminal instead of opening files by hand:

```bash
researchtree spec --summary          # the whole design on one page: section titles and their summary lines
researchtree spec --intent           # the intent document at the latest version
researchtree spec --claims           # each claim with the experiments that tested it and their status
researchtree spec --diff             # sections changed since the previous version, with line diffs
researchtree spec --experiment NAME  # what one experiment's branch changed in the spec
researchtree spec --history KEY      # the versions where one section was added or changed
researchtree spec --check            # sections without a summary line, sections over 40 lines, broken includes
```

In Python: `research.spec()`, `research.version("v3").spec()`, `spec.summary()`, `version.spec_changes()`, `experiment.spec_changes()`, `research.spec_history(key)`, `research.claims()`, `research.intent()`.

### Intent

- A short statement of the problem, the goals, the constraints and the non-goals.
- **Claims** with stable ids (`N1`, `N2`, …), each one sentence a reader can agree or disagree with, written as headings like `### N1. A duration predictor removes the need for an attention aligner`. An experiment that tests a claim lists it in its YAML: `claims: [N1]`. The viewer shows them on the experiment, and `researchtree spec --claims` groups experiments by claim.
- **Open decisions**: questions whose answer would change a claim.
- When evidence contradicts a claim, say so to the user and propose the edit to `INTENT.md`. Never leave a claim in place that the adopted results refute.

### Spec

- **Decisions only.** What the system does, the formula it uses, the value chosen. The derivation, the measurements and the comparison tables go in the PR, and the spec links to it: `Evidence: experiment/nonneg-penalty (#24)`.
- **Every section starts with a one-line summary** (a `>` quote line), so the titles plus summaries read as a one-page overview of the whole design.
- **Only the current state.** No "previously we used X", no "tried Y, rejected". History lives in Git and in the PRs.
- **Section titles name features**, one feature per `##` or `###` section, so an experiment's diff shows which features it added or changed. Keep titles stable: a section is recognized by its title under its parent. Before renaming one, give it an id on the heading line, `## Vocoder <!-- id: vocoder -->`, so its history continues (the comment is invisible on GitHub).
- **Split a section into its own file only when it outgrows a screen**, and leave `<!-- include: path/to/part.md -->` on its own line where it was (relative to the including file). The tools assemble the included files back into one document.
- An experiment that changes the design edits the spec **on its own branch**, in the same PR. Adopting it merges the spec change; rejecting it keeps the change out of `research`. Experiments that only tune values without changing the design do not touch the spec.
- Moving text between files or sections without changing its meaning is a refactor, not an experiment. Keep it in its own commit.

### Evidence (the PR body)

- Setup, results table, interpretation, conclusion, in that order, following Rule zero.
- Put the numbers and the reasoning here in full. This is where a reader goes to check a decision in the spec.
- Link back to the spec sections the experiment changed and the claims it tested.

### Before proposing an experiment

Run `researchtree spec --intent`, `researchtree spec --summary` and `researchtree spec --claims` before proposing an experiment, then read the memory (below). Propose experiments that test a claim or answer an open decision, and say which. After an adopted experiment, run `researchtree spec --check`.

## Setup check

Before doing anything else, confirm the repository uses ResearchTree and that you can reach GitHub:

```bash
git branch -a | grep -E 'research|experiment/'   # root branch and experiment branches exist
git tag --list 'research/v*'                       # research versions (may be empty at first)
researchtree --version                             # package installed (pip install researchtree)
researchtree memory                                # prints the manifest; fails if the token or repo is wrong
```

- Authentication: `researchtree login` (GitHub Device Flow, token stored in the OS keyring or a `0600` file), or the `RESEARCHTREE_TOKEN` environment variable. Public repositories can be read without a token.
- The repository comes from the `origin` remote; override with `RESEARCHTREE_REPO=owner/name`.
- The root branch (`research`) and experiment prefix (`experiment/`) are defaults. If the team changed them, set `RESEARCHTREE_ROOT` and `RESEARCHTREE_PREFIX` (for example `trunk` and `exp/`) and read every `research` / `experiment/` below with those names.
- `main`, `develop` and every other branch are ignored by the tree. Never open experiment PRs against them.

## Before you propose an experiment: read the memory

Always start from what has already been tried. Load the tree and work top-down: short manifest first, then only the parts you need.

```python
import researchtree as rt

research = rt.load()                 # or rt.load("owner/name")
print(research.manifest())           # versions (islands), best results, running work, alerts
```

| Level | Call | Shows |
|---|---|---|
| Summary | `research.manifest()` | Each version with experiment counts and best metrics, running experiments, alerts, suggested next calls |
| Version | `research.version("v3").describe()` | The experiments merged into it, its island's experiments, best per metric |
| Experiment | `research["duet-mix"].describe()` | Hypothesis, change, metrics with Δ against the parent, conclusion, children, W&B link |
| Body | `e.body`, `e.sections`, `e.conclusion`, `e.section("Notes")` | PR body Markdown (without the YAML block), split by `##` headings |
| Raw | `e.commits()`, `e.comments()`, `e.files()` | Branch commits, PR comments, changed files, fetched lazily from GitHub |

Objects:

- `research.versions` (oldest first), `research.root`, `research.latest`, `research.version("v3")` (also accepts `"research@v3"`).
- `research.experiments` is an `Experiments` list sorted by start time. `research["name"]` finds an experiment by short name, branch name or PR number.
- `Version`: `name`, `tag`, `date`, `sha`, `merged`, `experiments`, `started_here`, `grown_from`, `metrics`, `next`, `previous`, `parent`, `children`.
- `Experiment`: `name`, `branch`, `number`, `url`, `title`, `author`, `status` (`running` | `adopted` | `rejected`), `draft`, `state`, `finished`, `hypothesis`, `change`, `metrics`, `wandb`, `tags`, `meta` (the whole YAML block), `started`, `ended`, `version`, `produces`, `parent`, `children`, `siblings`, `ancestors()`, `descendants()`, `path`, `depth`, `baseline`, `delta(key)`, `improved(key)`, `warnings`.
- `Experiments` is a plain list with helpers: `.where(status="adopted", version="v3")`, `.where(lambda e: ...)`, `.with_metric(key)`, `.sort_by(key, reverse=...)`, `.best(key, lower=None)` (direction guessed from the name: `loss`, `wer`, `ppl`, `latency` are lower-is-better; `acc`, `f1`, `bleu`, `auc` are higher-is-better), `.search(text)`, `.names`, `.metric(key)`, `.table(*metrics)`.
- `research.running`, `research.search(text)`, `research.metric_keys`, `research.check()` (alerts: stale running experiments, adopted without metrics, closed without a conclusion, adopted regressions, record warnings), `research.to_dict()`.

Answer questions with computation rather than by reading every PR, for example:

```python
research.experiments.where(status="adopted").best("val_loss")
{v.name: len(v.experiments.where(status="rejected")) for v in research.versions}
research.latest.experiments.table("val_loss", "wer")
[e.name for e in research.search("warmup") if e.status == "rejected"]   # was this idea already tried?
```

The same data is available from the terminal: `researchtree memory` (manifest), `researchtree memory v3`, `researchtree memory <experiment>`, `researchtree memory --check`, `researchtree memory --json`.

Before proposing a new experiment, check that the same hypothesis has not already been rejected, and choose the right parent: the latest version for a fresh idea, or the experiment whose change you want to build on.

## Starting an experiment

1. Pick the base and create the branch in its own worktree (see the next section).
   - From a research version (a new line of work): branch from the version tag, not from the tip of `research`.
     ```bash
     git fetch --tags
     git worktree add -b experiment/warmup-cosine ../myrepo-warmup-cosine research/v3
     ```
   - From another experiment (a derived idea): branch from that experiment's branch.
     ```bash
     git worktree add -b experiment/depth-lr-half-warmup ../myrepo-depth-lr-half-warmup experiment/depth-lr-half
     ```
   Then work, commit and push from the new directory.
2. Name the branch `experiment/<short-kebab-name>`. Keep names flat (one level under `experiment/`); the hierarchy comes from the parent, not the name.
3. Test exactly one hypothesis. Two ideas are two branches.
4. Push and open a **draft PR** right away, so the tree shows the experiment as running:
   - Base branch: the parent experiment branch, or `research` for an experiment that starts from a version.
   - Title: the hypothesis in one line.
   - Body: the YAML block below at the very top.

   ```bash
   git push -u origin experiment/warmup-cosine
   gh pr create --draft --base research --title "Cosine decay after warmup lowers final val_loss" --body-file body.md
   ```

## Working on several experiments at once: git worktree

Research usually has several experiments running at the same time: one is training, another is being written, a third is being analyzed. **Do not switch branches back and forth in one checkout.** Switching changes the files under a running training job, mixes uncommitted work between experiments, and makes `rt.log()` write to whichever branch happens to be checked out. Give each experiment its own working directory with `git worktree`.

- **One worktree per experiment branch.** The main checkout stays on `research` (or whatever it was on); each experiment gets a sibling directory.
  ```bash
  git fetch --tags
  git worktree add -b experiment/warmup-cosine ../myrepo-warmup-cosine research/v3   # new experiment from a version
  git worktree add -b experiment/lr-half-warmup ../myrepo-lr-half-warmup experiment/lr-half   # derived experiment
  git worktree add ../myrepo-depth-lr-half experiment/depth-lr-half                  # existing branch
  git worktree list
  ```
- **Name the directory after the experiment** (`<repo>-<experiment name>`), next to the main checkout rather than inside it, so tools that scan the repository do not see the other experiments' files. If the team prefers a folder inside the repository, use `.worktrees/<name>` and make sure it is in `.gitignore`.
- **Run everything for an experiment from its worktree:** training, `rt.log()`, commits, `git push`, `gh pr create`. `rt.log()` finds the PR from the branch checked out in the current directory, so each training job writes to its own PR.
- **Each worktree is a separate directory**, so it needs its own environment: run `uv sync` (or the project's setup) in it. Keep large shared files (model weights, datasets, caches) in one shared location configured by environment variables such as `HF_HOME`, not copied into each worktree.
- **A branch can be checked out in only one worktree.** To look at another experiment's code, read it with `git show experiment/x:path/to/file` or open its worktree; do not check it out a second time.
- **Clean up when an experiment is finished** (merged or closed) and nothing is running in it:
  ```bash
  git worktree remove ../myrepo-warmup-cosine
  git worktree prune
  ```
- **Before `researchtree release`, remove the worktrees of finished experiments** and run the release from the main checkout. The release switches to `research`, archives and deletes finished branches, and Git refuses to do that to a branch checked out in another worktree. `researchtree release --yes` checks this first and lists the `git worktree remove` commands to run. Worktrees of running experiments can stay.
- Never delete a worktree directory by hand while a job is running in it, and never use `--force` to remove a worktree that has uncommitted changes without asking the user.

## The PR body

The first ```` ```yaml ```` block at the **top** of the PR body is the record. Only the first YAML block is read; everything below it is free Markdown.

````markdown
```yaml
parent: research@v3
hypothesis: Cosine decay after warmup lowers the final val_loss
change: lr schedule linear → cosine (warmup 2k steps kept)
metrics:
  val_loss: 2.31
  wer: 0.184
wandb: https://wandb.ai/team/project/runs/abc123
status: running
tags: [lr, schedule]
```

## Conclusion
Converged 8% faster to the same val_loss. Adopted.

## Notes
Free-form notes, tables, plots, links.
````

| Field | | Meaning |
|---|---|---|
| `hypothesis` | required | One sentence you want to test |
| `parent` | recommended | Parent experiment branch (`experiment/depth-lr-half`) or `research@vN` for an experiment that starts from a version. Falls back to the PR's base branch |
| `change` | recommended | What changed in code or config, in one line |
| `metrics` | optional | Final metrics, `name: number` or `name: string`. Reuse the key names already in the tree (`research.metric_keys`) |
| `wandb` | optional | Link to curves or other external records |
| `status` | optional | `running`, `adopted` or `rejected`. Only needed to override what the PR state implies |
| `tags` | optional | Labels such as `lr`, `data`, `arch` |
| `started`, `ended` | optional | Dates (`2026-08-01` or ISO 8601) when the PR dates are wrong, for example for records moved from elsewhere |

Rules for editing a PR body:

- Keep the YAML block valid and first in the body. If it is broken, fix it carefully by hand; the tools refuse to overwrite a broken block.
- Preserve unknown fields, comments and key order, and never delete the Markdown below the block.
- Prefer the Python API (below) for metrics, fields and the conclusion: it edits only the YAML block and keeps everything else exactly as it was.
- The conclusion section may be titled `## Conclusion` or `## 결론`; both are recognized. Keep whichever the repository already uses.

How the tree is resolved, so you can predict where an experiment appears:

- Status: YAML `status`, else merged → `adopted`, closed without merging → `rejected`, otherwise `running` (a draft PR is drawn as a sprout).
- Parent: YAML `parent`, else the PR's base branch. Base `research` attaches to the latest version tagged before the PR was opened. Any other base (for example `main`) makes the node an orphan.
- Only PRs whose head is `experiment/*` appear. A PR from `experiment/*` into `main` or `develop` is hidden unless its YAML has a `parent`.

## Logging from the training script

The training project installs the same package (`pip install researchtree` or `uv add researchtree`). On an `experiment/*` branch with an open PR, these calls update that PR:

```python
import researchtree as rt

rt.log(val_loss=2.31, wer=0.184)     # merge into `metrics`; a None value deletes a key
rt.set(wandb=run.url, tags=["lr"])   # set any YAML field; None deletes it
rt.conclude("rejected", "Divergence went down but convergence was too slow.", heading="Conclusion")
```

- These calls never raise into training. If the branch is not an experiment branch, no PR is found, there is no token or GitHub fails, they emit a `RuntimeWarning` and return.
- In distributed runs they act only on rank 0 (`RANK` / `LOCAL_RANK`).
- `conclude(status, text, heading=None)`: `status` must be `running`, `adopted` or `rejected`. An existing conclusion section is replaced in place; a new one uses `heading` (default `결론`; pass `heading="Conclusion"` for English).
- Log final metrics, not per-step values. Use the metric key names already used in the tree so experiments stay comparable.

## Concluding an experiment

1. Write the conclusion in the PR body: what happened, the numbers, and why the hypothesis holds or not. Negative results matter; say clearly what was learned.
2. Set the final metrics.
3. Then decide:
   - **Adopt**: merge the PR into its parent branch (`research` or the parent experiment). If it was merged into `research`, the next step is a new version (below).
   - **Reject**: close the PR without merging. Do not delete the branch.
4. When several competing ideas were tried and one is adopted, close the others with their conclusions so every result is final.

Do not merge or close PRs, push to `research`, or create tags unless the user asked for it. Report the result and let the user decide on adoption.

## Cutting a research version

`research` is the main line where adopted experiments are merged, and each merged state gets a version tag: `research/v1`, `research/v2`, … (`research/v1.1` also works). Tags are prefixed with the root branch name so they never clash with release tags like `v1` on `main`; bare `vN` tags are ignored. In YAML and in the viewer the version is called `v2`, and `parent: research@v2` points at the tag `research/v2`.

Use `researchtree release`, which also cleans up finished branches:

```bash
git switch research
git merge --no-ff experiment/warmup-cosine      # adopt (or merge the PR on GitHub)
researchtree release                            # plan only: what gets deleted, archived or kept
researchtree release --yes                      # archive merges → tag research/vN → push → delete branches
researchtree release v4 --yes                   # choose the version name explicitly
```

What the release does:

- Branches already contained in `research` (adopted and merged, or merged into such a branch) are deleted.
- Finished branches not in `research` (rejected, or adopted into a parent that never reached `research`) are **archived**: their commits are reverted on the branch (a revert, not a reset), the branch is merged into `research` with `--no-ff`, and then it is deleted. The code in `research` does not change, but the history stays reachable.
- Branches with running descendants on the same island are kept until those finish.
- PRs stay on GitHub, so the tree is unchanged.

Always run the plan first and show it to the user before running `--yes`. Keep "Automatically delete head branches" off in the repository settings; deleting a parent branch with running children makes GitHub retarget the children's PRs and the tree loses the link.

## Guardrails

- Every PR body, conclusion, spec edit and commit message follows Rule zero. If a teammate would need to ask what it means, rewrite it before posting.
- Never rewrite history on `research` or on experiment branches that have a PR (no force push, no reset of pushed commits).
- Never delete experiment branches by hand; `researchtree release` does it safely.
- Never put tokens in PR bodies, commits, logs or command lines that get saved.
- One hypothesis per branch, one PR per branch, one YAML block at the top of each PR.
- Base branch and `parent` must agree: an experiment that starts from `research/v3` has base `research` and `parent: research@v3`.
- If `research.check()` or `researchtree memory --check` reports alerts (stale running experiments, missing metrics, closed without a conclusion), mention them to the user.

## Quick reference

| Task | Command |
|---|---|
| Sign in | `researchtree login` (or `RESEARCHTREE_TOKEN`) |
| Read the memory | `researchtree memory`, `researchtree memory <name>`, `rt.load()` |
| Read the design | `researchtree spec --summary`, `researchtree spec --intent`, `researchtree spec --claims` |
| Check the record | `researchtree memory --check`, `research.check()` |
| Open the tree | `researchtree open` (hosted viewer) or `researchtree serve` (local) |
| Work on an experiment | `git worktree add -b experiment/<name> ../<repo>-<name> <base>` |
| Log from training | `rt.log(...)`, `rt.set(...)`, `rt.conclude(...)` |
| Cut a version | `researchtree release`, then `researchtree release --yes` |
| Update this skill | `researchtree skill install` |

Guide: https://darkpyonix.github.io/researchtree/guide/
