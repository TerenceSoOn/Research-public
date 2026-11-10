# Process Note

This file records development decisions, rationale, affected code paths, and expected experiment impact.

## 2026-03-18 1
- Decision: EMPO rewards for filtering-style methods are recomputed after selection instead of being inherited from the original oversampled pool.
- Reason: the tutor identified that training on K selected samples with rewards computed from M samples is conceptually wrong.
- Affected code path: `coding_edis_empo/src/edis_reward_manager.py`
- Expected experiment effect: filtering-style methods should now optimize the selected training pool directly; previous filtering results are not directly comparable.
- Open questions / follow-up runs: rerun the filtering and oversampled baselines after this fix before drawing conclusions.

## 2026-03-18 2
- Decision: split vote-time weighting from trainer-side advantage weighting.
- Reason: the current `edis_weighting` implementation modifies GRPO advantages after reward computation, while the new tutor note describes weighting during winner selection / voting.
- Affected code path: `coding_edis_empo/src/edis_weighting_trainer.py`, `coding_empo/project_EMPO/EMPO/verl-empo/verl/utils/reward_score/empo/compute_metrics.py`
- Expected experiment effect: avoid mixing two different mechanisms under one “weighting” label and make experiment naming clearer.
- Open questions / follow-up runs: compare `edis_adv_weighting` against vote-time `edis_borda` under matched compute.

## 2026-03-18 3
- Decision: add explicit per-response vote artifacts (`cluster_id`, `cluster_size`, `vote_supports_winner`, `vote_tag`, winner label/score) to EMPO voting output.
- Reason: later selection logic and analysis should use explicit cluster / winner membership instead of inferring correctness from reward magnitude.
- Affected code path: `coding_empo/project_EMPO/EMPO/verl-empo/verl/utils/reward_score/empo/compute_metrics.py`
- Expected experiment effect: balanced selection and future variants can consume stable vote metadata directly.
- Open questions / follow-up runs: verify the artifact stays correct when vote mode is `edis_borda`.

## 2026-03-18 4
- Decision: implement a new additive `balanced_selection` reward manager using largest-cluster vs rest pools.
- Reason: the tutor wants balanced sampling between the largest cluster and everything outside it, with configurable proportion and skip-on-underfill behavior.
- Affected code path: `coding_edis_empo/src/edis_reward_manager.py`, `coding_edis_empo/src/main.py`, `coding_edis_empo/src/config/edis_empo_trainer.yaml`
- Expected experiment effect: new A/B/C/D-style experiments can be run without changing the existing baseline managers.
- Open questions / follow-up runs: compare 50/50, 70/30, and other ratios; inspect how often prompt groups are skipped.

## 2026-03-18 5
- Decision: underfilled balanced groups are skipped entirely, not backfilled.
- Reason: the tutor explicitly said not to borrow from the second-largest cluster or the opposite side when quota is not met.
- Affected code path: `coding_edis_empo/src/edis_reward_manager.py`
- Expected experiment effect: better semantic consistency, but fewer effective training prompts when clustering is too diffuse.
- Open questions / follow-up runs: measure skip rate and decide whether some ratios are too aggressive for Qwen2.5-Math-1.5B.

## 2026-03-18 6
- Decision: map `算法流程` presets onto generic configuration axes instead of separate bespoke managers.
- Reason: the selection policy and reward policy vary independently, so presets are clearer as configuration overlays.
- Affected code path: `coding_edis_empo/src/edis_reward_manager.py`, `coding_edis_empo/scripts/run_balanced_selection.sh`
- Expected experiment effect: A/B/C/D experiments are easier to launch and extend.
- Open questions / follow-up runs: confirm whether preset D should always pair with recomputed EMPO reward or remain manually configurable.

## 2026-03-18 7
- Decision: keep the original M-sample vote tag and the selected-set training vote separate in balanced-selection metadata.
- Reason: later selection logic depends on the original largest-vs-rest partition, while recomputed EMPO reward on the selected K can produce a different winner tag.
- Affected code path: `coding_edis_empo/src/edis_reward_manager.py`
- Expected experiment effect: downstream analysis can safely distinguish `pre_selection_*` artifacts from the current training reward artifact.
- Open questions / follow-up runs: decide which artifact should drive future weighting or curriculum logic when both are logged.

## 2026-03-19 1
- Decision: remove the old trainer-side EDIS advantage weighting path entirely.
- Reason: the tutor-defined "weighting" is now vote-time weighting during winner determination, so the old trainer patch is both conceptually wrong for current experiments and confusing in the codebase.
- Affected code path: `coding_edis_empo/src/main.py`, `coding_edis_empo/src/edis_weighting_trainer.py`
- Expected experiment effect: eliminate an invalid experiment family and force all new weighting experiments onto the vote-time path.
- Open questions / follow-up runs: none for implementation; historical runs should be treated as legacy-only references.

## 2026-03-19 2
- Decision: collapse the active experiment surface to three script families: baseline, oversample, and filtering.
- Reason: selection method, reward method, and vote weighting are now orthogonal controls, so separate weighting scripts are redundant and misleading.
- Affected code path: `coding_edis_empo/scripts/run_empo_baseline.sh`, `coding_edis_empo/scripts/run_empo_oversampled_32.sh`, `coding_edis_empo/scripts/run_edis_filtering_32.sh`
- Expected experiment effect: easier experiment management and fewer chances to launch the wrong method because of stale script names.
- Open questions / follow-up runs: consider renaming the surviving filtering/oversample scripts later if the `*_32` suffix becomes confusing.

## 2026-03-19 3
- Decision: initially represent note step 3C as `reward_method=direct_vote_tag` plus `vote_weighting=edis_borda`, not as a standalone reward path.
- Reason: weighted voting affects the original winner/tag artifact, while reward assignment remains either recomputed EMPO on selected K or direct binary reward from the original vote tag.
- Affected code path: `coding_edis_empo/src/edis_reward_manager.py`, `coding_empo/project_EMPO/EMPO/verl-empo/verl/utils/reward_score/empo/compute_metrics.py`, `coding_edis_empo/scripts/run_edis_filtering_32.sh`
- Expected experiment effect: cleaner experiment semantics and direct support for independently choosing selection, reward, and vote weighting.
- Open questions / follow-up runs: superseded by 2026-03-19 7 after clarifying that step 3C should let EDIS affect reward magnitude directly.

## 2026-03-19 4
- Decision: keep the balanced largest-vs-rest split based on raw cluster size even when `vote_weighting=edis_borda`.
- Reason: the tutor-defined Step 2 uses the largest cluster by size for selection, while vote-time weighting is only used to decide the winner/tag artifact for later reward logic.
- Affected code path: `coding_edis_empo/src/edis_reward_manager.py`, `coding_empo/project_EMPO/EMPO/verl-empo/verl/utils/reward_score/empo/compute_metrics.py`
- Expected experiment effect: selection diversity stays tied to cluster structure, while Step 3C can still flip the winner/tag based on EDIS-stability weighting.
- Open questions / follow-up runs: inspect how often `largest_cluster` and `weighted winner` disagree under `edis_borda`.

## 2026-03-19 5
- Decision: remove the hardcoded default resume checkpoint from the merged filtering script.
- Reason: remote smoke testing showed that fresh runs fail immediately because the script tries to resume from a stale checkpoint path unless `RESUME_PATH` is explicitly cleared.
- Affected code path: `coding_edis_empo/scripts/run_edis_filtering_32.sh`
- Expected experiment effect: new smoke runs and fresh experiments start cleanly by default; resume remains opt-in.
- Open questions / follow-up runs: none; rerun the remote smoke job after syncing the script fix.

## 2026-03-19 6
- Decision: treat the current remote smoke failures on spare GPUs as hardware-capacity issues, not refactor regressions.
- Reason: on `direct-server`, both the balanced filtering path and the oversample wrapper completed rollout and reward-manager execution, emitted the expected selection logs, and then failed only at the actor optimizer step with CUDA OOM on 24 GB spare GPUs.
- Affected code path: `coding_edis_empo/scripts/run_edis_filtering_32.sh`, `coding_edis_empo/scripts/run_empo_oversampled_32.sh`, `coding_edis_empo/src/edis_reward_manager.py`
- Expected experiment effect: the merged script/model surface is validated for startup, rollout, voting, selection, masking, and reward assignment; full training still needs a 48 GB-class GPU or the usual multi-GPU configuration.
- Open questions / follow-up runs: run a one-step smoke on an actually roomy GPU set, or skip directly to normal 4-GPU experiments for final validation.

## 2026-03-19 7
- Decision: implement note step 3C as a real third reward method, `edis_weighted_vote_tag`, instead of treating it as only `direct_vote_tag + edis_borda`.
- Reason: if reward uses raw `0/1` tags, EDIS weighting cannot affect the negative side at all; a signed bounded EDIS-scaled tag reward preserves positive/negative supervision on both sides.
- Affected code path: `coding_edis_empo/src/edis_reward_manager.py`, `coding_edis_empo/scripts/run_edis_filtering_32.sh`, `coding_edis_empo/src/config/edis_empo_trainer.yaml`
- Expected experiment effect: step 3C can now test whether EDIS should modulate reward magnitude directly, independently from whether weighted voting is enabled.
- Open questions / follow-up runs: compare `direct_vote_tag` versus `edis_weighted_vote_tag` under both `vote_weighting=off` and `vote_weighting=edis_borda`; inspect whether signed rewards are stable enough or need tighter clipping.

## 2026-03-19 8
- Decision: accept `vote_weighting=on` as a script-level alias for `edis_borda`, and `majority` as an alias for `off`.
- Reason: the experiment intent is obvious, and the script should not fail on a natural `on/off` phrasing when the underlying vote mode is unambiguous.
- Affected code path: `coding_edis_empo/scripts/run_edis_filtering_32.sh`
- Expected experiment effect: fewer avoidable launch failures from config wording; no change to experiment semantics.
- Open questions / follow-up runs: none; the alias only changes script parsing, not reward logic.

## 2026-03-19 9
- Decision: make the oversample wrapper re-exec under bash when launched via `sh`.
- Reason: the wrapper enables `set -o pipefail`, which fails immediately under `dash`; users naturally invoke scripts with `sh` on the server, so the wrapper should match the main filtering script's behavior.
- Affected code path: `coding_edis_empo/scripts/run_empo_oversampled_32.sh`
- Expected experiment effect: `sh run_empo_oversampled_32.sh` and `bash run_empo_oversampled_32.sh` both enter the same bash code path.
- Open questions / follow-up runs: none; the change is a shell bootstrap fix only.

## 2026-03-19 10
- Decision: include balanced preset metadata in the filtering script's default experiment name, and restore the filtering-script defaults from smoke-test values to the research-safe baseline.
- Reason: balanced presets `A/B/C/D` with the same reward mode were producing indistinguishable experiment names and log filenames, which makes B/C/D runs easy to confuse; the script had also been left on temporary `K=4, M=8, balanced + weighted` defaults from testing.
- Affected code path: `coding_edis_empo/scripts/run_edis_filtering_32.sh`
- Expected experiment effect: balanced runs are now named with `balanced_selection_method` and `majority_share`, and accidental launches will default back to the intended baseline-style `K=8, M=32, edis_topk + recompute_empo + off` configuration.
- Open questions / follow-up runs: rerun balanced presets B/C/D with the new naming so each log is unambiguous.

## 2026-03-19 11
- Decision: fail fast if `balanced_selection_method` is set to `B/C/D` while `selection_family` is not `balanced_largest_vs_rest`.
- Reason: recent test runs used `selection_family=edis_topk` together with `balanced_selection_method=B/C/D`, which silently ignored the preset and produced misleading “B/C/D” experiment logs.
- Affected code path: `coding_edis_empo/src/edis_reward_manager.py`, `coding_edis_empo/scripts/run_edis_filtering_32.sh`
- Expected experiment effect: invalid preset combinations stop immediately instead of running the wrong method.
- Open questions / follow-up runs: rerun the intended balanced B/C/D experiments with `selection_family=balanced_largest_vs_rest`.

## 2026-03-19 12
- Decision: require a staged preflight validation suite (config gate -> selector invariants -> 1-step server dry-run -> log audit) before launching long training.
- Reason: long runs are expensive, and prior issues came from silent config mismatches rather than immediate crashes.
- Affected code path: operational workflow around `coding_edis_empo/scripts/run_edis_filtering_32.sh` and `coding_edis_empo/src/edis_reward_manager.py`.
- Expected experiment effect: catches method-mismatch and reward-selection wiring errors before expensive training begins.
- Open questions / follow-up runs: add a small automated log-check script if manual audit becomes too slow.

## 2026-03-19 13
- Decision: validate all selection/reward/vote logic with synthetic server-side checks before long runs, and explicitly document venv constraints.
- Reason: logic checks passed in a full RL environment (`/home/pterence6699/coding_empo/venv`), while `/home/pterence6699/coding_edis/.venv` required dependency repair (`setuptools<81`) and still lacks `ray`, so manager-level imports fail there.
- Affected code path: `coding_edis_empo/src/edis_reward_manager.py`, `coding_empo/project_EMPO/EMPO/verl-empo/verl/utils/reward_score/empo/compute_metrics.py`, and runtime environments.
- Expected experiment effect: method semantics are now verified independently of long training, and environment mismatch risk is explicit before launch.
- Open questions / follow-up runs: decide whether to install `ray` in `/home/pterence6699/coding_edis/.venv` or standardize all RL runs on `/home/pterence6699/coding_empo/venv`.

## 2026-03-19 14
- Decision: complete server smoke verification on spare GPU with one-step runs for oversample and filtering families, including balanced B/C/D.
- Reason: we needed proof that the merged script surface and new selection/reward/vote combinations actually execute end-to-end, not just pass local logic assertions.
- Affected code path: `coding_edis_empo/scripts/run_empo_oversampled_32.sh`, `coding_edis_empo/scripts/run_edis_filtering_32.sh`, `coding_edis_empo/src/edis_reward_manager.py`, `coding_empo/project_EMPO/EMPO/verl-empo/verl/utils/reward_score/empo/compute_metrics.py`.
- Expected experiment effect: launch-time confidence is higher because all target combinations now start, run reward logic, mask/select correctly, and finish step 1 without config/runtime crashes.
- Open questions / follow-up runs: install `latex2sympy2` in the RL venv to remove repeated eval-math fallback warnings; run a longer 50-100 step sanity job for the chosen method before full training.

## 2026-03-19 15
- Decision: use a staged experiment order that first isolates selection effects under `recompute_empo`, then adds reward-method ablations, and only then enables vote-time weighting (`edis_borda`) for tag-based rewards.
- Reason: this minimizes confounding factors and avoids spending long-run budget on low-value full-factorial combinations too early.
- Affected code path: experiment orchestration using `coding_edis_empo/scripts/run_edis_filtering_32.sh` (and optional `run_empo_oversampled_32.sh` baseline wrapper).
- Expected experiment effect: faster identification of the best selection method (A/B/C/D vs classic) before testing whether tag-style rewards and weighted voting actually help.
- Open questions / follow-up runs: once top 1-2 combinations are identified, repeat with multiple seeds and extend steps to check late-stage stability.

## 2026-03-19 16
- Decision: standardize default output-directory naming with method metadata and enable W&B logging by default (toggleable).
- Reason: generic output paths were hard to trace back to exact configs, and tutor explicitly required W&B usage.
- Affected code path: `coding_edis_empo/scripts/run_edis_filtering_32.sh`, `coding_edis_empo/scripts/run_empo_baseline.sh`, `coding_edis_empo/scripts/run_empo_oversampled_32.sh`.
- Expected experiment effect: each run now defaults to a self-descriptive `outputs/<run_signature>` directory and logs to W&B via `trainer.logger=['console','tensorboard','wandb']` unless `USE_WANDB=False`.
- Open questions / follow-up runs: confirm `wandb login` / `WANDB_API_KEY` is ready on server and decide whether to shorten run-signature tokens after first batch of real runs.

## 2026-03-19 17
- Decision: keep W&B in normal direct-network mode on the server (no local proxy wrapper) after successful `wandb login`.
- Reason: `wandb login` succeeded and wrote credentials to `/home/pterence6699/.netrc`; API root `curl` returning `404` is expected connectivity behavior, while proxy mode introduced SSL EOF errors.
- Affected code path: runtime environment for `coding_edis_empo` experiment scripts (`USE_WANDB=True`, `WANDB_MODE=online` defaults).
- Expected experiment effect: W&B uploads should remain stable during training runs; fewer network-side failures compared with proxy-on mode.
- Open questions / follow-up runs: run a tiny W&B init/log/finish smoke command before starting long training to confirm project/entity routing.

## 2026-03-19 18
- Decision: fix `verl-empo` tracking initialization to respect `WANDB_MODE` and remove local-scope `os` imports inside `Tracking.__init__`.
- Reason: local `import os` inside other backend branches makes `os` a function-local variable and can crash the wandb branch (`UnboundLocalError`); hardcoded `mode="offline"` also blocks intended online syncing.
- Affected code path: `coding_empo/project_EMPO/EMPO/verl-empo/verl/utils/tracking.py`.
- Expected experiment effect: `USE_WANDB=True` runs should initialize wandb reliably in the configured mode (`online` by default from scripts) and upload step metrics as training progresses.
- Open questions / follow-up runs: run a one-step server smoke with `trainer.logger` including `wandb` and verify the run URL appears in console output.

## 2026-03-19 19
- Decision: validate W&B end-to-end with a one-step server smoke run after syncing tracker fix commit `ea68765` to both local and remote.
- Reason: we needed proof that scripts, tracker init, and metric logging all work together on the real training path, not only in standalone `wandb.init()` tests.
- Affected code path: `coding_empo/project_EMPO/EMPO/verl-empo/verl/utils/tracking.py`, `coding_edis_empo/scripts/run_edis_filtering_32.sh`.
- Expected experiment effect: training runs now print W&B project/run URLs and stream step metrics (including `val-core/amc/acc/*`, loss/entropy/timing/perf keys) to dashboards for graphing.
- Open questions / follow-up runs: optional cleanup of repetitive eval-math fallback warnings by installing `latex2sympy2` in `/home/pterence6699/coding_empo/venv`.

## 2026-03-19 20
- Decision: standardize W&B chart setup guidance for experiment review (`X=Step`, `Y=metric`, avoid config constants as axes).
- Reason: using static config fields (for example `trainer.total_training_steps`) as X-axis produces non-informative charts and confuses metric trend analysis.
- Affected code path: experiment monitoring workflow on W&B dashboards (no code behavior change).
- Expected experiment effect: run curves become interpretable and comparable across methods; easier diagnosis of collapse and stability windows.
- Open questions / follow-up runs: build a saved W&B workspace template after the next real batch of runs.

## 2026-03-19 21
- Decision: use W&B `Add panel -> Charts -> Line plot` flow (or metric quick-add) instead of editing a scatter/custom panel when plotting training curves.
- Reason: the scatter/custom editor can lock users into X/Y schema selection and make it seem like "line graph is unavailable", while line panels are created from chart type selection at panel creation time.
- Affected code path: W&B workspace operation guide (no training code change).
- Expected experiment effect: faster creation of standard experiment-tracking charts for `acc/loss/entropy` without UI confusion.
- Open questions / follow-up runs: create a reusable workspace template with fixed panels once first full run batch completes.

## 2026-03-27 1
- Decision: split vote-time weighting into two explicit phases: `pre_selection_vote_mode` for the original M-sample vote artifact and `selected_vote_mode` for the selected-K recompute vote.
- Reason: the tutor correctly pointed out that one overloaded `vote_mode` flag was ambiguous. In the previous implementation, Borda weighting affected the first round only, while `recompute_empo` silently forced plain majority on the second round. The new surface makes that behavior explicit and controllable.
- Affected code path: `coding_edis_empo/src/edis_reward_manager.py`, `coding_edis_empo/src/main.py`, `coding_edis_empo/src/config/edis_empo_trainer.yaml`, `coding_edis_empo/scripts/run_edis_filtering_32.sh`, `coding_edis_empo/scripts/run_empo_oversampled_32.sh`.
- Expected experiment effect: clearer experiment semantics. `edis_topk + recompute_empo + pre_selection_vote_mode=edis_borda + selected_vote_mode=majority` is now explicitly recognized as "weighted tags only, unchanged selection, unweighted recompute". If selected-set Borda is ever tested, it must be turned on deliberately.
- Open questions / follow-up runs: decide whether to keep `selected_vote_mode=edis_borda` available for ablation or to hard-disable it later if the tutor wants only first-round weighting experiments.

## 2026-03-27 2
- Decision: add exact balanced-count controls `largest_group_count` and `rest_group_count`, while keeping `majority_share` as the fallback.
- Reason: share-based rounding is too coarse when the tutor wants explicit splits such as `largest=3, rest=5`. Exact counts better match the intended algorithm and avoid ambiguity from rounding.
- Affected code path: `coding_edis_empo/src/edis_reward_manager.py`, `coding_edis_empo/src/main.py`, `coding_edis_empo/src/config/edis_empo_trainer.yaml`, `coding_edis_empo/scripts/run_edis_filtering_32.sh`, `coding_edis_empo/scripts/run_empo_oversampled_32.sh`.
- Expected experiment effect: balanced-selection experiments can now directly enforce tutor-specified allocations; run naming and console config also expose whether selection used `share...` or `largest.../rest...`.
- Open questions / follow-up runs: compare `4/4`, `3/5`, and `2/6` on balanced B/C/D, and track skip-prompt rate because aggressive exact splits may raise underfilled skips.

## 2026-03-27 3
- Decision: remove `majority_share` entirely from the active experiment surface and use only `largest_group_count` / `rest_group_count` for balanced largest-vs-rest selection.
- Reason: once the tutor requested exact control such as `largest=3, rest=5`, keeping `majority_share` only preserved an unnecessary second way to express the same idea and invited ambiguity.
- Affected code path: `coding_edis_empo/src/edis_reward_manager.py`, `coding_edis_empo/src/main.py`, `coding_edis_empo/src/config/edis_empo_trainer.yaml`, `coding_edis_empo/scripts/run_edis_filtering_32.sh`, `coding_edis_empo/scripts/run_empo_oversampled_32.sh`.
- Expected experiment effect: balanced runs are now parameterized by explicit counts only; default behavior becomes `largest=K/2, rest=K/2` when no counts are provided.
- Open questions / follow-up runs: none for implementation; only the Borda semantics remain to be fixed or clarified.

## 2026-03-27 4
- Decision: adopt tutor-required weighted-largest semantics for first-round voting and disable second-round Borda entirely.
- Reason: previous behavior allowed weighted winner tags to diverge from raw largest-vs-rest pools and left second-round weighting as an optional branch, which contradicted the tutor's algorithm constraints.
- Affected code path: `coding_empo/project_EMPO/EMPO/verl-empo/verl/utils/reward_score/empo/compute_metrics.py`, `coding_edis_empo/src/edis_reward_manager.py`, `coding_edis_empo/src/main.py`, `coding_edis_empo/src/config/edis_empo_trainer.yaml`, `coding_edis_empo/scripts/run_edis_filtering_32.sh`, `coding_edis_empo/scripts/run_empo_oversampled_32.sh`.
- Expected experiment effect: when first-round Borda is on, "largest cluster", winner, tag assignment, and balanced largest/rest pools are all derived from weighted cluster mass; selected-K reward recomputation always uses majority vote.
- Open questions / follow-up runs: compare first-round `majority` vs `edis_borda` under identical balanced settings now that second-round behavior is fixed.

## 2026-03-30 1
- Decision: prioritize the next experiment batch around balanced `D` as the main branch, with reruns of weighted variants only after the weighted-largest semantics fix.
- Reason: among the post-filtering redesign runs, `run26_balanced_D_directVoteTag` is currently the strongest result (`best=57.19% @ step 1430`, `last=54.69% @ step 1500`), while `run25_balanced_D_directVoteTag_weighting` was weaker (`best=55.62% @ step 500`, `last=53.44% @ step 1200`) and was executed before the weighted-largest/Borda fix. Balanced `B` and `C` showed severe instability or collapse, so they are lower-priority branches.
- Affected code path: experiment planning for `coding_edis_empo/scripts/run_edis_filtering_32.sh` and result interpretation for weighted-vote runs.
- Expected experiment effect: spend compute on the branch with the highest current empirical upside, while rerunning weighted variants only under the corrected semantics so the teacher-facing conclusion is defensible.
- Open questions / follow-up runs:
  - rerun `D + direct_vote_tag + edis_borda` under the corrected weighted-largest semantics
  - rerun classic `edis_topk + recompute_empo + edis_borda` under the corrected semantics
  - test exact count splits around `D` (`4/4`, `3/5`, `5/3`) to measure whether the current success comes from balance or from the random/random diversity pattern
  - decide whether `A + direct_vote_tag` deserves one confirmation rerun or should be deprioritized behind the `D` family

## 2026-03-30 2
- Decision: treat first-round Borda as a meaningful ablation only when the pre-selection vote artifact influences training through balanced selection pools or direct tag-based rewards.
- Reason: after disabling second-round Borda, combinations such as `edis_topk + recompute_empo + edis_borda` barely change the actual training signal, because selection is still pure EDIS top-k and rewards are recomputed on selected-K with plain majority. In contrast, balanced largest-vs-rest selection and direct/weighted vote-tag rewards directly consume the first-round vote artifact.
- Affected code path: experiment interpretation for `selection_family`, `reward_method`, and `pre_selection_vote_mode`.
- Expected experiment effect: avoid spending GPU time on ablations that are logically close to no-ops; concentrate weighting tests on settings where weighting can actually change groups, tags, or reward values.
- Open questions / follow-up runs:
  - compare `balanced D + direct_vote_tag` with `pre_selection_vote_mode=majority` vs `edis_borda`
  - compare `balanced D + edis_weighted_vote_tag` with `pre_selection_vote_mode=majority` vs `edis_borda`
  - decide whether any `edis_topk + recompute_empo + edis_borda` run is still worth keeping purely as a sanity check

## 2026-03-30 3
- Decision: clarify that `reward_method` and first-round vote weighting are orthogonal controls, and that `edis_weighted_vote_tag` changes reward magnitude rather than only changing the vote result.
- Reason: the naming is easy to misread. In the current implementation, `direct_vote_tag` uses the first-round vote tag exactly as produced by `pre_selection_vote_mode` (`majority` or `edis_borda`), while `edis_weighted_vote_tag` uses that same tag for sign/support and then applies an inverse-EDIS scaling to the reward value.
- Affected code path: `coding_edis_empo/src/edis_reward_manager.py`, `coding_empo/project_EMPO/EMPO/verl-empo/verl/utils/reward_score/empo/compute_metrics.py`.
- Expected experiment effect: cleaner interpretation of runs; avoids incorrectly describing `edis_weighted_vote_tag` as "just Borda reward" or describing `direct_vote_tag` as always majority-based.
- Open questions / follow-up runs:
  - consider renaming `edis_weighted_vote_tag` later to something more explicit such as `scaled_vote_tag_reward`
  - decide whether incorrect-tag rewards should remain negative or be changed to zero if the tutor wants a strict binary reward family

## 2026-03-30 4
- Decision: simplify the script-facing vote-weighting control to a single `BORDA_WEIGHTING=on|off` switch and remove the old fallback chain involving `VOTE_WEIGHTING` and `PRE_SELECTION_VOTE_WEIGHTING`.
- Reason: the previous two-line fallback was needlessly confusing in the experiment scripts. The user edits script config blocks directly, so a single visible switch is the correct interface.
- Affected code path: `coding_edis_empo/scripts/run_edis_filtering_32.sh`, `coding_edis_empo/scripts/run_empo_oversampled_32.sh`.
- Expected experiment effect: less configuration ambiguity and fewer accidental launches with the wrong first-round vote mode.
- Open questions / follow-up runs:
  - none for implementation
  - if old notes or commands still mention `VOTE_WEIGHTING`, update them to `BORDA_WEIGHTING`

## 2026-04-02 1
- Decision: prioritize fresh reruns of `balanced D + direct_vote_tag` with and without first-round Borda before rerunning oversampled EMPO.
- Reason: the balanced `D` branch is the current best-performing direction, while the weighted `D` result was collected before the weighted-largest semantics fix and therefore needs a clean rerun. Oversampled EMPO should still be rerun as a clean post-fix baseline because the historical oversampled result predates the "recompute rewards after selection" fix, but it is lower priority because recent Borda-related fixes do not materially affect the oversampled path when it uses `random_topk + recompute_empo + BORDA_WEIGHTING=off`.
- Affected code path: experiment planning around `coding_edis_empo/scripts/run_edis_filtering_32.sh` and `coding_edis_empo/scripts/run_empo_oversampled_32.sh`.
- Expected experiment effect: obtain an up-to-date apples-to-apples comparison for the strongest balanced branch first, then refresh the oversampled baseline under the corrected reward semantics.
- Open questions / follow-up runs:
  - rerun `D + direct_vote_tag + borda_weighting=off + 4/4`
  - rerun `D + direct_vote_tag + borda_weighting=on + 4/4`
  - rerun `random_topk + recompute_empo + borda_weighting=off`
  - if D remains strongest, move next to `3/5` and `5/3` count ablations

## 2026-04-02 2
- Decision: update the filtering script defaults to match the immediate rerun plan for balanced `D`: `BORDA_WEIGHTING=off`, `GROUP_LARGEST_COUNT=4`, and `GROUP_REST_COUNT=4`.
- Reason: the script still defaulted to `on` and `5/3`, which would make the generated output directory names and default launches diverge from the intended `D + direct_vote_tag` reruns.
- Affected code path: `coding_edis_empo/scripts/run_edis_filtering_32.sh`.
- Expected experiment effect: default run signatures and output directories now correctly encode the intended `4/4` and non-Borda baseline setup; the Borda ablation only requires flipping one switch to `on`.
- Open questions / follow-up runs:
  - none for implementation
  - verify the first two reruns produce output dirs ending in `largest4-rest4_rw-directVoteTag_preVote-majority` and `largest4-rest4_rw-directVoteTag_preVote-edisBorda`

## 2026-04-04 1
- Decision: document that checkpoint resume and W&B run resume are currently independent, and that the existing remote code starts a fresh W&B run on every process launch.
- Reason: on the remote server, `tracking.py` calls `wandb.init(project=..., name=..., config=..., mode=..., dir=...)` without supplying a persistent `id` or `resume=...`, while the training scripts only pass `trainer.resume_from_path` / `trainer.resume_mode=resume_path` for checkpoint recovery. As a result, continuing from a step-1500 checkpoint resumes model state but does not reattach to the previous W&B run.
- Affected code path: `coding_empo/project_EMPO/EMPO/verl-empo/verl/utils/tracking.py`, `coding_edis_empo/scripts/run_edis_filtering_32.sh`, `coding_edis_empo/scripts/run_empo_baseline.sh`.
- Expected experiment effect: resumed training appears as a new W&B run even when using the same script, same experiment name, and a valid checkpoint path.
- Open questions / follow-up runs:
  - persist a W&B run id alongside the output/checkpoint directory
  - on resume, pass that id into `wandb.init(..., id=..., resume=\"must\")` so W&B appends instead of creating a new run

## 2026-04-04 2
- Decision: implement automatic W&B run-id persistence and reuse based on trainer checkpoint/output paths.
- Reason: resumed training should continue in the same W&B run, not create a fresh dashboard entry. The fix stores the run id in `wandb_run_id.txt` beside the checkpoint directory and reuses it when `trainer.resume_mode` is `resume_path` or `auto`.
- Affected code path: `coding_empo/project_EMPO/EMPO/verl-empo/verl/utils/tracking.py`.
- Expected experiment effect: launching a resumed run from an existing checkpoint directory should append metrics into the original W&B run instead of creating a new run, provided the matching `wandb_run_id.txt` is still present.
- Open questions / follow-up runs:
  - sync the tracker patch to the remote server before restarting training
  - confirm on the next resumed run that W&B shows the same run id and continuous step history

## 2026-04-04 3
- Decision: recover the original W&B run IDs for the two Apr-2 balanced `D + direct_vote_tag` experiments and seed `wandb_run_id.txt` manually in their remote output directories.
- Reason: those runs were started before automatic run-id persistence existed, so the output directories had no `wandb_run_id.txt`. Mapping the experiment names in the remote W&B cache showed:
  - `...preVote-majority...` -> `w87ezk5j`
  - `...preVote-edisBorda...` -> `nxolzmx8`
- Affected code path: remote output directories under `coding_edis_empo/outputs/...` and the resumed W&B attach behavior in `verl/utils/tracking.py`.
- Expected experiment effect: the next resumed launch from those exact output/checkpoint directories should reattach to the original Apr-2 W&B runs instead of creating another fresh run.
- Open questions / follow-up runs:
  - verify the next resumed launch prints the expected reused run id
  - decide whether the accidental Apr-4 duplicate runs should simply be ignored or explicitly marked as superseded in W&B

## 2026-04-02 1
- Decision: make balanced largest-vs-rest selection consume the first-round vote artifact's `in_largest_cluster` membership instead of recomputing "largest" from raw cluster size.
- Reason: when `pre_selection_vote_mode=edis_borda`, the tutor wants weighted-largest semantics to define both clustering outcomes and the later largest-vs-rest split. Re-deriving the largest cluster from raw counts breaks that requirement.
- Affected code path: `coding_edis_empo/src/edis_reward_manager.py`.
- Expected experiment effect: first-round Borda can now genuinely change balanced selection pools, not only the logged tag metadata.
- Open questions / follow-up runs:
  - rerun the weighted `D` branch under the fixed semantics and compare against the earlier `run25`

## 2026-04-02 2
- Decision: reinterpret `reward_method=edis_weighted_vote_tag` as binary vote-tag reward plus post-GRPO advantage scaling, instead of directly scaling the reward tensor.
- Reason: the tutor explicitly wants EDIS weighting to act on the policy update signal rather than on EMPO reward construction. This preserves EMPO reward semantics and avoids conflating reward shaping with advantage reweighting.
- Affected code path: `coding_edis_empo/src/edis_reward_manager.py`, `coding_empo/project_EMPO/EMPO/verl-empo/recipe/empo/src/dapo_ray_trainer.py`, `coding_edis_empo/scripts/run_edis_filtering_32.sh`, `coding_edis_empo/src/main.py`.
- Expected experiment effect: `direct_vote_tag` and `edis_weighted_vote_tag` now share the same binary reward source, while the latter only changes actor advantages through bounded inverse-EDIS weights.
- Open questions / follow-up runs:
  - inspect whether the bounded weight range `[0.5, 1.5]` is too conservative or too aggressive
  - decide whether to log both weighted and unweighted advantage summaries in W&B

## 2026-04-02 3
- Decision: baseline EMPO remains "single EMPO over the full in-group responses", while oversampled EMPO remains "select K, recompute EMPO on the selected K, mask the rest".
- Reason: baseline has no selection stage, so there is nothing to recompute. For oversampled filtering-style runs, recomputing on the selected K is the correct design and does not break GRPO grouping because discarded samples are masked and assigned a `_discarded` uid suffix before advantage computation.
- Affected code path: `coding_empo/project_EMPO/EMPO/verl-empo/verl/workers/reward_manager/empo.py`, `coding_edis_empo/src/edis_reward_manager.py`, `coding_empo/project_EMPO/EMPO/verl-empo/verl/trainer/ppo/core_algos.py`.
- Expected experiment effect: GRPO group statistics are computed on the effective training set only; discarded samples contribute zero masked advantages and do not leak into the selected group's in-group normalization.
- Open questions / follow-up runs:
  - remote smoke-check one oversample run to verify the logged reward/selection metadata matches this interpretation end-to-end

## 2026-04-04 4
- Decision: make the oversampled wrapper clear `GROUP_LARGEST_COUNT` and `GROUP_REST_COUNT` explicitly instead of inheriting them from the caller environment.
- Reason: when the script is launched from a tmux shell that previously exported balanced-selection variables, `random_topk` runs fail validation inside `run_edis_filtering_32.sh` because those balanced-only counts leak into the oversample baseline.
- Affected code path: `coding_edis_empo/scripts/run_empo_oversampled_32.sh`, `coding_edis_empo/scripts/run_edis_filtering_32.sh`.
- Expected experiment effect: oversampled EMPO launches are isolated from stale balanced-selection environment variables and work reliably from reused tmux sessions.
- Open questions / follow-up runs:
  - sync the wrapper fix to the remote server and verify that a run still succeeds even if `GROUP_LARGEST_COUNT=4` and `GROUP_REST_COUNT=4` are exported in the shell

## 2026-04-09 1
- Decision: pivot the project away from EDIS entirely and redefine the active experiment surface around EMPO-only oversampling, balanced largest-vs-rest selection, and optional entropy-based advantage weighting.
- Reason: the EDIS paper may not be publishable, and recent experiments already show that first-round Borda weighting is not helping. The strongest branch so far (`D + direct_tag`) mostly relies on EMPO vote structure plus diversified selection rather than on EDIS.
- Affected code path: future refactor of `coding_edis_empo/src/edis_reward_manager.py`, `coding_edis_empo/src/main.py`, `coding_edis_empo/scripts/run_edis_filtering_32.sh`, `coding_edis_empo/scripts/run_empo_oversampled_32.sh`, and the experiment naming/documentation surface.
- Expected experiment effect: cleaner research story and experiment matrix:
  - A) EMPO baseline
  - B) oversampled EMPO + random selection + first-round reward reuse
  - C) oversampled EMPO + balanced D selection + first-round reward reuse
  - D) same as C, plus entropy-based advantage weighting
- Open questions / follow-up runs:
  - confirm whether "reuse first-round reward" means original EMPO entropy reward or the current binary first-round vote tag
  - implement entropy-based advantage weights from token-level entropy statistics and compare them against plain C
  - only delete EDIS-specific code after the new EMPO-only path is working and reproducible

## 2026-04-09 2
- Decision: remove all active second-round reward recomputation from the configurable oversample path and reuse only first-round reward artifacts.
- Reason: the tutor clarified that oversampled experiments should select from the original M-sample EMPO pass and train directly on first-round reward signals rather than rescoring after selection.
- Affected code path: `coding_edis_empo/src/edis_reward_manager.py`, `coding_edis_empo/src/main.py`, `coding_edis_empo/src/config/edis_empo_trainer.yaml`, `coding_edis_empo/scripts/run_edis_filtering_32.sh`, `coding_edis_empo/scripts/run_empo_oversampled_32.sh`, `coding_empo/project_EMPO/EMPO/verl-empo/recipe/empo/src/dapo_ray_trainer.py`
- Expected experiment effect:
  - baseline A remains original EMPO entropy reward
  - oversampled B defaults to `random_topk + use_binary_reward`
  - balanced C defaults to `largest/rest random 4+4 + use_binary_reward`
  - D adds entropy-based post-GRPO advantage weighting without changing rewards
- Open questions / follow-up runs:
  - compare `use_binary_reward` vs `use_empo_reward` on balanced largest-vs-rest
  - validate entropy-weighted advantage against the plain binary-reward D branch

## 2026-04-13 1
- Decision: enforce selected-only training masks in the trainer and actor loss while keeping all generated responses available for entropy-threshold calculation.
- Reason: discarded responses previously received zero rewards, but the trainer recomputed `response_mask` from `attention_mask` and actor loss could still include discarded zero-advantage tokens in the denominator. That would dilute selected-sample updates and violate the intended "train with selected K only" behavior.
- Affected code path: `coding_empo/project_EMPO/EMPO/verl-empo/recipe/empo/src/dapo_ray_trainer.py`, `coding_empo/project_EMPO/EMPO/verl-empo/verl/workers/actor/dp_actor.py`, `coding_empo/project_EMPO/EMPO/verl-empo/verl/workers/actor/megatron_actor.py`
- Expected experiment effect: B/C/D oversampled runs now optimize only the selected K responses, while D's entropy threshold still uses all valid tokens from the original M generated responses.
- Open questions / follow-up runs:
  - run a remote smoke test and confirm `selection/num_training_responses` / `selection/num_masked_responses` match the expected batch totals under M32/K8, i.e. 8 selected and 24 masked per prompt group
  - compare C vs D after this masking fix because previous entropy-weighted results are not directly comparable
  - monitor whether the fail-fast all-skipped-batch guard triggers; if it does, reduce largest/rest counts or inspect cluster concentration

## 2026-04-13 2
- Decision: hard-lock the oversampled baseline wrapper to `random_topk + use_binary_reward + adv_weight_mode=off` and keep balanced-count validation scoped only to balanced selection.
- Reason: the oversampled baseline should represent experiment B only. Allowing stale tmux environment variables to override `SELECTION_FAMILY` or `ADV_WEIGHT_MODE` could silently launch the wrong experiment. Random selection also should not inherit balanced-selection-only count constraints.
- Affected code path: `coding_edis_empo/scripts/run_empo_oversampled_32.sh`, `coding_edis_empo/src/edis_reward_manager.py`.
- Expected experiment effect: rerunning the oversampled baseline always uses random 8-from-32 selection with first-round binary rewards, unless the script file itself is edited.
- Open questions / follow-up runs:
  - after syncing to the remote server, run a lightweight launch/smoke check from a dirty tmux environment to confirm stale `SELECTION_FAMILY`, `ADV_WEIGHT_MODE`, and group-count variables cannot alter the baseline wrapper behavior

## 2026-04-14 1
- Decision: add one explicit launch script per current experiment: `run_A_empo_baseline.sh`, `run_B_empo_oversampled_random.sh`, `run_C_empo_oversampled_methodD.sh`, and `run_D_empo_oversampled_methodD_entropy_adv.sh`.
- Reason: the experiment-defining parameters should be visible in script files and not depend on command-line overrides or stale tmux exports. The old `run_empo_oversampled_32.sh` remains a compatibility alias for experiment B, while `run_edis_filtering_32.sh` remains the shared oversampled runner used by B/C/D.
- Affected code path: `coding_edis_empo/scripts/`, `coding_edis_empo/README.md`.
- Expected experiment effect: A/B/C/D can be launched directly with one script each, with output and W&B names tagged by the experiment letter and core parameter choices.
- Open questions / follow-up runs:
  - done: synced commit `46cbc96` to the remote server, ran `bash -n` on all launch scripts, and dry-ran A/B/C/D with `DRY_RUN=True` to confirm the resolved config without starting training

## 2026-04-14 2
- Decision: make the A/B/C/D launch scripts fully standalone instead of wrappers around older scripts.
- Reason: the user explicitly wanted each experiment script to be independent, and wrappers around `run_edis_filtering_32.sh` or `run_empo_baseline.sh` made the launch surface too easy to confuse with legacy scripts.
- Affected code path: `coding_edis_empo/scripts/run_A_empo_baseline.sh`, `coding_edis_empo/scripts/run_B_empo_oversampled_random.sh`, `coding_edis_empo/scripts/run_C_empo_oversampled_methodD.sh`, `coding_edis_empo/scripts/run_D_empo_oversampled_methodD_entropy_adv.sh`, `coding_edis_empo/README.md`.
- Expected experiment effect: launching A/B/C/D no longer depends on old script behavior. Each script directly contains its fixed experiment-defining config and generates a matching output/W&B run name.
- Open questions / follow-up runs:
  - done: pushed commit `03f62c0`, pulled it on `direct-server`, ran `bash -n`, verified A/B/C/D dry-run Hydra args, and confirmed the standalone scripts do not call legacy launchers

## 2026-04-14 3
- Decision: put A/B/C/D under one W&B project/group by default while keeping run names distinct.
- Reason: separate W&B projects made comparison harder; the desired comparison is among methods A/B/C/D in one dashboard.
- Affected code path: `coding_edis_empo/scripts/run_A_empo_baseline.sh`, `coding_edis_empo/scripts/run_B_empo_oversampled_random.sh`, `coding_edis_empo/scripts/run_C_empo_oversampled_methodD.sh`, `coding_edis_empo/scripts/run_D_empo_oversampled_methodD_entropy_adv.sh`, `coding_empo/project_EMPO/EMPO/verl-empo/verl/utils/tracking.py`, `coding_edis_empo/README.md`.
- Expected experiment effect: new A/B/C/D runs will appear in W&B project `EMPO-Only-ABCD`, group `EMPO-Only-ABCD`, with individual method-specific run names for side-by-side plots.
- Open questions / follow-up runs:
  - done: pushed `coding_edis_empo` commit `3baab29` and `verl-empo` commit `b218e97`, pulled both on `direct-server`, and verified dry-run Hydra args include `trainer.project_name=EMPO-Only-ABCD` and `trainer.wandb_group=EMPO-Only-ABCD`
  - existing runs that already initialized under old projects/groups will not move automatically; update them manually in W&B if needed, or restart from the updated scripts

## 2026-04-15 1
- Decision: make GSPO and GSPO-token losses ignore fully masked response rows.
- Reason: B/C/D oversampled runs keep discarded samples in the batch with all-zero `response_mask`; GSPO divided sequence log-prob sums by zero-length masks and produced `actor/pg_loss=nan` and `actor/grad_norm=nan` from step 1, causing actor updates to be skipped.
- Affected code path: `coding_empo/project_EMPO/EMPO/verl-empo/verl/trainer/ppo/core_algos.py`.
- Expected experiment effect: selected-only masking remains intact for B/C/D, discarded rows no longer poison sequence-level policy loss, and actor updates should report finite `actor/pg_loss` / `actor/grad_norm` during smoke tests and real runs.
- Open questions / follow-up runs:
  - run remote smoke tests for A/B/C/D on spare GPUs and inspect logs for finite actor loss/grad, expected selection counts, reward method, and entropy-advantage metrics for D

## 2026-04-15 2
- Decision: default missing advantage-weight config keys to `off` / standard entropy-weight defaults inside the DAPO trainer.
- Reason: the baseline A script intentionally keeps the original EMPO reward config and does not define `reward_model.reward_kwargs.adv_weight_mode`; the trainer should not require new oversampling-only keys for the unchanged baseline path.
- Affected code path: `coding_empo/project_EMPO/EMPO/verl-empo/recipe/empo/src/dapo_ray_trainer.py`.
- Expected experiment effect: A remains `reward_manager=empo`, `rollout.n=8`, and `reward_type=entropy`, while B/C/D can still opt into entropy-based advantage weighting through explicit config.
- Open questions / follow-up runs:
  - done: pushed Verl commit `ac68665` and pulled it on `direct-server`
  - done: A/B/C/D one-step smoke tests on GPUs 4/5 produced finite `actor/pg_loss` and `actor/grad_norm`
  - remaining environment warning: Ray reports `/tmp/ray` over 95% full during smoke tests; avoid running long jobs until scratch/checkpoint pressure is checked or cleaned

## 2026-04-16 1
- Decision: make EXP-D entropy advantage weighting reward-aware by using opposite stability directions for largest-cluster and rest-cluster samples.
- Reason: tutor clarified the intended logic: for largest-cluster / binary-positive samples, lower entropy should increase the advantage weight; for rest-cluster / binary-zero samples, lower entropy should decrease the advantage weight so stable wrong samples are penalized less while unstable wrong samples are penalized more.
- Affected code path: `coding_empo/project_EMPO/EMPO/verl-empo/recipe/empo/src/dapo_ray_trainer.py`.
- Expected experiment effect: EXP-D no longer applies the same inverse-entropy weighting to all selected responses. It now uses inverse high-entropy ratio for selected largest-cluster samples and direct high-entropy ratio for selected rest-cluster samples, clipped to the existing `[0.5, 1.5]` range.
- Open questions / follow-up runs:
  - active server EXP-D runs started before this patch still use the old non-reward-aware weighting and should be treated as old-D results
  - after the active jobs finish or are stopped, pull the new Verl commit on the server and rerun D from scratch

## 2026-04-16 2
- Decision: start public model downloads into `/home/pterence6699/coding_edis/pretrained` with a resumable background script, but do not merge the three project repositories yet.
- Reason: Qwen model downloads are needed for upcoming scale/model-size experiments, while repo merging should be planned separately to avoid disrupting active runs and Git synchronization.
- Affected code path: remote helper script `/tmp/download_pretrained_models.py`; target model directory `/home/pterence6699/coding_edis/pretrained`.
- Expected experiment effect: Qwen2.5-Math-1.5B remains available at the existing path, Qwen2.5-Math-7B and Qwen3-1.7B-Base are downloaded for future runs, and Llama-3.2-3B-Instruct is skipped until Hugging Face login and Meta access are available.
- Open questions / follow-up runs:
  - direct `huggingface.co` access stalled, so the downloader was restarted with `HF_ENDPOINT=https://hf-mirror.com`
  - monitor `/home/pterence6699/coding_edis/pretrained/download_models.log`
  - plan repo consolidation after current experiments finish; do not physically merge repos now

## 2026-04-16 3
- Decision: add dataset-aware A-J multiple-choice evaluation for MMLU-Pro and GPQA Diamond, with converted RLHF-format eval parquets that explicitly request final answers inside `\boxed{}`.
- Reason: the raw non-math parquets do not match Verl's `prompt` / `reward_model` / `extra_info` schema, and the existing EMPO scorer only handles math-style boxed or numeric answers. The fallback extractor must not scan the full response because that would mis-score reasoning text such as "option A is wrong".
- Affected code path: `coding_empo/project_EMPO/EMPO/verl-empo/verl/utils/reward_score/`, `coding_edis_empo/utils/convert_mcq_eval_parquets.py`, `coding_edis_empo/scripts/run_A_empo_baseline.sh`, `coding_edis_empo/scripts/run_B_empo_oversampled_random.sh`, `coding_edis_empo/scripts/run_C_empo_oversampled_methodD.sh`, `coding_edis_empo/scripts/run_D_empo_oversampled_methodD_entropy_adv.sh`.
- Expected experiment effect: offline validation can include `mmlu_pro` and `gpqa_diamond` and will report per-dataset metrics while preserving existing math evaluation behavior.
- Open questions / follow-up runs:
  - done: generated `/home/pterence6699/coding_empo/project_EMPO/EMPO/math_data/mmlu_pro_eval.parquet` and `gpqa_diamond_eval.parquet` on the server from the raw files
  - done: ran server scorer tests for boxed A/J, final-region fallback, middle-response trap cases, invalid K rejection, and math-route preservation
  - done: verified converted files load through `RLHFDataset`; non-AMC23 datasets are reserved for offline evaluation rather than training-time validation
  - prompt length note: with Qwen2.5-Math-1.5B tokenizer and `MAX_PROMPT_LENGTH=1024`, MMLU-Pro has 15/12032 prompts over 1024 tokens and GPQA Diamond has 1/198 prompt over 1024 tokens; use a larger `MAX_PROMPT_LENGTH` for full non-math eval if avoiding left-truncation matters

## 2026-04-16 4
- Decision: add validation-triggered top-3 AMC checkpoint retention, separate from periodic `save_freq` checkpoints.
- Reason: online training validation only uses AMC23, and the important checkpoints are the steps with the highest `val-core/amc/acc/mean@16`, not necessarily the periodic 300-step snapshots.
- Affected code path: `coding_empo/project_EMPO/EMPO/verl-empo/recipe/empo/src/dapo_ray_trainer.py`, `coding_empo/project_EMPO/EMPO/verl-empo/verl/trainer/ppo/ray_trainer.py`, `coding_edis_empo/scripts/run_A_empo_baseline.sh`, `coding_edis_empo/scripts/run_B_empo_oversampled_random.sh`, `coding_edis_empo/scripts/run_C_empo_oversampled_methodD.sh`, `coding_edis_empo/scripts/run_D_empo_oversampled_methodD_entropy_adv.sh`.
- Expected experiment effect: after each validation, a checkpoint enters `${CKPTS_DIR}/top_amc_checkpoints/` only if it is in the current top 3 AMC scores; lower-ranked top checkpoints are automatically deleted, while `global_step_*` periodic saves and `latest_checkpointed_iteration.txt` remain controlled by `save_freq`.
- Open questions / follow-up runs:
  - done: local and server `py_compile` checks pass for modified trainer files
  - done: local and server `bash -n` checks pass for A/B/C/D scripts
  - done: server-side mocked top-k retention test kept the best three scores, pruned lower-ranked top checkpoints, skipped missing metrics, and did not write `latest_checkpointed_iteration.txt`
  - done: A/B/C/D server dry-runs resolve `trainer.topk_checkpoint_enable=True`, `trainer.topk_checkpoint_k=3`, `trainer.topk_checkpoint_metric=val-core/amc/acc/mean@16`, and `trainer.topk_checkpoint_dir_name=top_amc_checkpoints`

## 2026-04-16 5
- Decision: hard-lock online training validation in A/B/C/D scripts to AMC23 only.
- Reason: multi-dataset evaluation is for offline testing; training-time validation should remain cheap and should drive top-k checkpoint retention from a single comparable AMC23 metric.
- Affected code path: `coding_edis_empo/scripts/run_A_empo_baseline.sh`, `coding_edis_empo/scripts/run_B_empo_oversampled_random.sh`, `coding_edis_empo/scripts/run_C_empo_oversampled_methodD.sh`, `coding_edis_empo/scripts/run_D_empo_oversampled_methodD_entropy_adv.sh`.
- Expected experiment effect: the training scripts always pass only `amc23_1010.parquet` to `data.val_files`; setting `VAL_FILES` to any non-AMC23 value now fails fast instead of silently running extra validation datasets.
- Open questions / follow-up runs:
  - done: pushed `coding_edis_empo` commit `fc03d91` and pulled it on `direct-server`
  - done: server `bash -n` passed for A/B/C/D scripts
  - done: server dry-runs for A/B/C/D resolve `data.val_files` to only `amc23_1010.parquet` and keep `topk_checkpoint_metric=val-core/amc/acc/mean@16`
  - done: server dry-run with `VAL_FILES=math500` fails fast before launching training
  - keep non-AMC23 datasets in separate offline-eval scripts or notebooks, not in training launch scripts

## 2026-04-17 1
- Decision: skip generated training batches where balanced largest-vs-rest selection leaves zero selected response tokens, instead of raising a fatal runtime error.
- Reason: method D can occasionally produce an all-underfilled batch when every prompt's largest/rest pools fail the `4+4` quota. This is a valid outcome of the underfilled skip policy and should discard that generated batch without optimizer update, not terminate a multi-hour run.
- Affected code path: `coding_empo/project_EMPO/EMPO/verl-empo/recipe/empo/src/dapo_ray_trainer.py`.
- Expected experiment effect: method D continues training after rare all-skipped batches, logs `selection/empty_training_batch=1`, keeps the same `global_steps` value for the next real optimizer update, and does not run validation or checkpointing for the skipped batch.
- Open questions / follow-up runs:
  - done: pushed Verl commit `7ea83ae` and pulled it on `direct-server`
  - done: local and server `py_compile` pass for `recipe/empo/src/dapo_ray_trainer.py`
  - done: server D dry-run still resolves `balanced_largest_vs_rest`, `use_binary_reward`, `entropy_high_token_ratio`, `4+4`, AMC23-only validation, and top-k AMC checkpoint metric
  - monitor `selection/empty_training_batch`; if it appears frequently, the `4+4` quota is too strict for the current model/checkpoint distribution

## 2026-04-17 2
- Decision: fix resume-step parsing for score-suffixed top-k checkpoint directories.
- Reason: top-k AMC checkpoints are saved as `global_step_<step>_score_<score>`. The previous loader tried to parse everything after `global_step_` as an integer, so resuming from `global_step_340_score_0.537500` failed to recover step 340 and reset `global_steps` to 0. This also caused W&B to reject resumed logs as out-of-order when the run already had later steps.
- Affected code path: `coding_empo/project_EMPO/EMPO/verl-empo/verl/trainer/ppo/ray_trainer.py`.
- Expected experiment effect: resuming from top-k AMC checkpoints restores the correct training step and continues with monotonic W&B logging when the resumed checkpoint is not older than the latest W&B step.
- Open questions / follow-up runs:
  - done: pushed Verl commit `821f0c0` and pulled it on `direct-server`
  - done: local and server `py_compile` pass for `verl/trainer/ppo/ray_trainer.py`
  - done: regex smoke test extracts `340` from `global_step_340_score_0.537500`
  - if intentionally rolling back to a top-k checkpoint older than the current W&B run's latest logged step, use a new W&B run/branch rather than merging into the old run, because W&B cannot accept lower step numbers into the same run

## 2026-04-18 1
- Decision: add an offline multi-checkpoint evaluation workflow in `coding_edis_empo`, separate from online training validation.
- Reason: online validation must remain AMC23-only and cheap, while final checkpoint comparison needs math, multiple-choice natural-language, and coding benchmarks with clean per-checkpoint summaries.
- Affected code path: `coding_edis_empo/evaluation/`, `coding_edis_empo/scripts/run_offline_eval_queue.sh`, `coding_empo/project_EMPO/EMPO/verl-empo/verl/trainer/ppo/ray_trainer.py`.
- Expected experiment effect: a YAML queue can evaluate checkpoints one by one on GPUs 0/3, generate grouped metrics for AMC23/AIME24/MATH500, MMLU-Pro/GPQA-Diamond, and LiveCodeBench, and write `summary.csv`, `summary.json`, and `summary.md`.
- Open questions / follow-up runs:
  - LiveCodeBench is intentionally isolated under `.venv_lcb` and `external/LiveCodeBench` so its dependencies do not modify the training venv
  - MMLU-Pro uses a fixed stratified 1000-question subset; LiveCodeBench uses a fixed 200-question `release_v6` subset
  - optional `math_row_limit` and `nl_row_limit` support tiny per-run smoke subsets without changing the fixed official subset artifacts
  - first full run should start with one checkpoint and inspect `eval_runs/<run_id>/summary.md` before queueing many checkpoints

## 2026-04-19 1
- Decision: validate the offline evaluation workflow with server-side static checks, parquet schema checks, and a fresh one-row math/NL val-only smoke on GPUs `0,3`.
- Reason: long offline evaluations are expensive, so the queue runner must prove dataset routing, checkpoint loading, val-only metric JSON output, and summary generation before running full checkpoint comparisons.
- Affected code path: `coding_edis_empo/evaluation/run_offline_eval_queue.py`, `coding_edis_empo/evaluation/prepare_eval_subsets.py`, `coding_edis_empo/scripts/run_offline_eval_queue.sh`, `coding_empo/project_EMPO/EMPO/verl-empo/verl/trainer/ppo/ray_trainer.py`.
- Expected experiment effect: AMC23/AIME24/MATH500/MMLU-Pro/GPQA-Diamond offline evals can run from a YAML queue and produce `summary.csv`, `summary.json`, and `summary.md` without relying on console-log parsing.
- Open questions / follow-up runs: install or otherwise provide `latex2sympy2` if exact `eval_math` grading is required instead of the current fallback verifier.

## 2026-04-19 2
- Decision: patch LiveCodeBench bootstrap to use non-editable install, add training-venv site-packages to the isolated LCB runtime, and expose prompt assets inside the LCB work directory.
- Reason: the official LiveCodeBench repo currently fails editable install, the isolated `.venv_lcb` does not see training packages like `torch`/`vllm`, and upstream prompt modules load files through relative paths.
- Affected code path: `coding_edis_empo/evaluation/run_offline_eval_queue.py`, `coding_edis_empo/evaluation/run_livecodebench_subset.py`, remote ignored runtime dirs `coding_edis_empo/external/LiveCodeBench` and `coding_edis_empo/.venv_lcb`.
- Expected experiment effect: the coding evaluator can import the official LCB runner without modifying the training venv.
- Open questions / follow-up runs: full LCB subset/evaluation is blocked by server connectivity to `huggingface.co`; rerun once HF access or a cached `livecodebench/code_generation_lite` dataset is available.

## 2026-04-19 3
- Decision: support an offline local parquet for LiveCodeBench `code_generation_lite release_v6` instead of requiring Hugging Face access at runtime.
- Reason: the server cannot reach `huggingface.co`, but the uploaded parquet contains the full official `test` split with `question_id`, problem content, starter code, public/private tests, and metadata.
- Affected code path: `coding_edis_empo/evaluation/prepare_eval_subsets.py`, `coding_edis_empo/evaluation/run_livecodebench_subset.py`, `coding_edis_empo/evaluation/run_offline_eval_queue.py`, `coding_edis_empo/evaluation/checkpoint_queue.yaml`.
- Expected experiment effect: offline coding evaluation can prepare fixed LCB subsets and run the official code-generation evaluator from local data.
- Open questions / follow-up runs: for coding-only evaluation, use a clean GPU and `livecodebench_tensor_parallel_size=1` when possible; stale memory on shared GPUs can break vLLM memory profiling.

## 2026-04-19 4
- Decision: allow coding-only LiveCodeBench evaluation to use a GPU count that differs from the original FSDP checkpoint world size.
- Reason: LiveCodeBench first merges FSDP shards into a Hugging Face model, so inference tensor parallelism is independent of the training checkpoint shard count. The world-size equality check is still required for Verl math/NL val-only resume.
- Affected code path: `coding_edis_empo/evaluation/run_offline_eval_queue.py`.
- Expected experiment effect: coding-only smoke and full LCB eval can run on one clean 48 GB GPU such as GPU 6, while math/NL evaluation still protects against mismatched FSDP resume configs.
- Open questions / follow-up runs: the one-problem LCB smoke passed on GPU 6 with `pass@1=0.0`; run the fixed 200-question subset once GPU availability is confirmed.

## 2026-04-19 5
- Decision: move offline-evaluation run configuration into `scripts/run_offline_eval_queue.sh` instead of requiring users to edit/pass a YAML queue.
- Reason: the requested workflow should make datasets, checkpoints, GPUs, and smoke/full settings editable directly in one script while still generating a resolved YAML internally for reproducibility.
- Affected code path: `coding_edis_empo/scripts/run_offline_eval_queue.sh`, `coding_edis_empo/evaluation/run_offline_eval_queue.py`.
- Expected experiment effect: users can run multi-checkpoint offline eval by editing one script; the result includes `summary_simple.txt/json` with dataset accuracies and group averages.
- Open questions / follow-up runs: full math/NL eval still needs a clean two-GPU pair matching the FSDP checkpoint world size; coding-only LCB can run on one clean 48 GB GPU.

## 2026-04-19 6
- Decision: cap the fixed offline MMLU-Pro subset at 300 questions and replace the previous 1000-question artifact with `mmlu_pro_eval_seed20260418_n300`.
- Reason: 1000 MMLU-Pro questions are still too expensive for routine multi-checkpoint offline evaluation; 300 keeps the subset deterministic and more practical.
- Affected code path: `coding_edis_empo/evaluation/prepare_eval_subsets.py`, `coding_edis_empo/evaluation/run_offline_eval_queue.py`, `coding_edis_empo/scripts/run_offline_eval_queue.sh`, `coding_edis_empo/evaluation/subsets/`.
- Expected experiment effect: offline NL evaluation uses the fixed 300-question stratified MMLU-Pro subset plus full GPQA-Diamond by default.
- Open questions / follow-up runs: `direct-server-2` has four idle L20 GPUs and no project workspace yet; set it up as a separate offline-evaluation worker by copying repos/data/model/checkpoints from `direct-server` over the internal network.

## 2026-04-19 7
- Decision: introduce a canonical non-Git runtime root at `coding_edis_empo/runtime` and use server-2 `/data` through `/home/pterence6699` symlinks for large assets.
- Reason: active scripts should stop depending on `coding_edis` for venv/model/data paths, while server-2 must avoid writing large checkpoints to the 50 GB overlay filesystem.
- Affected code path: `coding_edis_empo/scripts/runtime_env.sh`, A/B/C/D launch scripts, offline eval scripts, MCQ conversion utility, and runtime setup/sync helper scripts.
- Expected experiment effect: both servers can expose the same visible project paths while storing non-Git assets under the canonical runtime directory; Git sync remains for code and `scp` remains for models, datasets, venvs, and checkpoints.
- Open questions / follow-up runs:
  - server-2 requires one manual sudo step to make `/data/pterence6699` writable before symlink setup
  - old path fallbacks remain temporarily for validation and should be removed after both servers pass training and offline-eval smoke tests
  - done: server-2 symlinks `/home/pterence6699/coding_edis_empo` and `/home/pterence6699/coding_empo` resolve to `/data/pterence6699/...`
  - done: bootstrapped server-2 repos from server-1 Git bundles because direct GitHub fetch was unreliable on server-2
  - done: copied runtime model, math data, LiveCodeBench parquet, and training venv to server-2 over direct server-to-server `scp`
  - done: server-2 `bash -n`, Python compile, `torch.cuda.device_count()==4`, and A/B/C/D dry-runs pass with runtime model/data paths
  - done: server-2 one-step real smoke tests pass for A baseline, B random oversample, C method-D balanced selection, and D entropy advantage weighting with W&B disabled
  - done: removed server-2 smoke checkpoints/logs and transfer bundles after verification; GPUs returned to 0 MiB used
  - done: copied W&B authentication from server-1 to server-2 via `~/.netrc`; server-2 W&B API resolves user `pterence6699`

## 2026-04-19 8
- Decision: expose the training model as an explicit `MODEL_PATH` per A/B/C/D launch script instead of indirectly resolving by model name.
- Reason: the user wants model switching to be visible and editable in each script, for example switching directly between `Qwen2.5-math-1.5B` and `Qwen3-1.7B-Base` without relying on hidden name-resolution defaults.
- Affected code path: `coding_edis_empo/scripts/run_A_empo_baseline.sh`, `coding_edis_empo/scripts/run_B_empo_oversampled_random.sh`, `coding_edis_empo/scripts/run_C_empo_oversampled_methodD.sh`, `coding_edis_empo/scripts/run_D_empo_oversampled_methodD_entropy_adv.sh`.
- Expected experiment effect: model choice becomes explicit in script config blocks and experiment names track the chosen `MODEL_TAG` directly.
- Open questions / follow-up runs: after syncing to both servers, update any script that should target `Qwen3-1.7B-Base` by editing `MODEL_PATH` and `MODEL_TAG` together.

## 2026-04-19 9
- Decision: harden offline evaluation defaults by lowering rollout footprint and allowing automatic free-GPU selection.
- Reason: the previous offline-eval failures moved from a context-length assertion to resource contention and OOM; server logs showed the selected pair was already occupied by another user's Ray workers, so the script should avoid obviously busy GPUs and use safer eval memory defaults.
- Affected code path: `coding_edis_empo/scripts/run_offline_eval_queue.sh`.
- Expected experiment effect: offline eval defaults now use `MAX_PROMPT_LENGTH=1024`, `MAX_RESPONSE_LENGTH=3072`, `VAL_BATCH_SIZE=1`, `GPU_MEMORY_UTILIZATION=0.30`, and `GPU_IDS=\"auto\"` to pick a sufficiently free 2-GPU pair instead of assuming a fixed server-specific pair.
- Open questions / follow-up runs: the current math/NL eval path still uses the heavy Verl val-only stack; if OOM persists even on clean GPUs, replace math/NL offline eval with merged-HF + direct vLLM generation instead of the training runner.

## 2026-04-19 10
- Decision: remove FSDP resume loading from offline math/NL val-only evaluation and force FSDP checkpoints through merge-to-HF before eval.
- Reason: val-only runs were failing on `optimizer.load_state_dict` with parameter-group mismatch because the evaluator passed `trainer.resume_from_path` and restored training optimizer state, which is unnecessary for offline scoring.
- Affected code path: `coding_edis_empo/evaluation/run_offline_eval_queue.py`.
- Expected experiment effect: offline eval now loads model weights from merged Hugging Face checkpoints for math/NL/coding uniformly, allowing checkpoint evaluation on different GPU counts without optimizer-restore failures.
- Open questions / follow-up runs: sync this patch to server, rerun `scripts/run_offline_eval_queue.sh`, and confirm `math.log`/`nl.log` no longer include `trainer.resume_from_path` or optimizer-state restore errors.

## 2026-04-20 1
- Decision: shrink default offline-eval subsets to 100 for `math500`, `mmlu_pro`, `gpqa_diamond`, and `livecodebench`, while keeping `amc23` and `aime24` full.
- Reason: full offline queue with 5 checkpoints and high sample counts was too slow for iteration; we need a faster screening configuration that is still large enough to compare methods.
- Affected code path: `coding_edis_empo/scripts/run_offline_eval_queue.sh`, `coding_edis_empo/evaluation/run_offline_eval_queue.py`.
- Expected experiment effect: much shorter offline eval runtime per checkpoint with explicit per-dataset caps instead of globally shrinking all math/NL datasets.
- Open questions / follow-up runs: after fast screening, rerun top checkpoints on full sets for final report confidence intervals.

## 2026-04-20 2
- Decision: make offline-eval GPU selection manual-only and set the default validation batch size to `null`.
- Reason: `GPU_IDS=auto` made runs less predictable, and `VAL_BATCH_SIZE=1` forced the Verl validation loop to generate one prompt batch at a time, unlike online AMC23 validation where a null validation batch sends the validation set to the inference engine in larger scheduled batches.
- Affected code path: `coding_edis_empo/scripts/run_offline_eval_queue.sh`, `coding_edis_empo/evaluation/run_offline_eval_queue.py`.
- Expected experiment effect: offline math/NL evaluation should have fewer generation-loop iterations and closer behavior to online validation, while GPU allocation remains explicit in the script.
- Open questions / follow-up runs: restart the offline queue after syncing; the active run keeps the old `VAL_BATCH_SIZE=1` setting because its resolved YAML was already generated.

## 2026-04-20 1
- Decision: switch training scripts to default `RAY_ADDRESS=local` (A/B/C/D) and print `cuda_visible` + `ray_address` in A/B startup config logs.
- Reason: concurrent runs in separate tmux sessions were attaching to a shared Ray endpoint (`http://localhost:8265`) and causing cross-run resource interference; HAMI + `nvidia-smi` top table is also misleading under this environment, so explicit launch context is required in logs.
- Affected code path: `coding_edis_empo/scripts/run_A_empo_baseline.sh`, `coding_edis_empo/scripts/run_B_empo_oversampled_random.sh`, `coding_edis_empo/scripts/run_C_empo_oversampled_methodD.sh`, `coding_edis_empo/scripts/run_D_empo_oversampled_methodD_entropy_adv.sh`.
- Expected experiment effect: each training script starts an isolated local Ray runtime by default, reducing multi-session collisions when A/B/C/D are launched simultaneously on different GPU subsets.
- Open questions / follow-up runs: after current runs finish, sync scripts to server2 and continue using explicit launch env (`CUDA_VISIBLE_DEVICES=...`, `RAY_ADDRESS=local`, `GPU_MEMORY_UTILIZATION=...`) for paired concurrent experiments.

## 2026-04-20 2
- Decision: set non-overlapping default GPU pairs for concurrent A/B runs on 4-GPU servers (`A -> 0,1`, `B -> 2,3`).
- Reason: previous defaults overlapped across scripts and could cause misleading OOM/initialization failures when launching A and B at the same time.
- Affected code path: `coding_edis_empo/scripts/run_A_empo_baseline.sh`, `coding_edis_empo/scripts/run_B_empo_oversampled_random.sh`.
- Expected experiment effect: out-of-the-box concurrent launch of A and B should map to disjoint GPU sets without extra manual environment variables.
- Open questions / follow-up runs: monitor first 10-20 minutes of both runs using process-level GPU query (`nvidia-smi --query-compute-apps ...`) because top-table memory under HAMI may still be inaccurate.

## 2026-04-20 3
- Decision: harden A/B/C/D rollout runtime defaults for HAMI/vLLM stability by disabling vLLM sleep mode and forcing eager execution in scripts.
- Reason: runs on server-2 repeatedly failed at vLLM sleep init (`Memory usage increased after sleeping`) and later hit CUDA/NCCL instability; script defaults were still `enforce_eager=False` and did not pass `enable_sleep_mode`, while D also had an aggressive `GPU_MEMORY_UTILIZATION=0.95`.
- Affected code path: `coding_edis_empo/scripts/run_A_empo_baseline.sh`, `coding_edis_empo/scripts/run_B_empo_oversampled_random.sh`, `coding_edis_empo/scripts/run_C_empo_oversampled_methodD.sh`, `coding_edis_empo/scripts/run_D_empo_oversampled_methodD_entropy_adv.sh`.
- Expected experiment effect: lower startup crash rate and fewer runtime memory-assert failures under HAMI; defaults are now `ROLLOUT_ENFORCE_EAGER=True`, `VLLM_ENABLE_SLEEP_MODE=False`, and safer `GPU_MEMORY_UTILIZATION` for oversampled scripts.
- Open questions / follow-up runs: run short smoke tests for A and B first, then C and D; if illegal-memory failures persist, next step is to reduce rollout footprint further (`MAX_RESPONSE_LENGTH`, `MAX_NUM_BATCHED_TOKENS`, dynamic micro-batch settings) with a dedicated server-2 profile.

## 2026-04-20 4
- Decision: guard vLLM sleep assertions in the FSDP/Megatron sharding-manager exit path.
- Reason: A 4-GPU Qwen3 baseline run still crashed after generation in `fsdp_vllm.py::__exit__` with `Memory usage increased after sleeping`; the earlier guard only covered rollout initialization, not this post-generation sleep call.
- Affected code path: `coding_empo/project_EMPO/EMPO/verl-empo/verl/workers/sharding_manager/fsdp_vllm.py`, `coding_empo/project_EMPO/EMPO/verl-empo/verl/workers/sharding_manager/megatron_vllm.py`.
- Expected experiment effect: HAMI/vLLM sleep bookkeeping assertions no longer abort otherwise valid training steps.
- Open questions / follow-up runs: resume the interrupted A run after syncing server-2; if a new failure appears, inspect the next unguarded vLLM memory-management path rather than changing experiment logic.

## 2026-04-20 5
- Decision: remove per-run evaluation subset materialization and make capped offline-eval subsets fixed under `evaluation/subsets/`.
- Reason: run-local `limited_subsets/` made results harder to compare and could silently create different artifacts per run; once a seed/size subset is chosen, it should be reused globally.
- Affected code path: `coding_edis_empo/evaluation/prepare_eval_subsets.py`, `coding_edis_empo/evaluation/run_offline_eval_queue.py`, `coding_edis_empo/evaluation/README.md`.
- Expected experiment effect: Math500, GPQA-Diamond, optional AMC23/AIME24 smoke limits, MMLU-Pro, and LiveCodeBench all use stable subset files in `evaluation/subsets/`.
- Open questions / follow-up runs: future eval run directories can be simplified further by moving logs/raw metrics under `_artifacts/`, leaving only summaries and resolved config at the top level.

## 2026-04-20 6
- Decision: expose offline-eval sampling parameters in `scripts/run_offline_eval_queue.sh` instead of hardcoding them in Python.
- Reason: temperature/top-p/top-k are experimental settings and must be visible in the script-generated queue for reproducibility and easy adjustment.
- Affected code path: `coding_edis_empo/scripts/run_offline_eval_queue.sh`, `coding_edis_empo/evaluation/run_offline_eval_queue.py`.
- Expected experiment effect: math/NL eval now uses script-controlled `VAL_DO_SAMPLE`, `VAL_TEMPERATURE`, `VAL_TOP_P`, `VAL_TOP_K`; LiveCodeBench uses script-controlled `LIVECODEBENCH_TEMPERATURE` and `LIVECODEBENCH_TOP_P`.
- Open questions / follow-up runs: consider adding these values to `summary.md` header if comparing runs with different sampling settings becomes common.

## 2026-04-20 7
- Decision: add GSM8K to the offline math evaluation group as a fixed capped subset.
- Reason: offline model comparison should cover arithmetic word problems in addition to AMC23, AIME24, and MATH500; GSM8K is small enough to materialize once as an eval-ready RLHF parquet and then reuse deterministic subsets.
- Affected code path: `coding_edis_empo/evaluation/prepare_eval_subsets.py`, `coding_edis_empo/evaluation/run_offline_eval_queue.py`, `coding_edis_empo/scripts/run_offline_eval_queue.sh`, `coding_edis_empo/evaluation/README.md`.
- Expected experiment effect: future offline math evals include `gsm8k` and report it in `summary_simple.txt/json`; `GSM8K_SUBSET_SIZE` is script-configurable and capped at 300.
- Open questions / follow-up runs: server-1 cannot reach Hugging Face, so upload raw GSM8K as `runtime/math_data/gsm8k_test.parquet` or `gsm8k_test.jsonl`; then run a small GSM8K-only smoke eval after the current GPU 4/5 evaluation finishes.

## 2026-04-20 8
- Decision: investigate AMC23 online/offline mismatch by comparing exact metric keys and resolved validation settings before changing evaluation logic.
- Reason: current top-AMC checkpoint metadata and training logs monitor `val-core/amc/acc/mean@16`, while the latest offline run is also extracting `mean@16`; therefore metric-key mismatch is not the cause.
- Affected code path: read-only inspection of `topk_checkpoints.json`, training console logs, and offline `queue.resolved.yaml` / `math.json`.
- Expected experiment effect: the next controlled check should rerun one checkpoint on AMC23-only with online-matching settings and compare the merged-HF offline path against the saved online score.
- Open questions / follow-up runs: if temp/top-p/top-k/response-length settings are confirmed identical and the gap persists, verify FSDP-to-HF merge equivalence and tokenizer/chat-template consistency.

## 2026-04-20 9
- Decision: treat the apparent AMC23 online/offline mismatch as a combination of metric-surface ambiguity and stochastic validation, not as confirmed model-load corruption.
- Reason: the temp-0.2 offline result under `evaluation_result/qwen2.5_1` only preserves summaries, but comparing its AMC23 `mean@16` values to the selected online top-k checkpoint scores gives differences of -1.4pp, -2.2pp, -4.1pp, and +1.6pp for A/B/C/D. AMC23 has only 40 prompts and `n=16`, so one sampled `mean@16` estimate has high variance; top-k online scores are also winner-biased because they are selected from many stochastic validation calls.
- Affected code path: `coding_edis_empo/evaluation/run_offline_eval_queue.py` summary semantics, A/B/C/D script `TOPK_CHECKPOINT_METRIC` defaults, offline eval result interpretation.
- Expected experiment effect: future reports should distinguish `mean@16` from `maj@16`; if the project metric is majority vote, both online top-k checkpoint selection and offline summary primary metric should be switched to `val-core/amc/acc/maj@16/mean`.
- Open questions / follow-up runs: run AMC23-only offline re-evaluation 3-5 times for the same checkpoint with temp 0.2, then report mean/std; decide whether to change future top-k checkpoint selection from `mean@16` to `maj@16`.

## 2026-04-21 1
- Decision: replace server1 user-level proxy runtime from old Mihomo service to provider-supplied Clash under /home/pterence6699/clash.
- Reason: user bought a new provider and had downloaded its Clash binary/config; old Mihomo occupied 127.0.0.1:7890/7891/9090 and prevented the new Clash from starting.
- Changes made: disabled and stopped user mihomo.service; downloaded Clash-compatible Country.mmdb via the old proxy once; added and enabled user clash.service pointing to /home/pterence6699/clash/clash -d /home/pterence6699/clash.
- Verification: clash.service active; ports 7890, 7891, and 9090 are owned by clash; proxy checks returned OpenAI 401, GitHub 200, Hugging Face 200, and Google generate_204 204.
- Rollback note: old Mihomo files remain under /home/pterence6699/.config/mihomo and /home/pterence6699/.local/bin/mihomo, but mihomo.service is disabled.

## 2026-04-21 2
- Decision: install the same provider-supplied Clash runtime on server2 under /home/pterence6699/clash.
- Reason: user requested server2 to use the new proxy provider as well; server2 has no usable user systemd bus, so a local clashctl script manages the process.
- Changes made: downloaded Clash binary/config on server2; copied Clash-compatible Country.mmdb from server1 because server2 could not resolve cdn.jsdelivr.net; added /home/pterence6699/clash/clashctl for start/stop/restart/status/log/nodes/use/test.
- Verification: Clash running on server2 pid 2511814; ports 7890, 7891, and 9090 are owned by clash; proxy checks returned OpenAI 401, GitHub 200, and Hugging Face 200.
- Operational note: server1 uses user systemd clash.service; server2 uses /home/pterence6699/clash/clashctl because systemctl --user is unavailable there.

## 2026-04-21 3
- Decision: standardize Clash operations on both servers through ~/clash/clashctl.
- Reason: server1 uses user systemd while server2 lacks a user systemd bus; a unified wrapper hides that difference and provides the same start/stop/restart/status/test/nodes/use/log/on/off/help interface.
- Changes made: installed the unified clashctl wrapper on server1 and server2. On server1 it delegates lifecycle commands to clash.service; on server2 it manages the pid/nohup process directly.
- Verification: ~/clash/clashctl status and nodes work on both servers; ~/clash/clashctl test returned OpenAI 401, GitHub 200, and Hugging Face 200 on both.

## 2026-04-21 4
- Decision: diagnose Google/GitHub access failures as missing per-terminal proxy environment rather than Clash service failure.
- Reason: on both servers, explicit curl -x http://127.0.0.1:7890 reaches Google/GitHub, while direct curl fails with DNS resolution timeout/failure; SSH sessions had no proxy env set.
- Changes made: extended ~/clash/clashctl on both servers with a shell command that opens a new shell with http_proxy/https_proxy/all_proxy and uppercase variants set.
- Verification: explicit proxy env curl to GitHub returns HTTP 200 on both servers.

## 2026-04-21 5
- Decision: make new interactive bash terminals default to the local Clash proxy whenever Clash is running.
- Reason: user expects ordinary commands to use the proxy after starting Clash; the diagnosed failures were caused by missing proxy env in normal shells, not by Clash being down.
- Changes made: installed a clash auto-proxy block in ~/.bashrc on both servers, with proxy_on/proxy_off/proxy_status/proxy_set helpers and automatic http_proxy/https_proxy/all_proxy exports when 127.0.0.1:7890 is reachable. Backups saved as ~/.bashrc.bak.clash-auto-20260421.
- Verification: fresh interactive bash shells on both servers show proxy env set automatically; GitHub works without explicit curl -x; server2 Google works; server1 Google works after switching current node from US W01 to Hong Kong W01.

## 2026-04-22 1
- Decision: split offline evaluation randomness into fixed dataset subset seed and per-run generation seed.
- Reason: repeated offline evals on the same checkpoint were exactly identical because Verl/vLLM defaulted to rollout seed 0; the user wants offline eval to behave more like online validation, where sampled generations can differ slightly across runs.
- Affected code path: `coding_edis_empo/scripts/run_offline_eval_queue.sh`, `coding_edis_empo/evaluation/run_offline_eval_queue.py`, Verl val-only Hydra argument `actor_rollout_ref.rollout.seed`.
- Expected experiment effect: fixed subsets remain reproducible, but future offline math/NL validation runs produce new sampled generations unless `EVAL_SEED` is pinned.
- Open questions / follow-up runs: if exact repeat remains for LiveCodeBench, inspect the LiveCodeBench wrapper cache separately because this change targets Verl math/NL validation.

## 2026-04-22 2
- Decision: replace the active ABCD training surface with one EMPO-binary 2x2 script controlled by `OVERSAMPLED` and `WEIGHTED`.
- Reason: tutor roadmap removed balanced sample selection and reduced experiments to oversampling on/off plus entropy advantage weighting on/off; all settings should use first-round EMPO binary 1/0 vote-tag reward.
- Affected code path: `coding_edis_empo/scripts/run_empo_binary_2x2.sh`, legacy `run_A/B/C/D*.sh` wrappers, `src/config/edis_empo_trainer.yaml`, `src/edis_reward_manager.py`, `src/main.py`.
- Expected experiment effect: off/off is now binary-tag EMPO rather than old entropy-frequency EMPO; on/off matches old random oversampled B; on/on is random oversampled plus cluster-aware entropy advantage weighting, without Method-D balanced selection.
- Open questions / follow-up runs: run four short smoke tests on server after syncing to confirm resolved `M`, `K`, reward method, selected counts, and `adv_weight/*` metrics for weighted settings.

## 2026-04-22 3
- Decision: reorganize helper scripts so active experiment scripts stay uncluttered.
- Reason: `runtime_env.sh` is a shared dependency for training/eval, while server mirror and runtime migration scripts are operational tools, not experiment entrypoints.
- Affected code path: moved `scripts/runtime_env.sh` to `scripts/lib/runtime_env.sh`; moved server setup/sync/migration scripts to `scripts/server_ops/`; updated training/eval scripts to source the new runtime-env path.
- Expected experiment effect: no runtime behavior change; active scripts remain `run_empo_binary_2x2.sh` and `run_offline_eval_queue.sh`.
- Open questions / follow-up runs: after Git sync, run `bash -n` on server copies and use dry-run once to verify the new source path.
