# Resubmission Repair Plan — AML GNN Thesis

**Status:** Report graded **5.0 → Repair & Resubmit (minor repairs)**. Deadline **17 August 2026** (confirmed by Kees). Defend 17–28 August.
**Goal:** Lift the three failed criteria (2.3, 2.4, 2.5) above 5.5 while protecting the two that already pass (2.1, 2.2 at 6.5), and do it genuinely well — not a patch job.

## Graded outcome (combined form, 30-06-2026)

| Criterion | Weight | Grade | |
|---|---|---|---|
| 2.1 Problem statement & RQs | 10% | 6.5 | Pass |
| 2.2 Theoretical framework | 10% | 6.5 | Pass |
| 2.3 Research methodology & tool development | 30% | **5.0** | **Fail** |
| 2.4 Results, analyses & tool performance | 20% | **5.0** | **Fail** |
| 2.5 Discussion, recommendations & conclusions | 30% | **5.0** | **Fail** |

Cap rule: any criterion < 5.5 caps the overall grade at the lowest failing grade → overall **5.0**.

## Hard constraints

- **40 body-page cap** (Calibri 11, single spacing; excludes title, abstract, TOC, references, appendices). Report already ~38 pages. **We must cut to add.** Figures/tables go to appendices (not counted).
- AI-tools policy: NON-ISSUE (confirmed with supervisor). Do not spend effort here.
- References / Q1 journals: already satisfied (≥15 Q1, journal-quality xlsm done). Do not remove/replace refs unless swapping for equal-or-better.
- **STANDALONE-REPORT RULE (no meta / no history):** The report is a self-contained final thesis. It must NOT narrate its own process or history. Forbidden: references to earlier drafts or the previous submission, examiner feedback, "corrections", "earlier framing", "what we did/found before", bug-fix-as-changelog narratives, and self-justifying words like "honest", "corrected", "now report", "mislabelled". Present the final design, methods, and results as they simply are. (Development rationale lives in the methodology chapter as forward-looking design decisions, not as a log of problems fixed.) Also drop gratuitous implementation detail the reader does not need: wall-clock training times and specific hardware (GPU model / VRAM) unless a claim actually depends on them. Applies to EVERY chapter, not just Ch4.

## Canonical source files (read at point-of-need, NOT wholesale — avoid context pollution / re-litigating locked decisions)

This plan is the operating spine and the distillation of the analysis below. Open a primary only when a writing workstream needs the examiner's verbatim wording for a specific criterion.
- **Graded assessment form (authoritative outcome + per-criterion comments):** `Master Project assessment form_combined.pdf`
- **Assessment rubric / grading model:** `assessment-model-report-tool-dev.md`
- **Submitted report (final, as graded):** `final/Graph Neural Networks Applied to Money Laundering Detection Final.pdf`
- **Research plan (pre-report; APPROVED — do not over-focus, historical context only):** `research-plan-draft.md`
- **Second-assessor comments on research plan v2 (Bhaumik, approved):** `Master Project research plan v2 DB Comments.pdf`

## What the examiners LIKED — preserve during cuts

- **2.1:** genuine empirical gap; clean, non-double-barrelled MRQ; well-scoped; sub-questions internally consistent, answerable, feasible.
- **2.2:** required domains covered; **FATF typologies → graph-structural signatures** connection made explicit; snapshot vs continuous-time distinction technically accurate.
- **2.3:** well-organised project structure (separated modules, dataclass config, logging, CLI); nine architectures matches master's level; **EMAMemory substitution = genuine (narrow) contribution** beyond library use.
- **2.4:** the **per-slice temporal-generalisation analysis** (memory accumulation) is a real strength — keep it, and now interpret it against typologies.
- **2.5:** the **limitations section is the strongest element** — honest, specific, concrete future directions. Replicability well addressed (fixed seeds, deterministic splits, Appendix B).

## Repair workstreams (de-duplicated from all 5 criteria)

| # | Workstream | Lifts | Effort | Notes |
|---|---|---|---|---|
| A | **EDA / graph characterisation**: degree dist, counterparty spread, connectivity, hubs, clustering, components; temporal dist of laundering over 18 days; feature dist laundering-vs-legit; show structuring/layering/fan-in-out empirically present | 2.3, 2.4 | New (data exists) | Figures → appendix; prose in Ch4 (and/or new §) |
| B | **Graph-vs-features ablation**: non-graph classifier on the *same* node+edge features the GNN sees, vs GNN — proves graph adds signal beyond hand-crafted relational features | 2.3, 2.4 | New — intellectual crux | The examiner's deepest challenge to the thesis premise |
| C | **Methodology rebuild**: adopt **DSR (Peffers DSRM 2007)**, map 6 phases to chapters, justify vs **CRISP-DM**; add internal/external **validity** section (protocol mismatch, synthetic-data generalisability) | 2.3 | New writing, no compute | Peffers (JMIS) + Hevner (MISQ) are Q1 bonus refs |
| D | **Results rigor**: training metrics + **learning curves** for all 9 models; **re-run static GNNs on chronological split**; disclose TGN **6-run search + val scores**; sensitivity + seed-repeat runs (TGN + GCN); interpret results vs typologies | 2.4 | Mostly re-run existing code | Check logs for per-epoch data before re-training |
| E | **Explainability thread**: lit in Ch2; effectiveness limitation in Ch4; concrete recommendation + FATF-2021 in Ch5; **optional low-cost GAT-attention demo** | 2.1, 2.2, 2.4, 2.5 | Writing (± small code) | Discussion is the required floor; demo is bonus |
| F | **Scalability demonstration**: run pipeline on progressively larger subsets, measure train/inference time, memory, throughput → scaling curves | 2.5 | Re-run existing code | Curves → appendix |
| G | **Problem/RQ realignment**: fix practical-claim-vs-design disconnect; rewrite SQ4 answerable (EXPAND path, keep practitioner framing) | 2.1 (foundational) | Writing | Do FIRST — everything refers back to it |
| H | **Code documentation**: docstrings + inline comments across `src/` and `experiments/` | 2.5 | Mechanical | |
| I | **Conclusions depth**: connect back to theory + typologies; deepen theoretical implications | 2.5 | Writing | |

Re-run/mechanical (D, F, H) is a large share → consistent with "minor repairs."

## Decisions (agreed)

1. **Problem/RQ (G): EXPAND, not narrow.** Narrowing would walk back Bhaumik's plan-stage LG6 requirement and gut LG5/LG6. Keep practitioner framing; make it honest by adding explainability + validity + ecological-validity treatment; rewrite SQ4 to ask about *evidence-based model-selection guidance + preconditions/limitations for deployment* (answerable by the actual study). Decisions 1 and 2 are one move: adding explainability is what makes EXPAND valid.
2. **Explainability (E): discussion is required (floor) + optional low-cost post-hoc demo as bonus.** Feedback nowhere requires *implementing* explainability — it says discuss/address/recommend across Ch2/Ch4/Ch5. Ch2 lit is GENERIC (attention-based, GNNExplainer, SHAP/IG) with a bridge sentence mapping methods to our models. **Demo model = a model we recommend (TGN, optionally GCN), NOT GAT** (GAT's AUC-PR 0.0958 is below XGBoost's 0.1511 — explaining a sub-baseline model undercuts the point). Method = model-agnostic post-hoc **permutation feature importance** (cheap, eval-only, no retrain; caveat: partial for TGN since prediction also depends on node memory); SHAP/IG only if time. Avoid GNNExplainer-on-custom-EMA-TGN (high risk). Confirm feasibility when building E.
3. **Compute (D): concentrate seed-repeats/sensitivity on TGN + GCN**, single runs elsewhere with stated rationale. Do NOT confirm with Kees (internal choice; avoid date-extension risk). Learning curves: check existing `results/logs/*.log` for per-epoch data before re-training.
4. **Methodology (C): DSR (Peffers DSRM 2007), justified vs CRISP-DM.** Examiner's own suggestion; maps 1:1 to chapters; Q1-journal bonus refs.
5. **Per-slice memory (D/A): option (i) — warm continuous-memory eval.** Change eval so memory carries continuously across the test set (warm-started from train/val), re-run per-slice (eval-only, NO retrain; load `TGNModel_best.pt`), add a warm-vs-cold control comparison + a per-slice laundering-prevalence plot. Leakage-free (predict with old memory). Note: the headline TGN test AUC-PR may shift (likely up). Executed in workstream D at build time, not now.

## Suggested sequencing (against 17 Aug)

1. **G** (rewrite problem statement + SQ4) — foundational, everything refers to it.
2. **A + B together** — read the code, build EDA + ablation (the crux; lifts both failed technical criteria).
3. **D** — regenerate curves from logs if possible; chronological re-run of static GNNs; disclose TGN search; TGN/GCN stability runs.
4. **C** — methodology chapter (DSR + validity).
5. **E** — explainability thread across Ch2/4/5 (+ optional GAT demo).
6. **F** — scalability runs.
7. **I** — deepen conclusions/theoretical implications.
8. **H** — docstrings/comments (can run in parallel / late).
9. **Page-budget pass** — cut to ≤40, moving figures to appendices, protecting the "liked" content above.

## Code audit — confirmed facts affecting the plan (2026-07-01)

**Split (folds into D; resolves examiner 2.4 #1):** submitted code splits ALL tiers **chronologically** (`build_static_graph` → `_time_based_split`; no random path exists). Report mislabels Tiers 1–2 as "random 70/15/15" (§3.3.3, Table 4.1/4.2 captions) and builds a "random-vs-chronological" comparison narrative that the code does not support. Baseline numbers match the chronological logs; static-GNN numbers (GCN 0.1882 etc.) are plausibly real chronological (test set has ~2× val laundering prevalence → higher test AUC-PR). **Action:** re-run GCN/GAT/SAGE once to confirm exact numbers + produce clean logs; correct report to ONE uniform chronological protocol; delete the random-vs-chronological narrative.

**Report↔code hyperparameter mismatches to reconcile (reproducibility; folds into D):**
- LR solver: report "liblinear" vs code `lbfgs`.
- RandomForest: report "100 estimators, max_depth 10" vs code `n_estimators=200, max_depth=20, min_samples_leaf=10` (so §4.1's "depth 10 too restrictive" analysis rests on wrong values).
- XGBoost: report "default hyperparameters" vs code `n_estimators=300, max_depth=8, lr=0.05, early_stopping=20`.

**Recoverable without re-running:** TGN's ~6-run hyperparameter search is fully in `tgn.log` → build the disclosure table (D) from it.

**Curves (D):** trainers log per-interval VAL metrics + train LOSS but store only val AUC-ROC in `history` and never compute per-epoch TRAIN AUC/PR → full train-vs-val curves need a small logging change + re-run; loss+val curves derivable now.

**Ablation (B) feasible with existing features:** baselines already ingest the 28-dim edge features; add src+dst 12-dim node features to a non-graph model → isolates message-passing value beyond hand-crafted relational features.

**Explainability demo (E):** GAT attention not exposed by `forward()` (needs `return_attention_weights=True`); TGN has no native attention (needs feature attribution). Final choice pending.

**Per-slice memory (correctness — HIGH priority):** CONFIRMED — `evaluate_per_time_slice` calls `_evaluate_split` per slice, which **resets memory at the start of every slice**, so memory does NOT accumulate across the test period. The §4.3.3 / §5.2 claim that "rising AUC-PR (0.05→0.45) = memory accumulating training + preceding-test history" is **not supported by the code**; the trend is most likely a laundering-prevalence-over-time confound (the full-test eval also cold-starts memory over the test set only). Fix options: (i) make eval carry memory continuously across the test set (warm from train/val), re-run per-slice, AND plot per-slice laundering prevalence to disentangle — legitimises the claim; or (ii) rewrite the interpretation honestly. Intersects the examiner's "analyse temporal distribution of laundering" ask (2.4).

**Snapshot temporal eval leak (minor; weakest models):** val = snapshot 8 only, but `evaluate_test` and threshold calibration both use snapshots 8–11 → snapshot 8 leaks into test and the calibrated threshold is tuned on test snapshots. Report claims disjoint (val=8, test=9–11). Fix code or correct report.

**TGN repro config:** `run_tgn.py` defaults memory_dim=128/time_dim=16 (→289K params); the documented reproduction command sets neither → would NOT reproduce the reported 119K-param model (memory_dim=64/time_dim=8). Pin the actual final config from the checkpoint and document the exact command. (TGN training also shuffles mini-batch order; only eval is strictly chronological — nuance the "strict chronological processing" wording.)

**Other consistency items (reproducibility):** metrics `auc_pr` uses trapezoidal `auc(recall,precision)` not `average_precision_score`; GCN/models use BatchNorm not mentioned in report. Low stakes; align report to code.

## Progress log

- **2026-07-01 — G (partial) applied.** Rewrote the §1.4 practical-claim paragraph + SQ4 + §1.6 interconnection sentence (EXPAND path: keep practitioner value, scope the claim honestly, name explainability/alert-burden/ecological-validity) in BOTH `chapter1_introduction.md` and `thesis_draft_complete.md`; added FATF (2021) reference to `references.md` and the combined draft's reference list.
  - **G follow-ups (apply when the answering content exists, not orphaned):** §1.5 Objective 5 (add explainability + ecological validity); §1.7 Contributions practitioner bullets (add interpretability + honest ecological-validity caveat); §5.1.4 / §5.3 must actually answer SQ4's three new dimensions (lands with workstream **E**); Ch2 explainability literature (workstream **E**).
- **2026-07-01 — SQ4 de-barrelled.** Reworked SQ4 to a single interrogative (guidance "once preconditions are taken into account") to avoid the double-barrelled pattern Bhaumik flagged on the plan-stage MRQ; applied to both files.
- **Canonical-file rule:** every report edit must be applied to BOTH `docs/report/chapter1_introduction.md` (+ sibling chapter files) AND `docs/report/thesis_draft_complete.md`, kept in sync.
- **Word-doc rule:** markdown is the working source of truth. Do NOT port edits to the submitted `.docx` per-change. Port to Word **once per chapter, only when that chapter is finalised** (preserve 11pt Calibri / single-spacing / tables), to avoid repeated re-work and drift. Nothing is chapter-final yet, so no Word edits now.
- **Code working-copy rule:** DEVELOP in the ROOT repo (`src/`, `experiments/`) — it has `data/raw/`, `results/checkpoints/` (needed for the warm-memory eval, `TGNModel_best.pt`), logs, and the venv. `submission/` is the CLEAN deliverable (stale files removed for upload) and is code-identical to root. Sync changed/new code files root -> `submission/` and re-zip at the very end. **Track every changed/new code file** so the final sync is complete.
- **2026-07-01 — C applied.** Rewrote §3.1 (DSR / Peffers DSRM, DSRM-to-thesis mapping Table 3.1, explicit DSR-vs-CRISP-DM justification) and expanded §3.6 to "Ethical Considerations, Validity, and Reliability" (internal/external/construct validity + reliability), in BOTH `chapter3_methodology.md` and `thesis_draft_complete.md`; updated the combined-draft TOC entry; added Hevner et al. (2004), Peffers et al. (2007), Wirth & Hipp (2000) to `references.md` + combined draft. Fixes the examiner's "no named methodology / no alternative comparison / no validity discussion" (criterion 2.3).
  - **Promises now in §3.6 that D/A MUST deliver before submission (else soften):** GCN+TGN seed-repeat + sensitivity runs (→ §4.4, workstream D); per-slice laundering-prevalence analysis (→ §4.1, workstream A); warm-memory eval (→ §4.3.3, workstream D/decision 5); TGN candidate-config disclosure (→ §3.5.2 + Appendix E, workstream D).
  - **Housekeeping:** new Table 3.1 (DSRM mapping) may clash with Appendix E's stray "(Table 3.1 in the main text)" reference to the hyperparameter table — reconcile (point Appendix E to Table E.1, or renumber) during the results/appendix pass. Standalone `TOC.md` / `draft_title_toc.md` also carry the old §3.6 title; sync at finalisation.

## 2026-07-01 — A + B (analysis) DONE

- **EDA** (`experiments/run_eda.py`, rebuilt fresh — old version had a timestamp-parse bug) -> `results/eda/*.png` + `eda_stats.json`. Key findings: giant component 72.2%; heavy-tailed degree (median 6, mean 20, max 169,756; 39.6% of accounts have <=2 counterparties); **laundering accounts structurally distinctive** (median degree 22 vs 6, counterparties 6 vs 3) — proves the GNN premise in-data; laundering amounts ~6x larger ($8,667 vs $1,408); ACH dominant laundering channel (0.75% vs 0.10%); typologies present (130 fan-out, 111 fan-in, 1,003 layering pass-through, 84 structuring); laundering prevalence rises over time (defuses per-slice confound); sampled avg clustering ~0.58 (report with sampling caveat).
- **Ablation** (`experiments/run_ablation.py`) chronological XGBoost: edge-only AUC-PR 0.1460, node-only 0.0187, edge+node 0.1144. **GCN (message-passing) rung pending D.** Node features alone ~useless -> GNN power is message passing, not the engineered features (strong for thesis).
- **Report bug surfaced:** actual payment formats = ACH / Bitcoin / Cash / Cheque / Credit Card / Reinvestment / Wire. Report §3.2.2 wrongly lists "Domestic Wire, International Wire, unknown". **Fix §3.2.2** (both files).
- **VERSION DRIFT CONFIRMED & QUANTIFIED:** XGBoost edge-only 0.1460 now vs 0.1511 reported (xgboost 3.2 vs 2.x). **D scope EXPANDS: re-run ALL tiers on the current env (chronological, consistent) and re-pin Appendix B.** Do NOT mix old + new numbers. Est. ~9 CPU-hours total; spread across the timeline.
- New code files to sync into `submission/` at the end: `experiments/run_eda.py` (rewritten), `experiments/run_ablation.py` (new).
- **§4.1 EDA narrative LOCKED** (approved 2026-07-01), staged in `docs/report/_staged_ch4_eda.md`; insert as new §4.1 + renumber Ch4 + fix cross-refs during the D rebuild. Ablation subsection to be written after GCN completes.
- **Payment-format fix applied** (§3.2.2 + Appendix A Table A.2, both files): actual formats ACH/Bitcoin/Cash/Cheque/Credit Card/Reinvestment/Wire.

## 2026-07-02 — E (Chapter 2 explainability) DONE

- Inserted **§2.3.3 "Explainability and Interpretability of GNN Predictions"** into `chapter2_theoretical_framework.md` + `thesis_draft_complete.md` (generic lit: GNNExplainer / attention / SHAP / integrated gradients + FATF-2021 regulatory grounding + bridge to our models: GAT native attention vs GCN/TGN post-hoc, TGN memory as an open problem). Added TOC entry; added 5 refs to both lists (**Yuan et al. 2023 TPAMI = Q1**; Ying 2019, Jain & Wallace 2019, Lundberg & Lee 2017, Sundararajan 2017). Grounds the SQ4 explainability dimension.
- **E remaining:** Ch4 effectiveness-limitation paragraph + Ch5 recommendation (both land in the Ch4/Ch5 passes) + optional permutation-importance demo on TGN/GCN.
- **Drift flagged:** `chapter2_theoretical_framework.md` vs `thesis_draft_complete.md` §2.3.2 ending text differs (chapter files and combined draft not fully in sync in Ch2). Reconcile during final consolidation.

## 2026-07-02 — GPU setup + D code changes (compute infra)

- **GPU env `venv_gpu`** (CUDA torch 2.5.1+cu121, PyG 2.8, GTX 1650 4GB). Kept separate from CPU `venv_new`. Run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **4GB OOM fix (faithful, not a model change):** split static GNNs (GCN/GAT/SAGE) into `encode()` (full-graph node embeddings) + `decode(idx=...)` (edge head); trainer `_train_epoch` now does gradient-accumulated mini-batch over train edges (`edge_batch=500k`). Mathematically equivalent to full-batch (verified: GPU epoch-1 loss 1.5285 / val AUC-ROC 0.548 == CPU). Eval also fixed to score only needed edges. Speed ~46s/epoch (was ~1.2min CPU).
- **Learning curves:** `trainer.py` + `tgn_trainer.py` + `temporal_trainer.py` persist per-epoch history to `results/curves/<Model>_history.json` (val every epoch; train metrics every log_interval to bound cost).
- **Warm-memory per-slice (decision 5) IMPLEMENTED** in `tgn_trainer.py`: `_warmup_memory` (replay train+val), `evaluate_test_warm`, `evaluate_per_time_slice_warm` (carries memory continuously, reports per-slice prevalence). `run_tgn.py` now logs cold-vs-warm per-slice + warm overall + saves `results/curves/tgn_per_slice.json`. Leakage-free (predict with old memory then advance).
- **Robustness harness** `experiments/run_robustness.py` (new): seed stability (42/123/7) + sensitivity (GCN dropout, TGN pos_weight_mult) for GCN & TGN; writes `results/curves/robustness.json` incrementally.
- **Epoch policy:** 200 (GNN) / 100 (TGN) with early stopping (patience 25) — no quality compromise, stops when converged.
- **Overnight sleep disabled** on AC (powercfg standby/hibernate/monitor/disk = 0; lid-close = do nothing). Must stay plugged in.
- New/changed code to sync to `submission/` at end: models gcn/gat/sage (encode/decode), trainer.py, tgn_trainer.py, temporal_trainer.py, run_tgn.py, run_robustness.py (new), run_eda.py, run_ablation.py (new).

## 2026-07-02 (later) — overnight D compute: mostly DONE, GAT held, robustness running

- **Overnight batch (`run_all_overnight.sh`) run-1 completed CLEAN for all core models** on the fresh env, chronological, single protocol: baselines, ablation, GCN, TGN, SAGE, TemporalGCN, EvolveGCN-H. Logs in `results/logs/*.log`; curves in `results/curves/*_history.json`; TGN warm-vs-cold per-slice in `tgn_per_slice.json`.
  - **Fresh numbers (source of truth, replace ALL report tables — do NOT mix with old):** LR 0.9409/0.1249; XGBoost 0.9393/0.1460; **GCN 0.9708/0.2056** (cal F1 0.1971); SAGE 0.9452/0.0412; **TGN cold 0.9698/0.3213** (cal thr 0.1585), **TGN warm 0.9601/0.2708**; ablation edge-only 0.1460 / node-only 0.0187 / edge+node 0.1144. (AUC-ROC/AUC-PR.)
  - **⚠️ Warm-memory finding is OPPOSITE the plan's guess:** warm continuous-memory AUC-PR (0.2708) is LOWER than cold (0.3213), not higher. §4.3.3 must interpret honestly (memory accumulation does NOT lift the headline; cold per-edge cold-start is the stronger operating point). Decision-5 warm-vs-cold control is still valuable — just reframe.
- **A second overnight instance auto-relaunched ~09:02 after run-1 died** and re-did its checkpoint backup, moving run-1's fresh checkpoints into `results/checkpoints/_prev/` (the original submitted checkpoints there were overwritten — fine, we regenerate everything). **Restored run-1 checkpoints back to `results/checkpoints/` (copied; `_prev/` kept as backup).** Logs are append-mode so run-1 numbers survive; run-2 appended a FAILED baselines attempt + duplicate ablation (identical numbers — determinism confirmed).
- **GAT NOT done — HELD.** GAT OOM-kills on this laptop at ~epoch 10, **reproducibly** (run-1 GAT and a clean solo re-run both died at epoch 10 — CPU full-graph attention over 5M edges exhausts system RAM; GPU not an option at 4GB). **GAT does NOT block anything** except its single row in the Ch4 table (robustness=GCN/TGN only; scalability=representative model; all writing independent). **Decision: run GAT genuinely-new on a bigger-RAM machine (user's Mac if ≥16GB, else Kaggle 16GB GPU) — NOT reuse-as-if-new (integrity: report is re-graded on reproducibility).** Fill the GAT row last.
- **`run_robustness.py` hardened** (edit, sync to submission): added resume (`_load_existing`/`_done` — skip completed runs, survives relaunch on this flaky box) + `--model {both,gcn,tgn}` selector so GCN runs on GPU (venv_gpu/cuda) and TGN on CPU (venv_new) as separate, independently-resumable invocations; conditional graph building.
- **GCN robustness LAUNCHED** (background, venv_gpu, `--device cuda --model gcn --epochs_gnn 100 --seeds 42 123 7`; log `results/logs/robustness_gcn_rerun.log` + `results/logs/robustness.log`; results merge into `results/curves/robustness.json`). 100 epochs to match the headline GCN. 5 retrains (3 seed + 2 sensitivity) × ~82min GPU ≈ ~7h. **TGN robustness still to launch** (venv_new, `--device cpu --model tgn --epochs_tgn 100`) — run AFTER GCN (sequential, to avoid system-RAM contention on the flaky box), ~6.5h.
- **Scalability (F) still to run.** All writing workstreams (Ch4 table regen, §4.3.3 warm reinterpretation, ablation subsection, §4.1 EDA insert, E Ch4/Ch5, F, I) unblocked and independent of GAT.

## 2026-07-03 — ROBUSTNESS COMPLETE (workstream D compute DONE, machine stable overnight, GAT still parked)

- **All 10 robustness runs finished** on `results/curves/robustness.json` (resume edit verified working: TGN invocation logged "Resume: loaded 5 existing run(s)"). Machine ran ~14h / 10 retrains with **zero crashes** — the earlier OOM is confirmed GAT-specific (CPU full-graph attention), not general instability.
- **GCN robustness** (GPU, 100 epochs, matches headline): seed-stability n=3 → **AUC-ROC 0.9715 ± 0.0008 | AUC-PR 0.1776 ± 0.0203 | F1 0.2023 ± 0.0118**. Sensitivity dropout {0.2: PR .168 / 0.3: PR .206 / 0.5: PR .191, F1 collapses to .089}. NOTE: reported headline GCN AUC-PR (0.206, seed 42) is the FAVORABLE seed; seed-mean is ~0.178 — report GCN as mean±SD in §4.4.
- **TGN robustness** (CPU, 100 epochs): seed-stability n=3 → **AUC-ROC 0.9686 ± 0.0011 | AUC-PR(cold) 0.3396 ± 0.0131 | AUC-PR(warm) 0.2733 ± 0.0082 | F1 0.3450 ± 0.0382**. Sensitivity pos_weight_mult {0.005: PR .331 / 0.01: PR .321 / 0.02: PR .358} — robust across the sweep.
- **Headline verdict (seed-averaged, for §4.4 + §5):** AUC-ROC statistically TIED (GCN 0.9715 vs TGN 0.9686, both ~0.97). AUC-PR: TGN 0.340 ± 0.013 vs GCN 0.178 ± 0.020 — **non-overlapping bands, ~90% advantage that survives seed-averaging** (TGN's AUC-PR is even tighter than GCN's). The central claim is NOT seed luck; robustness STRENGTHENS it. TGN warm consistently < cold (per §4.3.3 reframe).
- **Still pending:** GAT (Mac ≥16GB / Kaggle, genuinely-new — parked, blocks only its table row); **Scalability (F)** not started; all writing passes (Ch4 regen incl. §4.4 robustness table, §4.1 EDA insert + per-slice prevalence, ablation subsection, §4.3.3 warm reframe, §3.3.3 split fix, TGN search disclosure §3.5.2/App E, E Ch4/Ch5, G follow-ups, I, H, page-cut, Appendix B version re-pin, final submission/ sync).

## 2026-07-04 — Ch4 REBUILD + Ch3 fixes + appendices + GAT DONE (both files synced)

- **GAT re-run on user's Mac (48GB M4 Pro):** heads=4 OOM'd (>48GB — confirms dense-attention memory wall); **heads=1 completed** → AUC-ROC 0.9575, AUC-PR 0.0912, F1 0.0898 (64,001 params). Ranks GCN 0.206 > XGB 0.146 > RF 0.125 > **GAT 0.091** > SAGE 0.041. Filled into Table 4.3, leaderboard (Table 4.8), §4.3 paragraph. Current pipeline code committed to git master (commit f6a1b14, branch merged) so the Mac could clone; `run_gnn.py` was already committed, model/trainer changes were not.
- **Ch4 fully rebuilt** (`chapter4_results.md`) and **synced into `thesis_draft_complete.md`** (script-based block replace, verified). New structure: 4.1 EDA, 4.2 baselines, 4.3 static, 4.4 ablation, 4.5 temporal (4.5.1 snapshot / 4.5.2 TGN / 4.5.3 per-slice), 4.6 cross-model, 4.7 robustness, 4.8 tool-perf. All fresh numbers; random-vs-chrono narrative deleted; §4.5.3 honestly reframed (reset-per-slice vs carried-continuously + prevalence, no "memory accumulation" overclaim, no meta/history); GAT in.
- **STANDALONE-REPORT RULE enforced** throughout Ch4 + Ch3 (no history/meta/"honest"/prior-submission language; no wall-clock times; no specific hardware). Applied per user feedback.
- **EDA figure fix:** user caught that G.5 looked contradictory (volume ~0 after day 10 but 60% laundering rate). Verified with data: days 10-17 hold only ~1,100 txns (invisible on linear millions-axis); ~42% of test positives sit in that dense tail. **Regenerated `05_temporal.png`** (log-volume axis + volume overlaid behind the rate line — self-explanatory), reworded §4.1 + G.5 caption, added a §3.6 validity sentence. **All 7 EDA captions visually verified against the actual PNGs.** `run_eda.py` edited (uncommitted).
- **Ch3 fixes (both files):** §3.3.3 → uniform chronological; baseline hyperparameters corrected to code (LR lbfgs, RF 200/20/leaf-10, XGB 300/8/0.05); training-times + Intel-i7 line removed; §3.4 param counts fixed (TemporalGCN 162,561; EvolveGCN **2,213,673** not 578K/33M; TGN **85,905** not 119K); §3.6 dense-tail sentence + cross-refs 4.3.3→4.5.3, 4.4→4.7; stale version list → Appendix B pointer; TGN "four issues discovered/resolved" meta reframed.
- **Appendices (both files):** B.1 exact versions (torch 2.12/PyG 2.8/xgb 3.2/np 2.4.6/pd 3.0.3/scipy 1.17.1 + CPU/GPU note); B.3 commands (epochs 100/60, TGN memory_dim/time_dim pin); B.5 expected-output refreshed; **Appendix D fully refreshed** (D.1-D.4, incl. warm/cold per-slice all 12 slices); E epoch caps + Table-3.1 ref; F baseline train/val/test refreshed; **new Appendix G (7 EDA figures)** + TOC entry.
- **Reference to add/verify:** cited "Velickovic et al., 2018" in the GAT §4.3 para — confirm it's in references (GAT paper; likely present).
- **STILL PENDING (new tasks 6/7/8):** (6) Ch5 headline numbers still OLD (GCN 0.1882, TGN 0.3195/119K, GAT 0.0958, XGB 0.1511, EvolveGCN 578K/33M) — update both files (ties to Ch5 E/I rewrite); (7) global "Section 4.x" cross-ref sweep (Ch1/Ch2/Ch5) after renumbering; (8) §3.4.4 remaining TGN bug-narrative standalone-rule review. Plus original workstreams E (Ch4/Ch5 explainability), F (scalability), I (conclusions), H (docstrings), page-cut, Word port. **Uncommitted:** `run_eda.py` + all `docs/report/*` edits + `resubmission_repair_plan.md`.

## 2026-07-04 (later) — Ch5 REVISED + whole-report number consistency DONE

- **Ch5 fully revised** (both files): all headline numbers + derived stats refreshed (GCN +41% / TGN +120% over XGBoost; TGN +56% over GCN; hierarchy TGN>GCN>XGB>RF>GAT>TemporalGCN>EvolveGCN>SAGE>LR; EvolveGCN 2.2M). **Deleted the leftover random-vs-chronological narrative** (§5.1.2/§5.1.5/§5.2 — it contradicted the corrected Ch3/Ch4). **Per-slice claim reframed** (§5.2) to the warm-vs-cold-at-low-prevalence framing (no "0.05→0.45 memory accumulation" overclaim). **§5.3 practitioner section reworked around TGN's fresh calibrated operating point** (22% precision / 42% recall; default 84%/17% noted as the high-precision option) — Table 5.1 + §5.3.2 alert-burden math redone; §5.4 limitations de-hardwared ("memory-constrained" not "single CPU"), EMA-vs-GRU bug reference softened.
- **Whole-report old-number sweep = 0** across Ch1-5 + appendices + combined draft. Report is now numerically self-consistent on the fresh chronological results.
- **STILL PENDING:** task 8 (§3.4.4 remaining TGN four-issues narrative standalone-rule review); workstreams **E** (Ch4 effectiveness-limitation para + Ch5 explainability recommendation + FATF-2021 + optional permutation demo), **F** (scalability run + writeup), **I** (deepen conclusions/theory), **H** (docstrings), page-cut to <=40, Word port. Decision-5 warm-eval, robustness, EDA, ablation all now reflected in the report.

## 2026-07-05 — feedback re-alignment review + two 2.4 gaps closed

- **Re-extracted the graded feedback PDF fresh** (`Master Project assessment form_combined.pdf`, via PyPDF2) and did a criterion-by-criterion alignment review. Verdict: **2.3 strongly addressed** (EDA + DSR + validity + ablation); **2.4 ~75%** (EDA/chronological/robustness/six-run/curves done; remaining: typology interpretation depth, explainability limitation); **2.5 least-addressed** (F scalability, H docstrings, E Ch5 recommendation, I conclusions all pending); **2.1** protected but SQ4 answer needs E; **2.2** explainability lit added but the "critical synthesis" + "why class imbalance hurts message-passing" asks remain (secondary). **Explainability (E) is the single biggest lever — cited across 2.1/2.2/2.4/2.5.**
- **CLOSED 2.4 bullet "disclose full six-run TGN search":** replaced the 3-category summary with the actual **six development runs** (Table E.2) parsed from `tgn.log` — params, pos_weight mult, lr, best val AUC-ROC/AUC-PR, minority-learned. Correction: all six used memory 128/time 16 (compact 64/8 was a later refinement); five failed to learn the minority class, gradient-clipping-off was decisive. §3.5.2 rewritten accordingly.
- **CLOSED 2.4 bullet "report training-set performance/convergence for every model":** generated **Figure F.1** (learning curves — val AUC-ROC + train loss per epoch for GCN/GAT/SAGE/TemporalGCN/EvolveGCN/TGN) from `results/curves/*_history.json` (GAT from its Mac log); added to Appendix F with an overfitting note (GCN train/val 0.950/0.947; SAGE 0.952/0.923). Script inline (uses matplotlib); PNG at `results/curves/learning_curves.png` (gitignored like EDA figs).
- **Wording fixes** (user review): dropped "only" (§4.1 tone); "exceeds available memory" -> "prohibitively memory-intensive, cost scaling with edges x heads" (machine-independent, §4.3/§5.1.2/§5.4); clarified warm-memory row (Table 4.6 — replays train+val then carries/updates through test); spelled out TGN dims (no "64/8" shorthand). §5.2 per-node-memory claim VERIFIED against §4.5.3 data (warm>cold at early slices: 0.095 vs 0.023 slice 0, 0.215 vs 0.048 slice 4).
- **§5.4/§5.5 critical read:** limitation->future-work mapping is coherent (not spam); **gap: no explainability limitation/future-direction** (add with E); "no ensemble methods" is the weakest bullet.
- Commits: `4d98398` (wording), `fdcce34` (six-run), `5c51301` (learning curves). Combined draft re-synced from chapter files via `scratchpad/resync_all.py` (Ch3/4/5/appendix blocks).
- **NEXT: E (explainability)** — Ch4 effectiveness-limitation para + §5.3 recommendation (FATF-2021) + §5.4 limitation + §5.5 future direction + optional permutation-importance demo. Then F, I, H, and the 2.2 secondary asks.

## 2026-07-05 (later) — E COMPLETE (incl. demo) + page-cut done

- **Workstream E fully complete.** Thread now spans: Ch2 §2.3.3 (lit) -> Ch4 §4.8 (interpretability limitation + post-hoc attribution demo) -> Ch5 §5.3.5 (recommendation, cites the demo) + §5.4 limitation + §5.5 future direction + §5.1.4 SQ4. Refs (FATF 2021, Ying, Lundberg, Sundararajan) already present.
- **Permutation-importance demo DONE.** `experiments/run_permutation_importance.py` (committed ac37da1) run by user on Mac (GCN 100 epochs, repeats=5, baseline test AUC-PR 0.2018). Result is coherent and higher-band: top feature **pmt_ACH** (AUC-PR drop 0.175 of 0.202 baseline — confirms EDA's ACH channel); amount + day-of-week next; top NODE features are structural (degree_out 0.048, degree_in 0.025, counterparties) -> closes loop with ablation (those relational features are useless as flat XGBoost inputs, 0.019, but the GCN uses them via message passing). Mac lacked matplotlib so figure was regenerated on Windows from the log values -> `results/curves/permutation_importance.png` = **Figure F.2** (Appendix F, edge=blue/node=orange). Note: Ctrl+T on macOS = SIGINFO (status print), does NOT kill the process.
- **Page-cut pass DONE** (commit f9dd6e0): removed the 3 triple-redundant per-tier tables (baselines/static/snapshot; kept in leaderboard + Appendix D), kept analysis tables inline, renumbered Ch4 tables to 4.1-4.7, fixed all cross-refs + stale "GAT pending" caption. Body table rows 71->57 (~0.5-0.7 page). Est. body ~38 pages; room for remaining additions; confirm only in Word render (backup levers: move EDA table to App G, or trim prose).
- **Remaining for higher band:** 2.2 secondary (critical evaluation of key works + why class imbalance hurts message-passing), **F** (scalability run+writeup — REQUIRED for 2.5, compute), **I** (conclusions depth + typology interpretation), **H** (docstrings). Commits this session: 4d98398, fdcce34, 5c51301, 40ef131, 9eeee3d, ac37da1, f9dd6e0, 3464c6f (all pushed to master).

## 2026-07-05 (even later) — F (scalability) DONE

- **Workstream F complete** (`experiments/run_scalability.py`, commit 0805a10; results 97c6edb). User ran on Mac (GCN, prefixes 20-100%, 3 timed epochs each). Result: training time grows super-linearly (1.7s@1M -> 13.5s@5M edges), **throughput falls 420k -> 264k edges/s** as graph grows 5x (empirically confirms full-batch message passing scales worse than TGN batched / SAGE sampled), inference ~linear (~1s@5M), memory sub-linear (~16.5GB@5M total RSS incl. dataset). §4.8 "Scalability" para now has real numbers (replaced "not run"); split into Scalability + Generalisability. Figure regenerated on Windows (Mac lacked matplotlib) -> `results/curves/scalability.png` = **Figure F.3** (Appendix F). No stale "scalability asserted" language remains.
- **All examiner-REQUIRED items now addressed:** 2.3 (EDA+DSR+validity+ablation), 2.4 (EDA, chronological, robustness, six-run TGN, learning curves, explainability limitation+demo, typology partial), 2.5 (scalability DONE, explainability recommendation DONE, docstrings=H pending, conclusions=I pending). Remaining are higher-band polish: **2.2 secondary** (critical eval of works + class-imbalance-x-message-passing), **I** (conclusions depth + deeper typology interpretation), **H** (docstrings, mechanical), task-8 (§3.4.4 narrative). Session commits through 97c6edb, all pushed.

## 2026-07-08 — 2.2, I, H DONE — repair substantively COMPLETE

- **2.2 secondary** (commit dfad22e): §2.3.1 critical evaluation of GCN/GAT/SAGE (limitations/transferability/implications); §2.5 mechanistic account of why message passing struggles under imbalance (minority-in-majority-neighbourhood + over-smoothing, tied to GraphSAGE result + design choices). Page-neutral (replacements).
- **I conclusions depth** (commit 0f588a7): new 5th theoretical implication in §5.2 (architecture<->typology correspondence: fan-in/out=relational=GCN, structuring/layering=sequential=TGN, grounded in §4.1); strengthened §5.6 integrating typology + explainability-precondition.
- **H docstrings** (commit 698d4a1): docstring coverage 74% -> **100% (153/153 defs + 4 package files)**; all src compiles; no behaviour change. Note: tracked .pyc artifacts updated too.
- **ALL examiner-required items + all higher-band levers now addressed.** 2.3/2.4/2.5 repairs complete; 2.1/2.2 protected+deepened. Every "what to do" bullet from the graded form is answered.
- **REMAINING (not repair-critical):** ~~task-8 (§3.4.4 TGN narrative tidy)~~ DONE 2026-07-08 (commit eeaa97c — reframed data-leakage + gradient-clipping paras as forward design rationale, "was fixed"->"was selected"); **submission/ folder rebuilt** 2026-07-08 (src synced w/ docstrings, +5 repair scripts, logs refreshed, .idea/pycache dropped, README+psutil updated; 47 files, untracked — zip is the deliverable); **final Word port** of all chapters (11pt Calibri/single-spacing, per Word-doc rule — nothing chapter-final was ported yet); **full read-through**; **sync changed code root->submission/ + re-zip** (run_permutation_importance.py, run_scalability.py new; many src docstrings; run_eda/run_tgn/trainers/models changed). Figures in results/ are gitignored (embed at Word-port). Commits this phase: dfad22e, 0f588a7, 698d4a1 (pushed).

## Pre-execution checks — RESOLVED (2026-07-01)

- [x] **Code audited** — root `src/` + `experiments/`; findings in "Code audit" section above.
- [x] **root ≡ submission (logically).** Diff of every differing file is 100% cosmetic (Unicode em-dash `—`/arrow `→` in root comments vs ASCII `--`/`->` in submission). NO logic differences. The audit applies to the graded code. **Final sync:** ASCII-fy any new comments when copying root -> `submission/`.
- [x] **Environment verified GREEN** — `venv_new/Scripts/python.exe`, Python 3.11.6. Smoke test passed (load 5.08M txns, feature-engineer, GCN forward). Installed: torch 2.12.0+cpu, PyG 2.8.0, sklearn 1.9.0, xgboost 3.2.0, numpy 2.4.6, pandas 3.0.3, scipy/matplotlib/seaborn/networkx present. Data files present.
  - **Version drift caveat:** installed libs are MUCH newer than the report's Appendix B pins (report: PyG 2.5.x / xgboost 2.x / numpy 1.x / pandas 2.x). Pipeline runs, but re-run numbers may differ slightly from originally-reported. Action in D: **re-pin Appendix B to actual versions** and treat freshly-generated numbers as source of truth (we regenerate all tables anyway).
- [x] **Logs checked** — per-interval VAL metrics + train LOSS present; no per-epoch TRAIN AUC/PR (see Curves item). TGN 6-run search recoverable from `tgn.log`.
- [x] **GAT-demo question moot** — decision 2 dropped GAT; demo (if any) = permutation importance on TGN/GCN.
