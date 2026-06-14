# Session Continuity Context

This file captures project context that lives outside CLAUDE.md — user preferences, grading targets, report status, and writing conventions. Read this alongside CLAUDE.md when picking up this project in a new session.

---

## User & project

- Shadab Gada (student #500981772), Master Digital Driven Business, Amsterdam University of Applied Sciences
- Supervisor: Kees van Montfort, PhD. Second assessor: Debarati Bhaumik, PhD
- Thesis: "Graph Neural Networks Applied to Money Laundering Detection (tool development)"
- Final deadline: 22 June 2026
- Target: highest grade band (8.5-10) per AUAS assessment rubric

## User preferences

- Proactive execution: write code → run → debug → fix → report. Don't ask permission for each step.
- Terse responses. No trailing summaries, no emojis, no markdown headers unless asked.
- Deep ML knowledge — don't explain basics, focus on design rationale and trade-offs.
- User copies markdown to Word for final formatting. Report uses bold text for section headings (not ##/###) for clean Word paste.

## Writing standards for thesis

- Academic yet natural English, not AI-sounding
- No em dashes, no special Unicode — ASCII only
- APA referencing throughout
- Every claim tied to a specific experimental result
- Present full model scope confidently — don't use defensive "this wasn't in the plan" language
- Chapter 5 §5.3 must provide concrete, actionable practitioner guidance (Debarati's requirement, Learning Goal 6)

## Grade targets per rubric criterion

| Criterion | Weight | Target band |
|-----------|--------|-------------|
| 1. Problem Statement & RQs | 10% | 8.5+ |
| 2. Theoretical Framework | 10% | 8.5+ |
| 3. Methodology | 30% | 8.5+ |
| 4. Results | 20% | 8.5+ |
| 5. Discussion/Conclusions | 30% | 8.5+ |

## Report status (2026-06-14)

All 5 chapters + appendices drafted. Combined file: `docs/report/thesis_draft_complete.md`.

| Chapter | Status |
|---------|--------|
| 1: Introduction | Drafted — Kees feedback applied |
| 2: Theoretical Framework | Drafted — no changes requested |
| 3: Methodology | Drafted — Kees feedback applied, N/A filled in tables |
| 4: Results | Drafted — all review fixes applied |
| 5: Discussion/Conclusions | Drafted — all review fixes applied |
| Appendices | Drafted — all review fixes applied |

Kees van Montfort reviewed Chapters 1-3 via tracked changes in .docx — all feedback applied.

## Review fixes applied to Chapters 4-5

1. Tables reordered by AUC-PR descending
2. Training time columns removed from all tables (Ch4 + Appendices + Table 5.1)
3. Em dashes replaced with N/A (not-applicable cells) or hyphens (project structure tree)
4. 12-slice explanation added to per-slice analysis in §4.3.3
5. Supervisor names removed from body text (title page only)
6. Synthetic dataset limitation softened — framed as domain constraint, not study flaw
7. Transaction count added to single dataset variant limitation
8. EMA memory limitation rephrased — removed "simpler and has fewer parameters"
9. "Real-world banking data" removed from future research and generalizability text

## Remaining TODOs before submission

- User to copy markdown to Word for final formatting
- Verify all cross-references between chapters
- Check APA reference list completeness
- Final proofread

## Key files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project overview, architecture, quick start, results |
| `docs/report/thesis_draft_complete.md` | Full thesis (all chapters + appendices, single file) |
| `docs/report/chapter1_introduction.md` | Individual chapter source |
| `docs/report/chapter2_theoretical_framework.md` | Individual chapter source |
| `docs/report/chapter3_methodology.md` | Individual chapter source |
| `docs/report/chapter4_results.md` | Individual chapter source |
| `docs/report/chapter5_discussion_conclusions.md` | Individual chapter source |
| `docs/report/appendices.md` | Appendices A-D |
| `docs/RESULTS.md` | Complete results leaderboard |
| `docs/THESIS_NARRATIVE.md` | Full research journey and novelty claims |

## Git conventions

- Commit message style: `area: brief description` (e.g., `docs: fix table ordering`)
- User creates commits — don't commit proactively unless asked
- Co-authored-by line not needed for standard commits
- Large files excluded via .gitignore (data/raw CSV, data/processed, checkpoints, logs)
