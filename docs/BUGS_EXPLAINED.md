# TGN Bugs — Simple Explanation

## Bug 1: dtype mismatch (type mismatch on timestamp buffer)

**What happened:** PyG's TGNMemory stores `last_update` (timestamp of last interaction per node) as `Long` (integer). But TGN feeds Float timestamps (e.g. `16847713.42`). Writing Float into Long buffer causes in-place mutation error during backward.

**Fix location:** `src/models/tgn_model.py`, line 99 → `batch_max = t.max().long()` converts Float → Long BEFORE storage.

## Bug 2: GRU never receives gradients

**What happened:** PyG splits memory into two pieces:
1. `forward()` — reads memory, makes predictions
2. `update_state()` — writes new memory state

Training loop calls `update_state()` AFTER `loss.backward()`. The GRU inside `update_state()` runs too late — gradients already finished flowing. GRU weights never get trained. They stay at random initialization forever.

**Fix location:** `src/models/tgn_model.py`, EMA update runs INSIDE the forward pass:
- Line 78: `msg_proj` produces message (trainable, inside forward)
- Line 94: EMA update `new_memory = 0.85 * old + 0.15 * msg` (inside forward)
- Line 97: Detached copy saved to persistent buffer
- Line 105: Non-detached copy returned → flows through edge classifier → loss → backward

**Gradient path:** `loss` → `edge_classifier` → `new_memory` → `msg_proj` ✅

**Analogy:** PyG GRU = student does homework after teacher graded it. Our EMA = student hands in homework during class and gets a real grade.

## Bug 3: Data leakage — train/eval memory mismatch (MOST CRITICAL)

**The bug:** During training, the model updated memory FIRST, then predicted from the UPDATED memory. During eval, it correctly used OLD memory. The model learned to cheat — it read its own just-stored edge features from memory instead of learning laundering patterns. AUC-ROC collapsed from 0.88 → 0.73 at eval.

**Fix location:** `src/models/tgn_model.py`, line 249 — always snapshot OLD memory BEFORE any update. `msg_proj` is trained through the edge classifier input (line 256 → line 133), NOT through memory. Train and eval both use old memory → identical behavior.

**Analogy:** Writing the answer in your notes, then reading your notes to "figure out" the answer. Cheating. The fix: only use notes from BEFORE the current question.

## Bug 4: Gradient clipping destroys minority-class signal

**What is a gradient?** The signal that tells each model weight "how wrong was this prediction, and which way should I adjust?" Big gradient = very wrong, make a big correction. Small gradient = almost right, just tweak.

**What is gradient clipping?** A safety cap: "no gradient can exceed max value." If total gradient norm > threshold (e.g. 1.0), ALL gradients are scaled down to fit. Used to prevent unstable training from outlier batches.

**The bug:** `pos_weight=12.4` means laundering gradients are 12.4× larger than legitimate ones. With `grad_clip=1.0`, nearly all laundering gradients exceed the cap and get chopped — regardless of whether the example was easy or hard. The model receives the same flattened signal for every laundering case and never learns to distinguish patterns.

**Fix:** `--grad_clip 0` (disable clipping entirely). Let the full 12.4× gradient flow. Epoch 1 AUC-ROC jumped from **0.794 → 0.934**.

**Analogy:** You give a megaphone (pos_weight) to a whisperer (laundering signal). A security guard (grad_clip) caps all sound at 80dB. The whisperer with megaphone = exactly 80dB, same as background noise. Still can't hear. Fix: fire the security guard.
