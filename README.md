# ZeRO from Scratch

Implementing data parallelism and ZeRO stages 1–3 on 32 virtual GPUs, and measuring
whether the memory and communication behave as the theory says.

All four schemes were verified to produce parameters **identical to data parallelism at
every step** — `max |θ − θ_DP| = 0` across 33,088 parameters and 10 steps. Without that,
a memory reduction proves nothing.

---

## What this is

<!-- WRITE THIS YOURSELF — 3 to 5 sentences.
     Suggested content:
       - what problem ZeRO solves, in your own words
       - why you simulated it rather than using DeepSpeed or FSDP2
       - what you were trying to find out
     Avoid restating the paper abstract. Say what YOU wanted to see. -->

---

## How to run it

```bash
git clone https://github.com/<you>/zero-from-scratch
cd zero-from-scratch
pip install numpy matplotlib pytest
pytest tests -q
```

Or open `notebooks/zero_stages_walkthrough.ipynb` in Colab and run all cells. No GPU
required — the 32 GPUs are simulated with numpy arrays.

---

## Results

### Memory per GPU

Measured at N = 32, on a 33,088-parameter MLP.

| Scheme | Formula | Bytes/param | On one GPU | vs DP |
|---|---|---:|---:|---:|
| data parallel | `16` | 16.0000 | 529,408 | 1.00× |
| ZeRO-1 | `4 + 12/N` | 4.3750 | 144,760 | 3.66× |
| ZeRO-2 | `2 + 14/N` | 2.4375 | 80,652 | 6.56× |
| ZeRO-3 | `16/N` | 0.5000 | 16,544 | **32.00×** |

Under data parallelism the cluster holds **16,941,056 bytes** to store **529,408 bytes**
of information — 32 identical copies.

### Memory across world size

| N | data parallel | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---:|---:|---:|---:|---:|
| 1 | 16.0000 | 16.0000 | 16.0000 | 16.0000 |
| 2 | 16.0000 | 10.0000 | 9.0000 | 8.0000 |
| 4 | 16.0000 | 7.0000 | 5.5000 | 4.0000 |
| 8 | 16.0000 | 5.5000 | 3.7500 | 2.0000 |
| 16 | 16.0000 | 4.7500 | 2.8750 | 1.0000 |
| 32 | 16.0000 | 4.3750 | 2.4375 | 0.5000 |

<!-- WRITE THIS YOURSELF — 2 to 4 sentences on what the SHAPE of this table means.
     Look at the ZeRO-1 column going down: 16, 10, 7, 5.5, 4.75, 4.375.
     What is it approaching, and why can it never get past it?
     What does that imply for a model too large to fit under ZeRO-1? -->

### Communication per step

In multiples of P, one copy of the parameters.

| N | data parallel | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---:|---:|---:|---:|---:|
| 2 | 1.0000 | 1.0000 | 1.0000 | 1.5000 |
| 4 | 1.5000 | 1.5000 | 1.5000 | 2.2500 |
| 8 | 1.7500 | 1.7500 | 1.7500 | 2.6250 |
| 16 | 1.8750 | 1.8750 | 1.8750 | 2.8125 |
| 32 | 1.9375 | 1.9375 | 1.9375 | 2.9062 |

Collective breakdown at N = 32:

| Scheme | all-gather | reduce-scatter | all-reduce | total |
|---|---:|---:|---:|---:|
| data parallel | — | — | 1.93750 | 1.93750 |
| ZeRO-1 | 0.96875 | 0.96875 | — | 1.93750 |
| ZeRO-2 | 0.96875 | 0.96875 | — | 1.93750 |
| ZeRO-3 | 1.93750 | 0.96875 | — | 2.90625 |

<!-- WRITE THIS YOURSELF — 3 to 5 sentences.
     The first three rows are identical. Why?
     ZeRO-3's all-gather is exactly twice the others. Why?
     What does "free" mean here, and what is NOT free? -->

### Correctness

| Scheme | max &#124;θ − θ_DP&#124; | Final loss |
|---|---:|---:|
| data parallel | — | 1.021900 |
| ZeRO-1 | 0 | 1.021900 |
| ZeRO-2 | 0 | 1.021900 |
| ZeRO-3 | 0 | 1.021900 |

The absolute loss value depends on the batch seed and is not the claim being tested —
what matters is that all four columns are the same value. Re-running with a different
seed changes the number and leaves the agreement intact.

---

## What each stage shards, and why

### Data parallel: the baseline and its waste

<!-- WRITE THIS YOURSELF.
     What does every GPU hold? Which of the five bands are genuinely needed
     on every card, and which are there for no reason? -->

### ZeRO-1: the optimizer state

<!-- WRITE THIS YOURSELF.
     Which three bands, how many of the sixteen bytes, and why is it SAFE to shard
     them? (Hint: when during a step is the optimizer state actually read?) -->

### ZeRO-2: why it costs nothing extra

<!-- WRITE THIS YOURSELF — this is the most interesting one.
     In the code, ZeRO2 subclasses ZeRO1 and overrides ONE METHOD, which only
     changes the memory accounting. No new communication. Explain why that is
     correct rather than a shortcut. What was ZeRO-1 already throwing away? -->

### ZeRO-3: where the third P comes from

<!-- WRITE THIS YOURSELF.
     Why does a GPU holding 1/32 of the weights have to gather them TWICE per step?
     What happens if you gather once and keep it — and why is that a bug even though
     the loss comes out right? -->

---

## Three things the measurements say that the summaries don't

### 1. Traffic never reaches 2P

<!-- WRITE THIS YOURSELF.
     Measured: 1.9375 at N=32, not 2.0. The exact ring cost is 2(N−1)/N.
     Is the quoted "2P" wrong, or is it something else? When is it closest to true? -->

### 2. The ZeRO-3 peak figure is an artefact of the toy model

Resident and peak memory differ only for ZeRO-3:

| Scheme | Resident | Peak |
|---|---:|---:|
| data parallel | 16.0000 | 16.0000 |
| ZeRO-1 | 4.3750 | 4.3750 |
| ZeRO-2 | 2.4375 | 2.4375 |
| ZeRO-3 | 0.5000 | **4.5000** |

<!-- WRITE THIS YOURSELF.
     At N=32 ZeRO-3's peak (4.5) is WORSE than ZeRO-2's resident (2.44).
     Taken at face value that says ZeRO-3 is the wrong choice. It isn't.
     What does this simulator gather, and what does a real implementation gather?
     What would the peak be if it gathered the largest layer (16,384 of 33,088
     parameters) instead of the whole model? -->

### 3. Bitwise equality is a property of this implementation, not of ZeRO

<!-- WRITE THIS YOURSELF.
     Our reduction sums in the same order in every scheme, so the results agree
     exactly. A real framework using NCCL rings sums in a different order.
     What would you expect to see instead, and would that be a bug? -->

---

## What this simulator does not model

- **Activation memory.** Every figure here is the 16 bytes per parameter. On a real run
  the activations share the same card, and past a certain sequence length they are the
  larger pile.
- **Node topology.** All 32 GPUs communicate at one flat rate. Real hardware is roughly
  450 GB/s inside a node and 50 GB/s between nodes — a 9× difference. Byte counts here
  are right; converting them to seconds would need the topology.
- **Kernel time and overlap.** No compute timing, and no overlapping of communication
  with the backward pass.
- **Per-layer gathering.** ZeRO-3 gathers the whole model at once. See finding 2.

<!-- OPTIONAL: add anything else you noticed was missing. -->

---

## Files

```

notebooks/
  zero_stages_walkthrough.ipynb
docs/
  index.html    background reading — the theory these measurements test
```

---

## Background

`docs/` contains a written walkthrough of the ideas this prototype tests: the memory
tax, collectives, the cost of communication, the ZeRO stages, offload, and precision.
Open `docs/index.html`.
