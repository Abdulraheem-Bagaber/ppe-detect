# ppe-detect

PPE compliance detection for work-site video: find people who are not wearing a
hard hat or a high-visibility vest, and raise an event.

**Status: baseline trained and measured; violation metrics still blocked.**
All seven phases are implemented and tested (307 tests). Detection metrics
below are real. The violation-level table stays `TBD` until annotated video
exists, because no public dataset provides it.

---

## The design decision this project is built around

The obvious approach is to train classes called `helmet` and `no_helmet`. This
project deliberately does not do that.

Training a `no_helmet` class bakes the safety policy into the model weights.
You then cannot answer "why was this flagged?" without appealing to the model,
you cannot change the policy for a site that only requires vests without
retraining, and you cannot unit-test the rule at all.

Instead:

1. The detector finds **`person`, `helmet`, `vest` as independent objects**. It
   is asked only what is visible, never what is compliant.
2. A separate **rule layer** associates them geometrically — a helmet belongs
   to a person if it sits in the upper band of that person's box, is
   sufficiently contained by it, and is near their centreline.
3. A person inside the work zone with no associated helmet or vest, sustained
   across enough consecutive frames, is a **violation**.

The rule layer is pure Python over plain dataclasses. It has no dependency on
the detector, on OpenCV, or on the web framework, so it can be exhaustively
unit-tested against hand-built geometry: two workers standing shoulder to
shoulder, a helmet carried at waist height, a person cropped by the frame edge.
Thresholds live in `configs/default.yaml`, so tuning site policy is a config
change, not a training run.

That boundary is enforced by `tests/test_import_boundaries.py`, which parses
the AST of every module under `src/ppe/rules/` and fails the build if any of
them imports `ultralytics`, `torch`, `cv2`, `supervision` or `fastapi`.

## Architecture

```
frame
  |
  v
Detector  (protocol; ultralytics today, ONNX in the served path)
  |  list[Detection]        person / helmet / vest, no compliance notion
  v
Tracker   (ByteTrack, people only)
  |  list[Track]            stable identity across frames
  v
Zones     (pure) -------> is this person inside the work area?
  |
  v
Association (pure) ------> PersonAssessment: matched / missing / not-judged
  |
  v
Debounce  (deterministic state machine, no clock)
  |
  v
list[ViolationEvent]  ->  JSON over the API, or a report from the eval harness
```

Layout:

| Path | Contains |
| --- | --- |
| `src/ppe/core/` | Shared vocabulary. Depends on nothing but numpy. |
| `src/ppe/detection/` | Detector backends. The only place ultralytics appears. |
| `src/ppe/rules/` | Association, zones, debounce. No I/O, no model. |
| `src/ppe/tracking/` | ByteTrack adapter. |
| `src/ppe/pipeline/` | Composition root and frame sources. |
| `src/ppe/privacy/` | Face blurring, applied before anything is emitted. |
| `src/ppe/api/` | FastAPI service. |
| `src/ppe/config/` | Pydantic schema and YAML loading. |
| `src/ppe/data/` | Dataset conversion, manifest, stats. Not on the runtime path. |
| `src/ppe/eval/` | Violation scoring and false-positive diagnosis. Not on the runtime path. |
| `scripts/` | Data prep, training, evaluation, export, benchmarks. |

### A note on licensing

Ultralytics YOLO is **AGPL-3.0**. That licence is viral over network use, which
matters for anything served over an API. It is confined to a single module,
`src/ppe/detection/ultralytics_detector.py`, behind the `Detector` protocol:

```python
class Detector(Protocol):
    @property
    def metadata(self) -> DetectorMetadata: ...
    def predict(self, frame: Frame) -> list[Detection]: ...
```

Replacing it with RT-DETR, YOLOX, or a plain ONNX graph under a permissive
licence means writing one new file and changing `model.backend` in the config.
A test asserts that no other module imports ultralytics, so the boundary cannot
erode quietly.

## Getting started

Requires Python 3.11 (provisioned automatically by `uv`) and [uv](https://docs.astral.sh/uv/).

```bash
pip install uv
```

```bash
uv sync --dev
```

```bash
uv run pytest
```

On Windows, `make` is not installed; `make.ps1` dispatches the same targets:

```bash
.\make.ps1 check
```

| Target | Does |
| --- | --- |
| `install` | Create the venv, install everything |
| `test` / `test-fast` | Test suite, with and without slow markers |
| `lint` / `format` / `typecheck` | ruff and mypy |
| `check` | All of the above, as CI runs it |
| `prepare-data` / `train` / `eval` | Phase 1, 2 and 5 pipelines |
| `export` / `benchmark` / `serve` | Phase 6 and 7 |

The full suite runs in about 5 seconds. One test is marked `slow` because
OpenCV's FFmpeg backend waits 30 seconds before giving up on an unreachable
stream; `test-fast` skips it.

### CPU by default, GPU on request

The default install pulls **CPU-only torch**, matching the CPU-only Docker
image and keeping the lockfile portable. Training on CPU works; it is just
slow. To use a local CUDA card:

```bash
.\make.ps1 gpu
```

Device selection is automatic at runtime (`model.device: auto`), so no code
changes either way.

## Data

### Getting the dataset

The project uses [**Worker-Safety**](https://universe.roboflow.com/computer-vision/worker-safety)
(1,206 images, **CC BY 4.0**), also distributed as `construction-safety-gsnvb` in
Intel's RF100 benchmark. It is the only shortlisted set that has full-body
`person` boxes, both PPE classes, a published split, and a licence clean enough
to put in a public repo.

Roboflow requires a free account to export, so this step is manual by design --
nothing in this repo downloads data automatically:

1. Sign in at [universe.roboflow.com](https://universe.roboflow.com/computer-vision/worker-safety)
2. **Dataset** -> **Download Dataset** -> format **YOLOv8** -> **Download zip**
3. Unzip so that `data/raw/worker-safety/data.yaml` exists

> **Generate your own version -- do not use the default v1 export.** Roboflow's
> published v1 of this dataset applies augmentation *before* export: 90-degree
> rotations (including upside-down), +/-15 degree rotation, shear, blur and
> salt-and-pepper noise, tripling 997 source images into 2,991. Upside-down
> training images are actively hostile to this architecture -- the association
> rule assumes a worn helmet sits near the top of its person, and in that export
> **47% of helmets sit in the lower half of their person box**. On Roboflow,
> fork the project and generate a version with augmentation set to none and
> preprocessing set to resize 640x640. Ultralytics applies its own sensible
> augmentation at training time; it does not need help, and it never flips
> images vertically. `scripts/dataset_stats.py` fails loudly if you skip this.

Then convert it into our three-class layout and check it:

```bash
uv run python scripts/prepare_data.py
```

```bash
uv run python scripts/dataset_stats.py
```

Conversion drops the source `no-helmet` and `no-vest` classes. That is the
whole design in one line: compliance is derived by the rule layer from a
missing association, never read off a label.

The mapping is declared in `configs/datasets/worker_safety.yaml` **by class
name, not index** -- a Roboflow re-export that reorders classes would otherwise
relabel every helmet as a vest without raising. Adding a second dataset is a
new YAML file, not a new code path. The split is committed to
`configs/manifests/worker-safety.json` so a reported metric stays reproducible.

### Datasets considered and rejected

| Dataset | Why not |
| --- | --- |
| SHWD | Classes are `hat` / `person`, where `person` means *a head without a helmet*. Head boxes only, negative class baked into the labels. |
| Hard Hat Workers | Derived from SHWD; `Workers` / `Head` / `Hard Hat`, no vest. |
| SH17 | 8,099 images and a real `Person` class, but only 927 helmet and **530 vest** instances (97 vests in test). Too sparse to train or honestly evaluate. |
| Pictor-v3 | Right classes (worker/hat/vest) and peer-reviewed, but no stated licence. |
| Ultralytics Construction-PPE | Good structure, but AGPL-3.0 on top of an already-AGPL detector. |

`scripts/dataset_stats.py` encodes the two checks that eliminated three of
these: a **person-box aspect-ratio check** (full-body boxes run 2-3x taller
than wide; roughly square means head boxes, and the association rules cannot
use them) and a **sparse-class check** calibrated against SH17's vest count.

### A known counter-argument

Nath, Behzadan & Paal (2020), the paper behind Pictor-v3, compared this
project's architecture against the alternative it rejects:

- *Approach-1*: detect worker/hat/vest independently, then verify -- **this project**
- *Approach-2*: detect combined `W`/`WH`/`WV`/`WHV` classes directly

They found Approach-2 scored higher, at 72.3% mAP. That result is about
detection mAP, which is explicitly not this system's headline metric, and it
says nothing about explainability, tunability without retraining, or
unit-testability. The trade is made deliberately, and the violation-level
metrics below are where the cost of it should show up.

### Evaluation footage is a known gap

There is no public event-level annotated PPE **video** benchmark. Every
published result on temporal PPE compliance uses private footage. The Phase 5
violation precision/recall numbers therefore require hand-annotated clips,
which do not exist yet. Tracked as the longest-lead item in the project.

## Configuration

Every threshold lives in `configs/default.yaml` — association bands,
containment thresholds, debounce counts, zone polygons, privacy settings. There
are no magic numbers in `src/`. Any value can be overridden from the
environment:

```bash
PPE_DEBOUNCE__CONFIRM_FRAMES=20 uv run python -m ppe.cli check-config
```

## Privacy

Face blurring is **on by default**, not opt-in, and is applied before any frame
is written to disk or returned by the API. Under Saudi PDPL, identifiable faces
in a workplace video stream are personal data; a misconfiguration should fail
closed rather than leak. Detection still runs on the unblurred frame, so
privacy costs no accuracy.

## Results

Nothing here has been measured yet. These tables are the deliverables of
Phases 2, 5 and 7 and will be filled in from real runs only.

### Detection (held-out test set)

yolov8n, 50 epochs, seed 1337, 997 training images, RTX 2060, 57 minutes.
Split committed as `configs/manifests/worker-safety.json`; full run config and
weights hash in `runs/train/baseline/run_manifest.json`.

| Class | mAP@50 | mAP@50-95 | Precision | Recall |
| --- | --- | --- | --- | --- |
| person | 0.880 | 0.572 | 0.849 | 0.874 |
| helmet | 0.927 | 0.490 | 0.895 | 0.921 |
| vest | 0.847 | 0.447 | 0.901 | **0.744** |
| **all** | **0.885** | **0.503** | **0.882** | **0.846** |

**Read the vest row, not the average.** Vest recall of 0.744 means roughly one
worn vest in four is not detected. In a system that infers non-compliance from
*absence*, a missed vest becomes a person falsely accused of not wearing one --
so this single number is the best available predictor of false alarms before
any video has been scored. It is also the class with the least training data
(1,343 instances against 2,543 helmets), which points at where more data would
pay.

No tuning was done. This is the documented baseline the next change gets
compared against.

### Violation events — the headline metric

Detection mAP is not what this system is judged on. What matters is whether a
supervisor gets alerted about the right people and is not buried in noise.

| Metric | Value |
| --- | --- |
| Event precision | TBD |
| Event recall | TBD |
| Event F1 | TBD |
| False alarms per hour | TBD |

### Why each false positive happened

The most useful output in the project: every false alarm attributed to a cause,
so effort goes where it pays.

| Cause | Count | Share |
| --- | --- | --- |
| Missed PPE detection | TBD | TBD |
| Association failure | TBD | TBD |
| Tracking ID switch | TBD | TBD |
| Annotation disagreement | TBD | TBD |

### Inference performance

Intel i5-10300H / RTX 2060, 640px, 50 frames, measured after warmup.
p95 rather than a mean, because the occasional slow frame is what decides
whether a live stream keeps up.

| Backend | Device | Latency p50 | Latency p95 | FPS |
| --- | --- | --- | --- | --- |
| PyTorch | CPU | 83.7 ms | 118.4 ms | 11.9 |
| ONNX Runtime | CPU | 77.5 ms | 114.1 ms | 12.9 |
| ONNX INT8 | CPU | not measured | | |
| PyTorch | CUDA | 18.2 ms | 21.5 ms | 55.1 |

ONNX buys about 8% on CPU here -- real but modest, and less than the usual
claim. The honest reading is that at 640px on yolov8n the bottleneck is not
graph execution overhead. INT8 quantisation is the untried lever and is not
reported because it has not been run.

**CPU is enough for this job.** A 12 FPS CPU path comfortably serves a 5 FPS
camera, which is a sane sampling rate for PPE compliance -- the debouncer needs
half a second of agreement anyway, so frames beyond that add cost without
adding evidence. The CUDA row is what makes *training* practical, not what
makes serving possible.

### ONNX export parity

| | Measured | Tolerance |
| --- | --- | --- |
| Box coordinate drift | 0.00104 px | 0.05 px |
| Class score drift | 1.4e-07 | 1e-04 |

The two are checked separately because they live on different scales -- box
coordinates in pixels on a 0-640 image, class scores in 0-1. A single absolute
threshold flagged a thousandth-of-a-pixel difference as a failure, which was a
bug in the check rather than in the export.

## Roadmap

| Phase | Scope | Code | Measured |
| --- | --- | --- | --- |
| 0 | Scaffold, tooling, interfaces | Done | n/a |
| 1 | Dataset preparation and class balance report | Done | Done |
| 2 | yolov8n baseline, fixed seed, split manifest | Done | **Done** |
| 3 | Rule layer, tests written first | Done | n/a |
| 4 | ByteTrack, debounce, work-area zones | Done | n/a |
| 5 | Violation-level evaluation harness | Done | **Blocked** |
| 6 | ONNX export with parity test, FastAPI, Docker | Done | **Done** |
| 7 | Face blurring, backend benchmarks | Done | **Done** |

### What "blocked" means

Three things gate every number in this README, and none of them are code:

1. ~~A clean dataset export.~~ Done -- regenerated without augmentation.
2. ~~A CUDA install.~~ Done -- `make gpu`, then train with `--no-sync`.
3. **Annotated video.** Still open. There is no public event-level PPE video
   benchmark. `configs/eval/clips.yaml` is committed empty with its schema
   documented; until clips are annotated there, the violation table cannot be
   filled in honestly. This is the longest-lead item in the project.

Note on `--no-sync`: `uv run` re-syncs from `uv.lock`, which pins CPU torch, so
a plain `uv run` silently undoes `make gpu`. The ML targets in the Makefile pass
`--no-sync` for that reason, and `scripts/train.py` warns if it finds an NVIDIA
card alongside a CPU-only torch build.

Out of scope by choice: gloves, boots, harnesses and face shields (two PPE
classes done properly beats six done badly), and a web dashboard (the API and
CLI are the deliverables).

## Licence

Project code: MIT. Note that installing the default `ultralytics` backend
brings AGPL-3.0 obligations into any deployment that serves it over a network;
see the licensing note above for how to avoid that.
