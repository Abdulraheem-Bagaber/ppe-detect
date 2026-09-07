# ppe-detect

PPE compliance detection for work-site video: find people who are not wearing a
hard hat or a high-visibility vest, and raise an event.

**Status: Phase 0 (scaffold).** No model has been trained yet. Every metric
table below reads `TBD` and will stay that way until it has actually been
measured.

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

### CPU by default, GPU on request

The default install pulls **CPU-only torch**, matching the CPU-only Docker
image and keeping the lockfile portable. Training on CPU works; it is just
slow. To use a local CUDA card:

```bash
.\make.ps1 gpu
```

Device selection is automatic at runtime (`model.device: auto`), so no code
changes either way.

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

| Class | mAP@50 | mAP@50-95 | Precision | Recall |
| --- | --- | --- | --- | --- |
| person | TBD | TBD | TBD | TBD |
| helmet | TBD | TBD | TBD | TBD |
| vest | TBD | TBD | TBD | TBD |
| **all** | **TBD** | **TBD** | **TBD** | **TBD** |

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

| Backend | Device | Latency p50 | Latency p95 | FPS |
| --- | --- | --- | --- | --- |
| PyTorch | CPU | TBD | TBD | TBD |
| ONNX Runtime | CPU | TBD | TBD | TBD |
| ONNX INT8 | CPU | TBD | TBD | TBD |
| PyTorch | CUDA | TBD | TBD | TBD |

## Roadmap

| Phase | Scope | State |
| --- | --- | --- |
| 0 | Scaffold, tooling, interfaces | Done |
| 1 | Dataset preparation and class balance report | Next |
| 2 | yolov8n baseline, fixed seed, committed split manifest | |
| 3 | Rule layer, tests written first | |
| 4 | ByteTrack, debounce, work-area zones | |
| 5 | Violation-level evaluation harness | |
| 6 | ONNX export with parity test, FastAPI, Docker | |
| 7 | Face blurring, backend benchmarks | |

Out of scope by choice: gloves, boots, harnesses and face shields (two PPE
classes done properly beats six done badly), and a web dashboard (the API and
CLI are the deliverables).

## Licence

Project code: MIT. Note that installing the default `ultralytics` backend
brings AGPL-3.0 obligations into any deployment that serves it over a network;
see the licensing note above for how to avoid that.
