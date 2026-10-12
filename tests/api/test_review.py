"""Tests for the local debug viewer.

The viewer is development tooling rather than product, but it renders frames
to disk and serves them over HTTP, so it inherits two obligations from the
rest of the service: faces must be blurred before anything is written, and it
must not serve arbitrary paths.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ppe.api.app import create_app
from ppe.api.render import annotate, assessment_colour, contact_sheet
from ppe.config.schema import AppConfig, PrivacyConfig
from ppe.core.protocols import DetectorMetadata
from ppe.core.types import (
    BBox,
    Detection,
    Frame,
    PersonAssessment,
    PPEClass,
    SkipReason,
    ViolationEvent,
)
from ppe.pipeline.runner import FrameResult

PERSON = Detection(BBox(60.0, 40.0, 140.0, 280.0), PPEClass.PERSON, 0.9)


def noisy(width: int = 320, height: int = 320) -> Frame:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def result(
    assessments: list[PersonAssessment], events: list[ViolationEvent] | None = None
) -> FrameResult:
    return FrameResult(
        frame_index=10,
        timestamp_s=0.33,
        detections=[PERSON],
        tracks=[],
        assessments=assessments,
        events=events or [],
    )


class TestAssessmentColour:
    def test_compliant_is_green(self) -> None:
        assert assessment_colour(PersonAssessment(PERSON)) == (0, 200, 0)

    def test_violation_is_red(self) -> None:
        a = PersonAssessment(PERSON, missing=frozenset({PPEClass.HELMET}))
        assert assessment_colour(a) == (0, 0, 255)

    def test_unjudged_is_neither(self) -> None:
        """Three outcomes need three colours, or the viewer hides the third."""
        a = PersonAssessment(PERSON, evaluated=False, skip_reason=SkipReason.TOO_SMALL)
        colour = assessment_colour(a)
        assert colour not in {(0, 200, 0), (0, 0, 255)}


class TestAnnotate:
    def test_it_draws_without_changing_the_frame_size(self) -> None:
        frame = noisy()
        out = annotate(frame, result([PersonAssessment(PERSON)]), AppConfig())
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_it_does_not_mutate_the_input_frame(self) -> None:
        frame = noisy()
        snapshot = frame.copy()
        annotate(frame, result([PersonAssessment(PERSON)]), AppConfig())
        assert np.array_equal(frame, snapshot)

    def test_faces_are_blurred_before_anything_is_drawn(self) -> None:
        """The obligation the whole privacy design rests on.

        Detail inside the head region must collapse. A debug tool that leaked
        faces would be a worse leak than the API, because nobody audits it.
        """
        frame = noisy()
        out = annotate(frame, result([PersonAssessment(PERSON)]), AppConfig())
        head = PERSON.box.band(0.0, AppConfig().privacy.head_band_bottom)
        y1, y2 = int(head.y1) + 2, int(head.y2) - 2
        x1, x2 = int(head.x1) + 2, int(head.x2) - 2
        assert out[y1:y2, x1:x2].std() < frame[y1:y2, x1:x2].std() / 2

    def test_blurring_off_is_honoured_and_visible(self) -> None:
        config = AppConfig(privacy=PrivacyConfig(blur_faces=False))
        frame = noisy()
        out = annotate(frame, result([PersonAssessment(PERSON)]), config)
        head = PERSON.box.band(0.0, config.privacy.head_band_bottom)
        y1, y2 = int(head.y1) + 2, int(head.y2) - 2
        x1, x2 = int(head.x1) + 20, int(head.x2) - 20
        assert out[y1:y2, x1:x2].std() > frame[y1:y2, x1:x2].std() / 2

    def test_an_event_paints_a_banner(self) -> None:
        event = ViolationEvent(
            track_id=1,
            missing=frozenset({PPEClass.HELMET}),
            first_frame_index=0,
            frame_index=10,
            timestamp_s=0.33,
            person_box=PERSON.box,
            streak_length=15,
            mean_person_confidence=0.9,
        )
        plain = annotate(noisy(), result([PersonAssessment(PERSON)]), AppConfig())
        flagged = annotate(noisy(), result([PersonAssessment(PERSON)], [event]), AppConfig())
        assert not np.array_equal(plain[-20:], flagged[-20:])


class TestContactSheet:
    def test_no_frames_yields_nothing(self) -> None:
        assert contact_sheet([], AppConfig()) is None

    def test_it_grids_the_frames(self) -> None:
        pairs = [(noisy(), result([PersonAssessment(PERSON)])) for _ in range(4)]
        sheet = contact_sheet(pairs, AppConfig(), columns=3)
        assert sheet is not None
        assert sheet.shape[0] == 320 * 2  # two rows for four tiles at three wide
        assert sheet.shape[1] == 320 * 3


class StubDetector:
    @property
    def metadata(self) -> DetectorMetadata:
        return DetectorMetadata(
            name="stub",
            weights="stub.pt",
            weights_sha256="a" * 64,
            imgsz=640,
            conf_threshold=0.25,
            iou_threshold=0.45,
            device="cpu",
            class_map={},
        )

    def predict(self, frame: Frame) -> list[Detection]:
        return []


@pytest.fixture
def api() -> TestClient:
    app = create_app(AppConfig())
    with TestClient(app) as client:
        app.state.detector = StubDetector()
        app.state.detector_error = None
        yield client


def tiny_video(path: Path, frames: int = 6) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    for _ in range(frames):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()
    return path


class TestViewerPage:
    def test_the_page_is_served(self, api: TestClient) -> None:
        response = api.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_it_says_it_is_not_the_product(self, api: TestClient) -> None:
        """The spec said no dashboard. The page must not pretend to be one."""
        body = api.get("/").text
        assert "development tool" in body
        assert "not the product" in body


class TestReviewUpload:
    def test_it_analyses_a_clip_and_returns_a_video(self, api: TestClient, tmp_path: Path) -> None:
        video = tiny_video(tmp_path / "clip.mp4")
        response = api.post(
            "/review/upload",
            files={"file": ("clip.mp4", video.read_bytes(), "video/mp4")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["frames_processed"] > 0
        assert body["video_url"].startswith("/review/")
        assert body["blur_faces"] is True

    def test_the_rendered_video_is_downloadable(self, api: TestClient, tmp_path: Path) -> None:
        video = tiny_video(tmp_path / "clip.mp4")
        body = api.post(
            "/review/upload", files={"file": ("clip.mp4", video.read_bytes(), "video/mp4")}
        ).json()
        fetched = api.get(body["video_url"])
        assert fetched.status_code == 200
        assert len(fetched.content) > 0

    def test_a_junk_upload_is_a_client_error(self, api: TestClient) -> None:
        response = api.post(
            "/review/upload", files={"file": ("x.mp4", io.BytesIO(b"nonsense"), "video/mp4")}
        )
        assert response.status_code == 400

    def test_it_refuses_without_a_model(self, tmp_path: Path) -> None:
        from ppe.config.schema import ModelConfig

        app = create_app(AppConfig(model=ModelConfig(weights="nope.pt")))
        with TestClient(app) as client:
            video = tiny_video(tmp_path / "clip.mp4")
            response = client.post(
                "/review/upload",
                files={"file": ("clip.mp4", video.read_bytes(), "video/mp4")},
            )
        assert response.status_code == 503

    def test_stride_is_bounded(self, api: TestClient, tmp_path: Path) -> None:
        video = tiny_video(tmp_path / "clip.mp4")
        response = api.post(
            "/review/upload?stride=99",
            files={"file": ("clip.mp4", video.read_bytes(), "video/mp4")},
        )
        assert response.status_code == 422


class TestReviewAssetSecurity:
    """It serves files from disk over HTTP, so the path handling matters."""

    def test_a_bad_review_id_is_not_found(self, api: TestClient) -> None:
        assert api.get("/review/not-a-uuid/annotated.mp4").status_code == 404

    def test_an_arbitrary_filename_is_refused(self, api: TestClient) -> None:
        review_id = "a" * 32
        assert api.get(f"/review/{review_id}/secrets.txt").status_code == 404

    def test_traversal_in_the_id_is_refused(self, api: TestClient) -> None:
        assert api.get("/review/..%2f..%2fetc/annotated.mp4").status_code == 404

    def test_a_well_formed_but_absent_id_is_not_found(self, api: TestClient) -> None:
        assert api.get(f"/review/{'b' * 32}/annotated.mp4").status_code == 404


class TestLiveStream:
    """The streaming view: frames arrive while the pipeline is still running."""

    def _prepare(self, api: TestClient, tmp_path: Path) -> str:
        video = tiny_video(tmp_path / "clip.mp4", frames=8)
        response = api.post(
            "/review/prepare", files={"file": ("clip.mp4", video.read_bytes(), "video/mp4")}
        )
        assert response.status_code == 200
        return response.json()["review_id"]

    def test_prepare_returns_an_id_without_processing(
        self, api: TestClient, tmp_path: Path
    ) -> None:
        """An <img> can only GET, so the upload has to happen separately."""
        review_id = self._prepare(api, tmp_path)
        assert len(review_id) == 32

    def test_the_summary_is_not_available_until_the_run_finishes(
        self, api: TestClient, tmp_path: Path
    ) -> None:
        review_id = self._prepare(api, tmp_path)
        assert api.get(f"/review/{review_id}/summary").status_code == 409

    def test_the_stream_emits_multipart_jpeg(self, api: TestClient, tmp_path: Path) -> None:
        review_id = self._prepare(api, tmp_path)
        response = api.get(f"/review/{review_id}/live?stride=1")
        assert response.status_code == 200
        assert "multipart/x-mixed-replace" in response.headers["content-type"]
        assert response.content.count(b"Content-Type: image/jpeg") > 0

    def test_the_summary_lands_once_the_stream_completes(
        self, api: TestClient, tmp_path: Path
    ) -> None:
        review_id = self._prepare(api, tmp_path)
        assert api.get(f"/review/{review_id}/live?stride=1").content  # drain it
        summary = api.get(f"/review/{review_id}/summary")
        assert summary.status_code == 200
        assert summary.json()["frames_processed"] > 0

    def test_a_scrubbable_copy_is_written_alongside(self, api: TestClient, tmp_path: Path) -> None:
        """MJPEG cannot seek, so the mp4 still gets written for afterwards."""
        review_id = self._prepare(api, tmp_path)
        assert api.get(f"/review/{review_id}/live?stride=1").content
        assert api.get(f"/review/{review_id}/annotated.mp4").status_code == 200

    def test_stride_is_bounded_on_the_stream_too(self, api: TestClient, tmp_path: Path) -> None:
        review_id = self._prepare(api, tmp_path)
        assert api.get(f"/review/{review_id}/live?stride=99").status_code == 422

    def test_a_bad_id_does_not_start_a_run(self, api: TestClient) -> None:
        assert api.get("/review/not-a-uuid/live").status_code == 404


class TestRouteOrdering:
    """A catch-all `{name}` route silently shadowed /live and /summary.

    FastAPI matches in registration order and a path parameter matches
    anything, so `/review/{id}/{name}` registered first swallowed every later
    `/review/{id}/...` route. Both endpoints were correctly defined and both
    returned 404. These assert the specific routes win.
    """

    def test_live_is_not_swallowed_by_the_asset_route(
        self, api: TestClient, tmp_path: Path
    ) -> None:
        video = tiny_video(tmp_path / "c.mp4", frames=4)
        review_id = api.post(
            "/review/prepare", files={"file": ("c.mp4", video.read_bytes(), "video/mp4")}
        ).json()["review_id"]
        assert api.get(f"/review/{review_id}/live?stride=1").status_code == 200

    def test_summary_is_not_swallowed_by_the_asset_route(
        self, api: TestClient, tmp_path: Path
    ) -> None:
        video = tiny_video(tmp_path / "c.mp4", frames=4)
        review_id = api.post(
            "/review/prepare", files={"file": ("c.mp4", video.read_bytes(), "video/mp4")}
        ).json()["review_id"]
        # 409 rather than 404: the route resolved, the run simply has not run.
        assert api.get(f"/review/{review_id}/summary").status_code == 409

    def test_an_unknown_asset_name_is_still_refused(self, api: TestClient) -> None:
        assert api.get(f"/review/{'a' * 32}/secrets.txt").status_code == 404


class TestVideoIsBrowserPlayable:
    """The annotated copy must be H.264, or the player silently shows nothing.

    OpenCV's pip wheels carry no H.264 encoder, so VideoWriter quietly falls
    back to mp4v. That produces a valid .mp4 file which every major browser
    refuses to decode: the file exists, the request returns 200, the <video>
    element renders a black rectangle, and nothing reports an error anywhere.
    Asserting the codec is the only way this stays fixed.
    """

    def _encode(self, tmp_path: Path, frames: int = 6) -> Path:
        from ppe.api.render import write_video

        pairs = [
            (
                np.zeros((120, 160, 3), dtype=np.uint8),
                FrameResult(
                    frame_index=i,
                    timestamp_s=i / 10.0,
                    detections=[PERSON],
                    tracks=[],
                    assessments=[PersonAssessment(PERSON)],
                ),
            )
            for i in range(frames)
        ]
        out = tmp_path / "annotated.mp4"
        assert write_video(iter(pairs), out, 10.0, AppConfig()) == frames
        return out

    def test_the_container_declares_h264(self, tmp_path: Path) -> None:
        head = self._encode(tmp_path).read_bytes()[:8192]
        assert b"avc1" in head, "not H.264; browsers will not play this"
        assert b"mp4v" not in head, "fell back to mp4v, which browsers cannot decode"

    def test_it_is_actually_decodable(self, tmp_path: Path) -> None:
        """A declared codec is not a working file."""
        import av

        with av.open(str(self._encode(tmp_path))) as container:
            stream = container.streams.video[0]
            assert stream.codec_context.name == "h264"
            assert sum(1 for _ in container.decode(video=0)) > 0

    def test_the_index_is_at_the_front_for_streaming(self, tmp_path: Path) -> None:
        """+faststart, so playback can begin before the file finishes loading."""
        data = self._encode(tmp_path).read_bytes()
        assert data.index(b"moov") < data.index(b"mdat")

    def test_odd_dimensions_do_not_break_encoding(self, tmp_path: Path) -> None:
        """yuv420p needs even width and height; real frames are not always."""
        from ppe.api.render import write_video

        odd = np.zeros((121, 161, 3), dtype=np.uint8)
        pairs = [
            (
                odd,
                FrameResult(
                    frame_index=0,
                    timestamp_s=0.0,
                    detections=[],
                    tracks=[],
                    assessments=[],
                ),
            )
        ]
        assert write_video(iter(pairs), tmp_path / "odd.mp4", 10.0, AppConfig()) == 1
