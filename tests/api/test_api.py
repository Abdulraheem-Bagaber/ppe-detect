"""Tests for the HTTP service.

The detector is stubbed, so these exercise the transport, the error handling
and -- most importantly -- that the privacy posture of a running instance is
inspectable without reading its config file.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from ppe.api.app import create_app
from ppe.config.schema import AppConfig, ModelConfig, PrivacyConfig
from ppe.core.protocols import DetectorMetadata
from ppe.core.types import Detection, Frame


class StubDetector:
    """Stands in for a loaded model."""

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
    with TestClient(app) as test_client:
        app.state.detector = StubDetector()
        app.state.detector_error = None
        yield test_client


@pytest.fixture
def api_without_model() -> TestClient:
    app = create_app(AppConfig(model=ModelConfig(weights="does/not/exist.pt")))
    with TestClient(app) as test_client:
        yield test_client


class TestHealth:
    def test_reports_ok_when_a_model_is_loaded(self, api: TestClient) -> None:
        body = api.get("/health").json()
        assert body["status"] == "ok"
        assert body["device"] == "cpu"
        assert body["weights_sha256"].startswith("aaaa")

    def test_reports_degraded_rather_than_dying_without_a_model(
        self, api_without_model: TestClient
    ) -> None:
        """A missing model must not stop the service coming up to say so."""
        response = api_without_model.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert "not found" in (body["detail"] or "")

    def test_it_reports_the_privacy_posture(self, api: TestClient) -> None:
        """An operator must be able to verify blurring without reading a file."""
        assert api.get("/health").json()["blur_faces"] is True

    def test_it_reports_blurring_being_off(self) -> None:
        config = AppConfig(privacy=PrivacyConfig(blur_faces=False))
        app = create_app(config)
        with TestClient(app) as c:
            app.state.detector = StubDetector()
            assert c.get("/health").json()["blur_faces"] is False


class TestConfigEndpoint:
    def test_the_thresholds_in_force_are_readable(self, api: TestClient) -> None:
        """A policy you cannot read from the running service is not inspectable."""
        body = api.get("/config").json()
        assert body["association"]["required"] == ["helmet", "vest"]
        assert body["debounce"]["confirm_frames"] == 15

    def test_it_exposes_the_association_bands(self, api: TestClient) -> None:
        rules = api.get("/config").json()["association"]["rules"]
        assert rules["helmet"]["band_top"] == 0.0
        assert rules["helmet"]["band_bottom"] == 0.35


class TestAnalyseUpload:
    def test_a_bad_video_is_a_client_error_not_a_crash(self, api: TestClient) -> None:
        response = api.post(
            "/analyse/upload",
            files={"file": ("clip.mp4", io.BytesIO(b"not a video"), "video/mp4")},
        )
        assert response.status_code == 400

    def test_an_oversized_upload_is_rejected(self) -> None:
        from ppe.config.schema import ApiConfig

        app = create_app(AppConfig(api=ApiConfig(max_upload_mb=1)))
        with TestClient(app) as c:
            app.state.detector = StubDetector()
            big = io.BytesIO(b"\0" * (2 * 1024 * 1024))
            response = c.post("/analyse/upload", files={"file": ("big.mp4", big, "video/mp4")})
        assert response.status_code == 413

    def test_it_refuses_to_work_without_a_model(self, api_without_model: TestClient) -> None:
        response = api_without_model.post(
            "/analyse/upload",
            files={"file": ("clip.mp4", io.BytesIO(b"x"), "video/mp4")},
        )
        assert response.status_code == 503


class TestAnalyseStream:
    def test_a_non_rtsp_url_is_rejected(self, api: TestClient) -> None:
        response = api.post("/analyse/stream", json={"rtsp_url": "http://example.invalid/x"})
        assert response.status_code == 400

    @pytest.mark.slow
    def test_an_unreachable_stream_is_a_bad_gateway(self, api: TestClient) -> None:
        """Slow: OpenCV's FFmpeg backend waits 30s before giving up on a stream."""
        response = api.post(
            "/analyse/stream",
            json={"rtsp_url": "rtsp://127.0.0.1:1/none", "max_frames": 1},
        )
        assert response.status_code == 502

    def test_max_frames_must_be_positive(self, api: TestClient) -> None:
        response = api.post("/analyse/stream", json={"rtsp_url": "rtsp://x/y", "max_frames": 0})
        assert response.status_code == 422


class TestOpenApi:
    def test_the_schema_documents_the_event_contract(self, api: TestClient) -> None:
        schema = api.get("/openapi.json").json()
        event = schema["components"]["schemas"]["ViolationEventModel"]["properties"]
        assert set(event) >= {"event_id", "track_id", "missing", "timestamp_s", "person_box"}
