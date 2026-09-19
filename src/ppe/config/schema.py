"""Typed configuration tree.

Every threshold in the system is declared here and nowhere else. If a magic
number appears in ``src/ppe`` outside this module, that is a bug: the whole
argument for a separate rule layer is that the business rule is tunable
without retraining, which only holds if the knobs are actually in config.

Values may be overridden by environment variables using the ``PPE_`` prefix
and ``__`` as the nesting delimiter, e.g. ``PPE_DEBOUNCE__CONFIRM_FRAMES=20``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from ppe.core.types import PPEClass


class _Base(BaseModel):
    """Reject unknown keys, so a typo in YAML fails loudly instead of silently."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Base):
    """Detector backend selection and inference thresholds."""

    backend: str = Field(default="ultralytics", description="'ultralytics' or 'onnx'.")
    weights: str = Field(default="models/yolov8n-ppe.pt")
    imgsz: int = Field(default=640, gt=0)
    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=300, gt=0)
    device: str = Field(default="auto", description="'auto', 'cpu', 'cuda', 'cuda:0'.")
    class_map: dict[int, PPEClass] = Field(
        default={0: PPEClass.PERSON, 1: PPEClass.HELMET, 2: PPEClass.VEST},
        description="Model class index -> our vocabulary. Owned by the dataset build.",
    )


class ItemRule(_Base):
    """Geometric rule for associating one PPE class with a person.

    The band is expressed as fractions of the person's own box height, which
    makes the rule scale-invariant: identical behaviour for a distant worker
    80px tall and a near one 800px tall.
    """

    band_top: float = Field(ge=0.0, le=1.0)
    band_bottom: float = Field(ge=0.0, le=1.0)
    min_containment: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of the item's area that must fall inside the person box.",
    )
    max_centre_offset: float = Field(
        default=0.6,
        ge=0.0,
        description=(
            "Max horizontal distance from the person's centreline, as a fraction of "
            "half the person's width. Stops a neighbour's helmet being borrowed."
        ),
    )
    min_confidence: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Items below this are ignored for association.",
    )

    # Once a candidate clears the gates above, these decide which person wins
    # it when several are plausible. They are weights rather than constants in
    # code so the ranking can be retuned for a site without a release.
    weight_containment: float = Field(default=0.6, ge=0.0, le=1.0)
    weight_band: float = Field(default=0.2, ge=0.0, le=1.0)
    weight_centre: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _band_ordered(self) -> ItemRule:
        if self.band_top >= self.band_bottom:
            msg = f"band_top ({self.band_top}) must be < band_bottom ({self.band_bottom})"
            raise ValueError(msg)
        total = self.weight_containment + self.weight_band + self.weight_centre
        if abs(total - 1.0) > 1e-6:
            msg = f"score weights must sum to 1.0, got {total}"
            raise ValueError(msg)
        return self


class AssociationConfig(_Base):
    """The rule layer's knobs."""

    required: list[PPEClass] = Field(
        default=[PPEClass.HELMET, PPEClass.VEST],
        description="Which items a person inside the zone must be wearing.",
    )
    rules: dict[PPEClass, ItemRule] = Field(
        default={
            PPEClass.HELMET: ItemRule(band_top=0.0, band_bottom=0.35),
            PPEClass.VEST: ItemRule(band_top=0.15, band_bottom=0.75, min_containment=0.6),
        }
    )
    min_person_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    min_person_height_px: float = Field(
        default=64.0,
        ge=0.0,
        description="Below this the person is too small to judge; skipped, not accused.",
    )
    edge_margin_px: float = Field(
        default=4.0,
        ge=0.0,
        description="A person box within this of a frame edge counts as cropped.",
    )
    skip_when_cropped: bool = Field(
        default=True,
        description=(
            "If the region a required item must occupy is outside the frame, record "
            "a skip reason instead of asserting a violation."
        ),
    )
    min_band_visibility: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of an item's expected band that must lie inside the frame "
            "before the person is judged for that item."
        ),
    )

    @model_validator(mode="after")
    def _rules_cover_required(self) -> AssociationConfig:
        missing = [item for item in self.required if item not in self.rules]
        if missing:
            msg = f"required items without an association rule: {missing}"
            raise ValueError(msg)
        if PPEClass.PERSON in self.required:
            msg = "'person' cannot be a required PPE item"
            raise ValueError(msg)
        return self


class TrackingConfig(_Base):
    """ByteTrack parameters. Names mirror ``supervision.ByteTrack``."""

    enabled: bool = True
    track_activation_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    lost_track_buffer: int = Field(default=30, ge=0)
    minimum_matching_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    frame_rate: int = Field(default=30, gt=0)


class DebounceConfig(_Base):
    """Turns per-frame assessments into events a human would agree with."""

    confirm_frames: int = Field(
        default=15,
        ge=1,
        description="Consecutive violating frames for one track before an event fires.",
    )
    cooldown_s: float = Field(
        default=60.0,
        ge=0.0,
        description="Minimum seconds before the same track can re-fire.",
    )
    max_gap_frames: int = Field(
        default=3,
        ge=0,
        description="Frames a track may go unseen without breaking its streak.",
    )
    forget_after_frames: int = Field(
        default=300,
        ge=1,
        description="Drop a track's debounce state after this long unseen.",
    )


class ZoneConfig(_Base):
    """A work area polygon. Only people inside an enabled zone are evaluated."""

    zone_id: str
    polygon: list[tuple[float, float]] = Field(min_length=3)
    enabled: bool = True

    @model_validator(mode="after")
    def _no_degenerate_polygon(self) -> ZoneConfig:
        if len({(round(x, 6), round(y, 6)) for x, y in self.polygon}) < 3:
            msg = f"zone '{self.zone_id}' needs at least 3 distinct vertices"
            raise ValueError(msg)
        return self


class ZonesConfig(_Base):
    """Zone set plus the policy for people outside every zone."""

    zones: list[ZoneConfig] = Field(default_factory=list)
    anchor: str = Field(
        default="bottom_centre",
        description="Point on the person box tested against the polygon.",
    )
    evaluate_when_no_zones: bool = Field(
        default=True,
        description="With no zones configured, evaluate the whole frame.",
    )


class PrivacyConfig(_Base):
    """PDPL posture: faces are blurred by default, not on request."""

    blur_faces: bool = True
    method: str = Field(default="gaussian", description="'gaussian' or 'pixelate'.")
    kernel_fraction: float = Field(
        default=0.12,
        gt=0.0,
        le=1.0,
        description="Blur kernel as a fraction of the blurred region's width.",
    )
    head_band_bottom: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description="Fraction of the person box treated as head when no face model is used.",
    )
    strict: bool = Field(
        default=True,
        description="If blurring fails, drop the frame rather than emit it unblurred.",
    )


class ApiConfig(_Base):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, gt=0, lt=65536)
    max_upload_mb: int = Field(default=512, gt=0)
    return_annotated_frames: bool = False


class AppConfig(BaseSettings):
    """Root configuration object, assembled from YAML then environment."""

    model_config = SettingsConfigDict(
        env_prefix="PPE_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    model: ModelConfig = Field(default_factory=ModelConfig)
    association: AssociationConfig = Field(default_factory=AssociationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    debounce: DebounceConfig = Field(default_factory=DebounceConfig)
    zones: ZonesConfig = Field(default_factory=ZonesConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    seed: int = Field(default=1337, description="Fixed everywhere for reproducibility.")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],  # noqa: ARG003 - pydantic's signature
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Give the environment the last word.

        Pydantic ranks constructor arguments highest by default, and the YAML
        file arrives as constructor arguments. That would make
        ``PPE_DEBOUNCE__CONFIRM_FRAMES`` silently useless whenever the key is
        also present in the file -- which is the only case anyone would use it.
        Reordering puts env above YAML above schema defaults, so a container
        can retune a deployment without editing a mounted file.
        """
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)
