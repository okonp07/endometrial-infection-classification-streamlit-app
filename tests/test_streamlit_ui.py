from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from endometrial_app.demo_bundle import build_demo_bundle_bytes, demo_bundle_filename
from endometrial_app.streamlit_ui import (
    _download_link_html,
    _hero_copy_html,
    _initial_inference_state,
    _probability_distribution_html,
    _visual_placeholder_panel_html,
)
from endometrial_app.ui import _load_training_summary


def test_streamlit_download_link_is_direct_download() -> None:
    project_root = Path(__file__).resolve().parents[1]
    bundle_bytes = build_demo_bundle_bytes(project_root)
    link_html = _download_link_html(bundle_bytes, demo_bundle_filename())

    assert "data:application/zip;base64," in link_html
    assert f'download="{demo_bundle_filename()}"' in link_html
    assert ">Download<" in link_html


def test_streamlit_demo_bundle_bytes_are_valid_zip() -> None:
    project_root = Path(__file__).resolve().parents[1]
    bundle_bytes = build_demo_bundle_bytes(project_root)

    with zipfile.ZipFile(BytesIO(bundle_bytes)) as archive:
        names = archive.namelist()
        assert "README.txt" in names
        image_members = [name for name in names if name.startswith("demo_samples/")]
        assert len(image_members) == 20
        assert all("infected_" not in name and "uninfected_" not in name for name in image_members)


def test_streamlit_probability_html_contains_expected_labels() -> None:
    html = _probability_distribution_html({"infected": 0.82, "uninfected": 0.18})

    assert "Class probabilities" in html
    assert "infected" in html
    assert "uninfected" in html
    assert "82%" in html
    assert "18%" in html


def test_streamlit_initial_state_contains_placeholders() -> None:
    state = _initial_inference_state()

    assert "Upload a scan to generate a prediction" in state["summary_html"]
    assert "Why the model predicted this" in state["explanation_html"]
    assert state["metadata"]["status"] == "Awaiting inference"
    assert state["model_input_image"] is None
    assert state["attention_heatmap_image"] is None
    assert state["probabilities"] == {}


def test_streamlit_hero_copy_matches_current_app_message() -> None:
    project_root = Path(__file__).resolve().parents[1]
    summary = _load_training_summary(project_root)
    hero_html = _hero_copy_html(summary)

    assert "AI-Assisted Endometrial Screening" in hero_html
    assert "TensorFlow inference pipeline" in hero_html
    assert "infected" in hero_html
    assert "uninfected" in hero_html


def test_streamlit_visual_placeholders_include_requested_watermarks() -> None:
    input_placeholder = _visual_placeholder_panel_html("Inference image")
    heatmap_placeholder = _visual_placeholder_panel_html("Attention heatmap")

    assert "Inference image" in input_placeholder
    assert "Attention heatmap" in heatmap_placeholder
    assert "visual-watermark" in input_placeholder
