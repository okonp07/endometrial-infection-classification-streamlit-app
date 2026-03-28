from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from endometrial_app.demo_bundle import build_demo_bundle_bytes, demo_bundle_filename
from endometrial_app.streamlit_ui import (
    _accuracy_interpretation_html,
    _download_link_html,
    _explanation_card_markdown,
    _friendly_metadata_summary,
    _hero_copy_html,
    _initial_inference_state,
    _metadata_panel_html,
    _ordered_probability_rows,
    _visual_placeholder_panel_html,
)
from endometrial_app.ui import AUTHOR_PROFILES, _load_training_summary


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


def test_streamlit_probability_rows_are_ordered_and_labeled() -> None:
    rows = _ordered_probability_rows({"infected": 0.82, "uninfected": 0.18})

    assert rows[0]["class_name"] == "infected"
    assert rows[0]["percentage"] == 82
    assert rows[1]["class_name"] == "uninfected"
    assert rows[1]["percentage"] == 18


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


def test_streamlit_metadata_placeholder_is_human_readable() -> None:
    metadata_html = _metadata_panel_html(_initial_inference_state()["metadata"])

    assert "Inference metadata will appear here" in metadata_html
    assert "Upload a scan and run the classifier" in metadata_html
    assert '"status"' not in metadata_html


def test_streamlit_metadata_summary_is_friendly_to_non_technical_users() -> None:
    summary = _friendly_metadata_summary(
        {
            "class_order": ["infected", "uninfected"],
            "input_size": [224, 224],
            "attention_layer": "input-gradient saliency",
            "focus_region": "middle left",
            "focus_pattern": "compact",
            "focus_coverage": 0.0138,
            "high_attention_threshold": 0.6419,
        }
    )

    assert summary["Prediction labels used"] == "Infected, Uninfected"
    assert summary["Scan size analyzed"] == "224 x 224 pixels"
    assert summary["Loaded model"] == "Current deployed production classifier"


def test_streamlit_explanation_card_uses_markdown_not_html_paragraph_tags() -> None:
    explanation_markdown = _explanation_card_markdown(
        {"predicted_label": "infected"},
        {
            "focus_region": "middle center",
            "focus_coverage": 0.026,
            "focus_pattern": "compact",
            "high_attention_threshold": 0.71,
            "margin": 0.9984,
            "runner_up_label": "uninfected",
            "attention_layer": "input-gradient saliency",
        },
        (224, 224),
    )

    assert "Why the model predicted this" in explanation_markdown
    assert "<p>" not in explanation_markdown
    assert "**224 x 224**" in explanation_markdown


def test_streamlit_accuracy_interpretation_adds_internal_evaluation_caution() -> None:
    history = pd.DataFrame(
        {
            "accuracy": [0.87, 0.99, 1.0],
            "val_accuracy": [1.0, 1.0, 1.0],
        }
    )

    note_html = _accuracy_interpretation_html(history)

    assert "does not show the classic divergence pattern" in note_html
    assert "internal result" in note_html
    assert "1.00%" not in note_html


def test_streamlit_author_role_uses_requested_two_line_title() -> None:
    prince = next(profile for profile in AUTHOR_PROFILES if profile["name"] == "Okon Prince")

    assert prince["role"] == "AI Engineer & Data Scientist |<br>Senior Data Scientist at MIVA Open University"
