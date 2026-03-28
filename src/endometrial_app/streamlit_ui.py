from __future__ import annotations

import base64
import html
import json
import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Any

import altair as alt
import streamlit as st
from PIL import Image

from endometrial_app.demo_bundle import build_demo_bundle_bytes, demo_bundle_filename
from endometrial_app.feedback import save_feedback
from endometrial_app.service import PredictionService
from endometrial_app import ui as gradio_ui

STREAMLIT_REPO_URL = "https://github.com/okonp07/endometrial-infection-classification-streamlit-app"
PRODUCTION_REPO_URL = "https://github.com/okonp07/endometrial-infection-classification-app"


STREAMLIT_CSS = """
:root {
    --brand-blue: #0e4d73;
    --brand-blue-deep: #092d46;
    --brand-green: #178b76;
    --brand-green-soft: #dff4ee;
    --brand-ash: #eef3f4;
    --brand-ink: #12242d;
    --brand-slate: #4d6069;
    --brand-white: #ffffff;
    --hero-panel-height: clamp(38rem, 43vw, 47rem);
}

.stApp {
    background:
        radial-gradient(circle at top right, rgba(23, 139, 118, 0.18), transparent 32%),
        radial-gradient(circle at top left, rgba(14, 77, 115, 0.12), transparent 24%),
        linear-gradient(180deg, #f8fbfb 0%, #edf2f3 100%);
    color: var(--brand-ink);
}

[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu,
footer {
    visibility: hidden;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1440px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(9, 45, 70, 0.08);
    border-radius: 28px;
    box-shadow: 0 20px 56px rgba(18, 36, 45, 0.08);
    padding: 1.2rem 1.2rem 1.1rem;
    backdrop-filter: blur(16px);
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0.35rem;
    background: linear-gradient(180deg, rgba(238, 243, 244, 0.94) 0%, rgba(229, 237, 239, 0.9) 100%);
    border: 1px solid rgba(9, 45, 70, 0.08);
    border-radius: 22px;
    padding: 0.4rem;
    box-shadow: 0 12px 32px rgba(18, 36, 45, 0.08);
    width: 100%;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    height: 44px;
    padding: 0.7rem 1rem;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.28);
    color: var(--brand-blue-deep);
    font-weight: 800 !important;
    border: 1px solid rgba(9, 45, 70, 0.04);
    transition: background 160ms ease, color 160ms ease, box-shadow 160ms ease;
}

[data-testid="stTabs"] [data-baseweb="tab"] *,
[data-testid="stTabs"] [data-baseweb="tab"] p,
[data-testid="stTabs"] [data-baseweb="tab"] span,
[data-testid="stTabs"] button[role="tab"] *,
[data-testid="stTabs"] button[role="tab"] span {
    color: var(--brand-blue-deep) !important;
    -webkit-text-fill-color: var(--brand-blue-deep) !important;
    font-weight: 800 !important;
}

[data-testid="stTabs"] [data-baseweb="tab"]:hover,
[data-testid="stTabs"] [data-baseweb="tab"]:focus-visible,
[data-testid="stTabs"] [data-baseweb="tab"]:active {
    background: rgba(255, 255, 255, 0.46);
    box-shadow: 0 10px 24px rgba(18, 36, 45, 0.1);
}

[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, var(--brand-blue-deep), var(--brand-green)) !important;
    color: var(--brand-white) !important;
    font-weight: 800 !important;
}

[data-testid="stTabs"] [aria-selected="true"] *,
[data-testid="stTabs"] [aria-selected="true"] p,
[data-testid="stTabs"] [aria-selected="true"] span,
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] *,
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] span {
    color: var(--brand-white) !important;
    -webkit-text-fill-color: var(--brand-white) !important;
    font-weight: 800 !important;
}

[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    padding-top: 1.2rem;
}

[data-testid="stHorizontalBlock"] {
    align-items: stretch;
}

h1, h2, h3 {
    color: var(--brand-blue-deep);
    font-family: "Space Grotesk", "Manrope", "Avenir Next", "Segoe UI", sans-serif;
}

p, li, label, span {
    color: var(--brand-slate);
    line-height: 1.75;
}

a {
    display: inline-block;
    color: var(--brand-blue);
    font-size: 1.03rem;
    font-weight: 800;
    padding: 0.08rem 0.34rem;
    border-radius: 0.5rem;
    text-decoration-thickness: 2px;
    text-underline-offset: 0.16rem;
    transition: background 160ms ease, color 160ms ease, box-shadow 160ms ease;
}

a:hover,
a:focus,
a:active {
    color: var(--brand-white) !important;
    background: linear-gradient(135deg, var(--brand-blue-deep), var(--brand-green));
    border-radius: 0.5rem;
    box-shadow: 0 10px 20px rgba(18, 36, 45, 0.12);
    outline: none;
    text-decoration: none;
}

code {
    background: rgba(14, 77, 115, 0.08);
    border-radius: 0.45rem;
    padding: 0.08rem 0.32rem;
}

.section-intro {
    margin-bottom: 1rem;
    padding: 1.2rem 1.3rem;
    border-radius: 24px;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.95) 0%, rgba(245, 250, 251, 0.92) 100%);
    border: 1px solid rgba(9, 45, 70, 0.08);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.78);
}

.section-intro h2 {
    margin: 0.2rem 0 0.55rem;
    font-size: 1.9rem;
    line-height: 1.08;
}

.section-intro p {
    margin: 0;
    color: var(--brand-slate);
    line-height: 1.72;
    max-width: 72ch;
}

.section-intro.compact {
    padding: 1rem 1.1rem;
}

.status-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
    margin: 0.1rem 0 1.1rem;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.52rem 0.88rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid rgba(9, 45, 70, 0.08);
    color: var(--brand-blue-deep);
    font-size: 0.92rem;
    font-weight: 700;
    box-shadow: 0 10px 24px rgba(18, 36, 45, 0.06);
}

.status-pill .dot {
    width: 0.6rem;
    height: 0.6rem;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--brand-blue), var(--brand-green));
}

.hero-copy {
    background: linear-gradient(135deg, rgba(9, 45, 70, 0.98) 0%, rgba(14, 77, 115, 0.95) 55%, rgba(23, 139, 118, 0.92) 100%);
    color: var(--brand-white);
    border: 1px solid rgba(9, 45, 70, 0.08);
    border-radius: 28px;
    box-shadow: 0 20px 56px rgba(18, 36, 45, 0.08);
    padding: 1.55rem;
    height: var(--hero-panel-height);
    display: flex;
    flex-direction: column;
}

.hero-copy h1 {
    color: var(--brand-white) !important;
    -webkit-text-fill-color: var(--brand-white);
    font-size: 3rem;
    line-height: 1.02;
    margin: 0.15rem 0 1rem;
    font-weight: 800;
    text-shadow: 0 10px 22px rgba(9, 45, 70, 0.18);
}

.hero-copy p {
    color: rgba(255, 255, 255, 0.92);
    font-size: 1.03rem;
    line-height: 1.75;
}

.hero-copy strong {
    color: var(--brand-white);
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 0.45rem;
    padding: 0.04rem 0.38rem;
    font-weight: 800;
}

.hero-eyebrow {
    display: inline-block;
    margin-bottom: 0.85rem;
    padding: 0.42rem 0.8rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: var(--brand-white);
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.hero-stat-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
    margin-top: 1.4rem;
}

.hero-stat {
    padding: 1rem;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.16);
    backdrop-filter: blur(10px);
}

.hero-stat-value {
    display: block;
    color: var(--brand-white);
    font-family: "Space Grotesk", "Manrope", sans-serif;
    font-size: 1.65rem;
    font-weight: 700;
}

.hero-stat-label {
    display: block;
    margin-top: 0.35rem;
    color: rgba(255, 255, 255, 0.84);
    font-size: 0.92rem;
}

.hero-banner-wrap {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.72) 0%, rgba(223, 244, 238, 0.72) 100%);
    border: 1px solid rgba(9, 45, 70, 0.08);
    border-radius: 28px;
    box-shadow: 0 20px 56px rgba(18, 36, 45, 0.08);
    padding: 0.95rem;
    height: var(--hero-panel-height);
    display: flex;
}

.hero-banner-wrap img {
    width: 100%;
    height: 100%;
    border-radius: 22px;
    object-fit: cover;
    box-shadow: 0 16px 34px rgba(18, 36, 45, 0.16);
    display: block;
    flex: 1 1 auto;
}

.hero-shell {
    margin-top: 0.15rem;
    margin-bottom: 1.35rem;
    height: 100%;
}

.content-card-head h3,
.content-card-head h2 {
    margin: 0.2rem 0 0.45rem;
}

.content-card-head p {
    margin: 0;
    color: var(--brand-slate);
}

.section-kicker,
.prediction-kicker {
    display: inline-block;
    margin-bottom: 0.55rem;
    color: var(--brand-green);
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.helper-copy p,
.helper-copy li,
.sample-copy p,
.sample-copy li,
.about-copy p,
.about-copy li {
    color: var(--brand-slate);
    line-height: 1.75;
}

.inline-download-link {
    color: var(--brand-blue);
    font-size: 1.05rem;
    font-weight: 800;
    text-decoration: underline;
}

.prediction-shell {
    padding: 1.2rem;
    border-radius: 22px;
    background: linear-gradient(180deg, #f7fafb 0%, #edf5f3 100%);
    border: 1px solid rgba(9, 45, 70, 0.08);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.82);
}

.prediction-shell.placeholder {
    background: linear-gradient(180deg, #f8fbfb 0%, #f1f5f6 100%);
}

.prediction-title {
    margin-top: 0.35rem;
    color: var(--brand-blue-deep);
    font-family: "Space Grotesk", "Manrope", sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
}

.prediction-chip {
    display: inline-flex;
    margin-top: 0.9rem;
    padding: 0.48rem 0.88rem;
    border-radius: 999px;
    font-weight: 700;
}

.prediction-chip.infected {
    background: rgba(14, 77, 115, 0.12);
    color: var(--brand-blue-deep);
}

.prediction-chip.uninfected {
    background: rgba(23, 139, 118, 0.12);
    color: #0e6c5b;
}

.prediction-shell p,
.explanation-shell p,
.download-card p,
.download-card li,
.feedback-card-copy p,
.feedback-card-copy li,
.eda-copy p,
.eda-copy li,
.author-card-copy p {
    color: var(--brand-slate);
    line-height: 1.7;
}

.probability-shell {
    margin-top: 1rem;
    padding: 1rem 1.05rem;
    border-radius: 20px;
    background: linear-gradient(180deg, rgba(247, 250, 251, 0.98) 0%, rgba(237, 245, 243, 0.92) 100%);
    border: 1px solid rgba(9, 45, 70, 0.08);
}

.probability-title {
    margin-bottom: 0.65rem;
    color: var(--brand-blue-deep);
    font-family: "Space Grotesk", "Manrope", sans-serif;
    font-size: 1.02rem;
    font-weight: 700;
}

.probability-placeholder {
    color: var(--brand-slate);
    margin: 0;
}

.probability-native {
    margin-top: 1rem;
    padding: 1rem 1.05rem;
    border-radius: 20px;
    background: linear-gradient(180deg, rgba(247, 250, 251, 0.98) 0%, rgba(237, 245, 243, 0.92) 100%);
    border: 1px solid rgba(9, 45, 70, 0.08);
}

.probability-native h3 {
    margin: 0 0 0.25rem;
    color: var(--brand-blue-deep);
    font-family: "Space Grotesk", "Manrope", sans-serif;
    font-size: 1.02rem;
    font-weight: 700;
}

.metadata-friendly {
    margin-top: 0.45rem;
    padding: 1rem 1.05rem;
    border-radius: 22px;
    background: linear-gradient(180deg, rgba(247, 250, 251, 0.98) 0%, rgba(237, 245, 243, 0.92) 100%);
    border: 1px solid rgba(9, 45, 70, 0.08);
}

.metadata-friendly h3 {
    margin: 0 0 0.6rem;
    color: var(--brand-blue-deep);
    font-family: "Space Grotesk", "Manrope", sans-serif;
    font-size: 1.02rem;
    font-weight: 700;
}

.metadata-friendly p {
    margin: 0 0 0.5rem;
}

.prob-row {
    margin-bottom: 0.85rem;
}

.prob-row:last-child {
    margin-bottom: 0;
}

.prob-row-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 0.32rem;
    color: var(--brand-blue-deep);
    font-weight: 700;
}

.prob-track {
    width: 100%;
    height: 0.75rem;
    border-radius: 999px;
    background: rgba(14, 77, 115, 0.08);
    overflow: hidden;
}

.prob-fill {
    height: 100%;
    border-radius: 999px;
}

.explanation-shell {
    padding: 0.15rem 0 0.4rem;
}

.explanation-title {
    color: var(--brand-blue-deep);
    font-family: "Space Grotesk", "Manrope", sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
}

.placeholder-guide {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.8rem;
    margin-top: 1rem;
}

.placeholder-guide-card {
    padding: 0.95rem;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(247, 250, 251, 0.98) 0%, rgba(237, 245, 243, 0.92) 100%);
    border: 1px solid rgba(9, 45, 70, 0.08);
}

.placeholder-guide-step {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--brand-blue-deep), var(--brand-green));
    color: var(--brand-white);
    font-size: 0.88rem;
    font-weight: 800;
    margin-bottom: 0.65rem;
}

.visual-label {
    display: inline-flex;
    margin-bottom: 0.7rem;
    padding: 0.42rem 0.76rem;
    border-radius: 14px;
    background: rgba(14, 77, 115, 0.08);
    color: var(--brand-blue);
    font-weight: 800;
    font-size: 0.98rem;
}

.visual-shell {
    position: relative;
    min-height: 360px;
    border-radius: 24px;
    border: 1px solid rgba(9, 45, 70, 0.08);
    background: linear-gradient(180deg, rgba(247, 250, 251, 0.98) 0%, rgba(237, 245, 243, 0.92) 100%);
    overflow: hidden;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.82);
}

.visual-shell img {
    width: 100%;
    height: 100%;
    min-height: 360px;
    object-fit: contain;
    display: block;
    background: linear-gradient(180deg, rgba(247, 250, 251, 0.98) 0%, rgba(237, 245, 243, 0.92) 100%);
}

.visual-shell.placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    background:
        linear-gradient(180deg, rgba(247, 250, 251, 0.98) 0%, rgba(237, 245, 243, 0.92) 100%),
        repeating-linear-gradient(
            135deg,
            rgba(14, 77, 115, 0.018),
            rgba(14, 77, 115, 0.018) 16px,
            rgba(23, 139, 118, 0.028) 16px,
            rgba(23, 139, 118, 0.028) 32px
        );
}

.visual-watermark {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: rgba(14, 77, 115, 0.12);
    font-family: "Space Grotesk", "Manrope", sans-serif;
    font-size: clamp(2rem, 4vw, 3.3rem);
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    transform: rotate(-26deg);
    pointer-events: none;
    user-select: none;
    white-space: nowrap;
}

.preview-shell {
    min-height: 360px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 24px;
    border: 1px dashed rgba(9, 45, 70, 0.15);
    background: linear-gradient(180deg, rgba(247, 250, 251, 0.98) 0%, rgba(237, 245, 243, 0.92) 100%);
    color: var(--brand-slate);
    text-align: center;
    padding: 1.2rem;
}

.preview-shell strong {
    display: block;
    color: var(--brand-blue-deep);
    margin-bottom: 0.2rem;
}

div[data-testid="stFileUploader"] section {
    border-radius: 20px;
    border: 1px dashed rgba(9, 45, 70, 0.16);
    background: linear-gradient(180deg, #f7fafb 0%, #edf5f3 100%);
}

div[data-testid="stFileUploaderDropzoneInstructions"] span,
div[data-testid="stFileUploaderDropzone"] small {
    color: var(--brand-slate) !important;
}

div[data-testid="stImage"] img {
    border-radius: 24px;
    border: 1px solid rgba(9, 45, 70, 0.08);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
}

div.stButton > button,
div.stDownloadButton > button,
div[data-testid="stFormSubmitButton"] > button {
    border-radius: 16px;
    color: var(--brand-blue-deep);
    font-weight: 800 !important;
    min-height: 3rem;
    border: 1px solid rgba(9, 45, 70, 0.08);
    box-shadow: 0 12px 26px rgba(18, 36, 45, 0.08);
    transition: transform 160ms ease, background 160ms ease, color 160ms ease, box-shadow 160ms ease;
}

div.stButton > button *,
div.stButton > button span,
div.stButton > button p,
div.stDownloadButton > button *,
div.stDownloadButton > button span,
div.stDownloadButton > button p,
div[data-testid="stFormSubmitButton"] > button *,
div[data-testid="stFormSubmitButton"] > button span,
div[data-testid="stFormSubmitButton"] > button p {
    color: inherit !important;
    -webkit-text-fill-color: currentColor !important;
    font-weight: 800 !important;
}

div.stButton > button[kind="primary"],
div.stDownloadButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, var(--brand-blue-deep), var(--brand-green));
    color: var(--brand-white) !important;
    border: none;
}

div.stButton > button[kind="primary"] *,
div.stButton > button[kind="primary"] span,
div.stButton > button[kind="primary"] p,
div.stDownloadButton > button[kind="primary"] *,
div.stDownloadButton > button[kind="primary"] span,
div.stDownloadButton > button[kind="primary"] p,
div[data-testid="stFormSubmitButton"] > button[kind="primary"] *,
div[data-testid="stFormSubmitButton"] > button[kind="primary"] span,
div[data-testid="stFormSubmitButton"] > button[kind="primary"] p {
    color: var(--brand-white) !important;
    -webkit-text-fill-color: var(--brand-white) !important;
}

div.stButton > button:hover,
div.stButton > button:focus-visible,
div.stButton > button:active,
div.stDownloadButton > button:hover,
div.stDownloadButton > button:focus-visible,
div.stDownloadButton > button:active,
div[data-testid="stFormSubmitButton"] > button:hover,
div[data-testid="stFormSubmitButton"] > button:focus-visible,
div[data-testid="stFormSubmitButton"] > button:active {
    transform: translateY(-1px);
    box-shadow: 0 14px 30px rgba(18, 36, 45, 0.12);
}

div.stButton > button:not([kind="primary"]):hover,
div.stButton > button:not([kind="primary"]):focus-visible,
div.stButton > button:not([kind="primary"]):active,
div.stDownloadButton > button:not([kind="primary"]):hover,
div.stDownloadButton > button:not([kind="primary"]):focus-visible,
div.stDownloadButton > button:not([kind="primary"]):active,
div[data-testid="stFormSubmitButton"] > button:not([kind="primary"]):hover,
div[data-testid="stFormSubmitButton"] > button:not([kind="primary"]):focus-visible,
div[data-testid="stFormSubmitButton"] > button:not([kind="primary"]):active {
    background: linear-gradient(135deg, var(--brand-blue-deep), var(--brand-green));
    color: var(--brand-white) !important;
    border-color: transparent;
}

div.stButton > button:not([kind="primary"]):hover *,
div.stButton > button:not([kind="primary"]):focus-visible *,
div.stButton > button:not([kind="primary"]):active *,
div.stDownloadButton > button:not([kind="primary"]):hover *,
div.stDownloadButton > button:not([kind="primary"]):focus-visible *,
div.stDownloadButton > button:not([kind="primary"]):active *,
div[data-testid="stFormSubmitButton"] > button:not([kind="primary"]):hover *,
div[data-testid="stFormSubmitButton"] > button:not([kind="primary"]):focus-visible *,
div[data-testid="stFormSubmitButton"] > button:not([kind="primary"]):active * {
    color: var(--brand-white) !important;
    -webkit-text-fill-color: var(--brand-white) !important;
}

.download-highlight,
.feedback-note,
.eda-note {
    padding: 1rem 1.1rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(14, 77, 115, 0.08), rgba(23, 139, 118, 0.12));
    border: 1px solid rgba(14, 77, 115, 0.08);
}

.download-highlight p,
.feedback-note p,
.eda-note p {
    margin: 0;
    color: var(--brand-blue-deep);
    line-height: 1.7;
}

.accuracy-note {
    margin-top: 0.9rem;
}

.feedback-status {
    padding: 1.1rem 1.15rem;
    border-radius: 22px;
    background: linear-gradient(180deg, rgba(247, 250, 251, 0.98) 0%, rgba(237, 245, 243, 0.92) 100%);
    border: 1px solid rgba(9, 45, 70, 0.08);
}

.feedback-status h3 {
    margin: 0 0 0.55rem;
    color: var(--brand-blue-deep);
    font-family: "Space Grotesk", "Manrope", sans-serif;
}

.metadata-shell {
    margin-top: 0.45rem;
    padding: 1rem 1.05rem;
    border-radius: 22px;
    background: linear-gradient(180deg, rgba(247, 250, 251, 0.98) 0%, rgba(237, 245, 243, 0.92) 100%);
    border: 1px solid rgba(9, 45, 70, 0.08);
}

.metadata-shell pre,
.metadata-friendly pre {
    margin: 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    color: var(--brand-ink);
    font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
    font-size: 0.88rem;
    line-height: 1.65;
}

.metadata-friendly details {
    margin-top: 0.85rem;
}

.metadata-friendly summary {
    cursor: pointer;
    color: var(--brand-blue-deep);
    font-weight: 700;
    margin-bottom: 0.6rem;
}

.metadata-empty {
    display: flex;
    min-height: 170px;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: var(--brand-slate);
    padding: 1.1rem 1rem;
}

.metadata-empty strong {
    display: block;
    margin-bottom: 0.3rem;
    color: var(--brand-blue-deep);
    font-size: 1rem;
}

.author-photo-shell {
    width: min(220px, 100%);
    aspect-ratio: 1 / 1;
    margin: 0 auto 1rem;
    border-radius: 50%;
    overflow: hidden;
}

.author-photo-shell img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 50%;
    border: 4px solid rgba(23, 139, 118, 0.14);
    box-shadow: 0 18px 46px rgba(18, 36, 45, 0.16);
}

.author-role {
    color: var(--brand-green);
    font-weight: 700;
    margin-bottom: 0.7rem;
    line-height: 1.55;
}

.footer-note {
    margin-top: 1.4rem;
    padding: 1.15rem 1.4rem 1.8rem;
    text-align: center;
    color: var(--brand-slate);
}

.footer-note p {
    margin: 0.2rem 0;
    line-height: 1.7;
}

@media (max-width: 960px) {
    :root {
        --hero-panel-height: auto;
    }

    .section-intro h2 {
        font-size: 1.6rem;
    }

    .hero-copy h1 {
        font-size: 2.15rem;
        line-height: 1.06;
    }

    .hero-copy p,
    .helper-copy p,
    .sample-copy p,
    .about-copy p,
    .about-copy li,
    .prediction-shell p,
    .explanation-shell p,
    .footer-note p {
        font-size: 0.97rem;
        line-height: 1.65;
    }

    .hero-stat-grid,
    .placeholder-guide {
        grid-template-columns: 1fr;
    }

    .status-strip {
        gap: 0.55rem;
    }

    .preview-shell {
        min-height: 300px;
    }

    .hero-copy,
    .hero-banner-wrap {
        height: auto;
        min-height: 31rem;
    }
}

@media (max-width: 640px) {
    .block-container {
        padding-left: 0.7rem;
        padding-right: 0.7rem;
    }

    .hero-copy,
    .hero-banner-wrap {
        border-radius: 22px;
        padding: 1rem;
    }

    .hero-copy h1 {
        font-size: 1.85rem;
    }

    .author-photo-shell {
        width: min(180px, 100%);
    }

    .section-intro,
    .section-intro.compact {
        padding: 0.95rem 1rem;
    }
}
"""


def _inject_css() -> None:
    st.markdown(f"<style>{STREAMLIT_CSS}</style>", unsafe_allow_html=True)


def _mime_for_path(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def _data_uri_from_bytes(raw_bytes: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(raw_bytes).decode('utf-8')}"


def _data_uri_from_path(path: Path) -> str:
    return _data_uri_from_bytes(path.read_bytes(), _mime_for_path(path))


def _data_uri_from_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return _data_uri_from_bytes(buffer.getvalue(), "image/png")


@st.cache_resource(show_spinner=False)
def _get_service() -> PredictionService:
    return PredictionService.from_settings()


@st.cache_data(show_spinner=False)
def _get_demo_bundle_bytes(project_root: str) -> bytes:
    return build_demo_bundle_bytes(Path(project_root))


def _initial_inference_state() -> dict[str, Any]:
    return {
        "summary_html": gradio_ui._prediction_placeholder_html(),
        "probabilities": {},
        "explanation_html": _explanation_placeholder_markdown(),
        "model_input_image": None,
        "attention_heatmap_image": None,
        "metadata": gradio_ui._metadata_placeholder(),
    }


def _ensure_state() -> None:
    if "streamlit_inference_state" not in st.session_state:
        st.session_state.streamlit_inference_state = _initial_inference_state()
    if "streamlit_upload_nonce" not in st.session_state:
        st.session_state.streamlit_upload_nonce = 0
    if "streamlit_uploaded_image_bytes" not in st.session_state:
        st.session_state.streamlit_uploaded_image_bytes = None
    if "streamlit_feedback_status_html" not in st.session_state:
        st.session_state.streamlit_feedback_status_html = gradio_ui._feedback_placeholder_html()
    if "streamlit_feedback_form_nonce" not in st.session_state:
        st.session_state.streamlit_feedback_form_nonce = 0


def _reset_inference_state() -> None:
    st.session_state.streamlit_inference_state = _initial_inference_state()
    st.session_state.streamlit_uploaded_image_bytes = None
    st.session_state.streamlit_upload_nonce += 1


def _uploaded_image() -> Image.Image | None:
    raw_bytes = st.session_state.get("streamlit_uploaded_image_bytes")
    if not raw_bytes:
        return None
    return Image.open(BytesIO(raw_bytes)).convert("RGB")


def _store_uploaded_image(uploaded_file: Any) -> None:
    if uploaded_file is None:
        return
    st.session_state.streamlit_uploaded_image_bytes = uploaded_file.getvalue()


def _download_link_html(bundle_bytes: bytes, filename: str) -> str:
    download_uri = _data_uri_from_bytes(bundle_bytes, "application/zip")
    return (
        f'<a class="inline-download-link" href="{download_uri}" download="{html.escape(filename)}">'
        "Download</a>"
    )


def _hero_stats_html(summary: dict[str, Any]) -> str:
    clean_counts = summary.get("clean_counts", {})
    split_counts = summary.get("split_counts", {})
    total_clean = sum(int(count) for count in clean_counts.values())
    test_total = sum(int(count) for count in split_counts.get("test", {}).values())
    cards = [
        ("2", "Target classes"),
        (f"{total_clean:,}" if total_clean else "N/A", "Deduplicated scans"),
        (f"{test_total:,}" if test_total else "N/A", "Held-out test images"),
    ]
    stats_markup = "".join(
        (
            '<div class="hero-stat">'
            f'<span class="hero-stat-value">{html.escape(value)}</span>'
            f'<span class="hero-stat-label">{html.escape(label)}</span>'
            "</div>"
        )
        for value, label in cards
    )
    return f'<div class="hero-stat-grid">{stats_markup}</div>'


def _hero_copy_html(summary: dict[str, Any]) -> str:
    return f"""
    <div class="hero-copy">
        <span class="hero-eyebrow">AI-Assisted Endometrial Screening</span>
        <h1>Endometrial Infection Classification App</h1>
        <p>
            This application helps classify endometrial scan images into <strong>infected</strong> and
            <strong>uninfected</strong> classes through a production-ready TensorFlow inference pipeline.
            It combines a modern web interface, a reusable FastAPI backend, and a curated medical-image
            workflow so the model can move beyond the notebook into a real deployment setting.
        </p>
        {_hero_stats_html(summary)}
    </div>
    """


def _hero_banner_html(banner_path: Path) -> str:
    return f"""
    <div class="hero-banner-wrap">
        <img src="{_data_uri_from_path(banner_path)}" alt="Endometrial classifier banner" />
    </div>
    """


def _section_intro_html(kicker: str, title: str, body: str, *, compact: bool = False) -> str:
    compact_class = " compact" if compact else ""
    return f"""
    <div class="section-intro{compact_class}">
        <span class="section-kicker">{html.escape(kicker)}</span>
        <h2>{html.escape(title)}</h2>
        <p>{body}</p>
    </div>
    """


def _status_strip_html(*items: str) -> str:
    pills = "".join(
        f'<div class="status-pill"><span class="dot"></span>{html.escape(item)}</div>'
        for item in items
    )
    return f'<div class="status-strip">{pills}</div>'


def _upload_intro_html(download_link_html: str) -> str:
    return f"""
    <div class="helper-copy">
        <span class="section-kicker">Step 1</span>
        <h2>Upload an image</h2>
        <p>
            Add an endometrial scan and send it through the classifier. If you do not have a scan available,
            use the {download_link_html} link to grab the bundled test image pack.
        </p>
    </div>
    """


def _result_intro_html() -> str:
    return """
    <div class="helper-copy">
        <span class="section-kicker">Step 2</span>
        <h2>Review the result</h2>
        <p>
            The app returns the predicted class, the model confidence, the class-probability distribution,
            and the inference metadata used for the prediction request.
        </p>
    </div>
    """


def _ordered_probability_rows(probabilities: dict[str, float]) -> list[dict[str, Any]]:
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    color_map = {"infected": "#0e4d73", "uninfected": "#178b76"}
    rows: list[dict[str, Any]] = []
    for class_name, probability in ordered:
        value = float(probability)
        rows.append(
            {
                "class_name": class_name,
                "label": class_name.replace("_", " ").title(),
                "probability": value,
                "percentage": int(round(value * 100)),
                "color": color_map.get(class_name, "#0e4d73"),
            }
        )
    return rows


def _preview_placeholder_html() -> str:
    return """
    <div class="preview-shell">
        <div>
            <strong>No scan uploaded yet</strong>
            Add an image above and the preview will appear here.
        </div>
    </div>
    """


def _visual_placeholder_panel_html(watermark: str) -> str:
    return f"""
    <div class="visual-shell placeholder" aria-label="{html.escape(watermark)} placeholder">
        <div class="visual-watermark">{html.escape(watermark)}</div>
    </div>
    """


def _visual_image_panel_html(image: Image.Image, alt: str) -> str:
    return f"""
    <div class="visual-shell">
        <img src="{_data_uri_from_image(image)}" alt="{html.escape(alt)}" />
    </div>
    """


def _visual_label_html(label: str) -> str:
    return f'<div class="visual-label">{html.escape(label)}</div>'


def _build_probability_metadata(
    result: dict[str, Any],
    explanation: dict[str, Any],
    service: PredictionService,
) -> dict[str, Any]:
    return {
        "predicted_index": result["predicted_index"],
        "class_order": service.settings.class_names,
        "model_path": str(service.settings.model_path),
        "input_size": list(service.settings.image_size),
        "attention_layer": explanation.get("attention_layer", "unavailable"),
        "focus_region": explanation.get("focus_region", "unavailable"),
        "focus_pattern": explanation.get("focus_pattern", "unavailable"),
        "focus_coverage": round(float(explanation.get("focus_coverage", 0.0)), 4),
        "high_attention_threshold": round(float(explanation.get("high_attention_threshold", 0.0)), 4),
    }


def _friendly_metadata_summary(metadata: dict[str, Any]) -> dict[str, str]:
    class_order = metadata.get("class_order", [])
    input_size = metadata.get("input_size", [])
    focus_coverage = float(metadata.get("focus_coverage", 0.0)) * 100
    threshold = float(metadata.get("high_attention_threshold", 0.0))
    return {
        "Prediction labels used": ", ".join(str(item).title() for item in class_order) or "Unavailable",
        "Scan size analyzed": " x ".join(str(item) for item in input_size) + " pixels" if input_size else "Unavailable",
        "Explanation method": str(metadata.get("attention_layer", "Unavailable")).replace("_", " "),
        "Main focus region": str(metadata.get("focus_region", "Unavailable")).replace("_", " ").title(),
        "Attention pattern": str(metadata.get("focus_pattern", "Unavailable")).replace("_", " ").title(),
        "High-attention coverage": f"{focus_coverage:.2f}%",
        "Adaptive threshold used": f"{threshold:.2f}",
        "Loaded model": "Current deployed production classifier",
    }


def _metadata_panel_html(metadata: dict[str, Any]) -> str:
    if metadata.get("status") == "Awaiting inference":
        return f"""
        <div class="metadata-friendly">
            {_visual_label_html("Inference metadata")}
            <div class="metadata-empty">
                <div>
                    <strong>Inference metadata will appear here</strong>
                    Upload a scan and run the classifier to populate the request details, input size, class order, and explanation metadata for the current prediction.
                </div>
            </div>
        </div>
        """

    summary_items = _friendly_metadata_summary(metadata)
    summary_markup = "".join(
        f"<p><strong>{html.escape(label)}:</strong> {html.escape(value)}</p>"
        for label, value in summary_items.items()
    )
    metadata_json = html.escape(json.dumps(metadata, indent=2))
    return f"""
    <div class="metadata-friendly">
        {_visual_label_html("Inference metadata")}
        <h3>Readable summary</h3>
        {summary_markup}
        <details>
            <summary>Technical details</summary>
            <pre>{metadata_json}</pre>
        </details>
    </div>
    """


def _render_probability_distribution(probabilities: dict[str, float]) -> None:
    st.markdown("#### Class probabilities")
    if not probabilities:
        st.caption("Probability bars will appear here after the model runs on an uploaded scan.")
        return

    for row in _ordered_probability_rows(probabilities):
        label_col, value_col = st.columns([5, 1], gap="small")
        with label_col:
            st.markdown(f"**{row['label']}**")
        with value_col:
            st.markdown(f"**{row['percentage']}%**")
        st.progress(max(row["probability"], 0.01))


def _explanation_placeholder_markdown() -> str:
    return """
### Why the model predicted this

After inference, this section will explain the processing steps used by the model, highlight the most influential image regions, and summarize the technical metadata behind the current run.

1. **Preprocess**: The uploaded scan is resized and prepared for the TensorFlow classifier.
2. **Inspect**: The model input preview and attention heatmap appear here once inference completes.
3. **Summarize**: The metadata card captures the run details used to explain the prediction outcome.
"""


def _explanation_card_markdown(
    result: dict[str, Any],
    explanation: dict[str, Any],
    image_size: tuple[int, int],
) -> str:
    predicted_label = str(result["predicted_label"]).replace("_", " ").title()
    focus_region = str(explanation.get("focus_region", "unavailable")).title()
    focus_coverage = float(explanation.get("focus_coverage", 0.0))
    focus_pattern = str(explanation.get("focus_pattern", "unavailable")).replace("_", " ")
    high_attention_threshold = float(explanation.get("high_attention_threshold", 0.0))
    margin = float(explanation.get("margin", 0.0))
    runner_up_label = str(explanation.get("runner_up_label", "unavailable")).replace("_", " ").title()
    attention_layer = str(explanation.get("attention_layer", "unavailable"))
    error_message = explanation.get("error")

    if error_message:
        return f"""
### Why the model predicted this

The prediction is available, but the attention view could not be generated for this run. The fallback model-input preview is still shown below.

**Technical detail:** {str(error_message)}
"""

    return f"""
### Why the model predicted this

The model first resized the uploaded scan to **{image_size[0]} x {image_size[1]}** pixels, then passed it through the trained TensorFlow feature extractor. For this run, the dominant high-attention mass was centered in the **{focus_region}** region of the scan and the resulting saliency mask was **{focus_pattern}**, spanning about **{focus_coverage:.1%}** of the analyzed frame.

The winning class, **{predicted_label}**, finished ahead of **{runner_up_label}** by a probability margin of **{margin:.2%}**. The attention heatmap below is an input-saliency view that highlights the areas that most influenced the network's decision, derived from `{attention_layer}`. The high-attention mask is computed with an adaptive threshold calibrated to the current saliency map rather than a fixed percentage of pixels.

Brighter colored zones indicate stronger influence on the model output. For this image, the adaptive threshold was **{high_attention_threshold:.2f}**. This is an interpretability aid for the current inference run, not a clinical segmentation or a standalone diagnosis.
"""


def _classify_current_image(service: PredictionService) -> None:
    image = _uploaded_image()
    if image is None:
        st.error("Please upload an image before running inference.")
        return

    if not service.is_ready():
        st.error("The model is not loaded yet. Export a trained model into the models directory first.")
        return

    with st.spinner("Running classification..."):
        result = service.predict(image)
        explanation = service.explain_prediction(image, result)

    st.session_state.streamlit_inference_state = {
        "summary_html": gradio_ui._prediction_card_html(result),
        "probabilities": result["probabilities"],
        "explanation_html": _explanation_card_markdown(result, explanation, service.settings.image_size),
        "model_input_image": explanation.get("model_input_image"),
        "attention_heatmap_image": explanation.get("attention_heatmap_image"),
        "metadata": _build_probability_metadata(result, explanation, service),
    }


def _author_image_html(image_path: Path) -> str:
    return f"""
    <div class="author-photo-shell">
        <img src="{_data_uri_from_path(image_path)}" alt="{html.escape(image_path.stem)}" />
    </div>
    """


def _altair_bar_chart(frame, x, y, color, y_domain: list[float] | None = None, sort=None, x_label_angle=0):
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X(x, sort=sort, axis=alt.Axis(labelAngle=x_label_angle)),
            y=alt.Y(y, scale=alt.Scale(domain=y_domain) if y_domain else alt.Undefined),
            color=alt.Color(
                color,
                scale=alt.Scale(domain=["Infected", "Uninfected"], range=["#0e4d73", "#178b76"]),
                legend=None,
            ),
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def _altair_line_chart(frame, series_domain: list[str], series_range: list[str], x: str, y: str, color: str):
    chart = (
        alt.Chart(frame)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X(x, title="Epoch"),
            y=alt.Y(y),
            color=alt.Color(
                color,
                scale=alt.Scale(domain=series_domain, range=series_range),
                legend=alt.Legend(title="Series"),
            ),
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def _test_metrics_chart(frame) -> None:
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color="#178b76")
        .encode(
            x=alt.X("metric", sort=None, title="Metric"),
            y=alt.Y("value", title="Score"),
            tooltip=["metric", alt.Tooltip("value", format=".4f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def _accuracy_interpretation_html(history) -> str:
    if history.empty or "accuracy" not in history or "val_accuracy" not in history:
        return """
        <div class="eda-note accuracy-note">
            <p>
                This chart should be read as an internal optimization view. Even when training and validation accuracy appear closely aligned, stronger evidence of generalization still comes from grouped resampling, study-level splitting, and external validation on independent data.
            </p>
        </div>
        """

    final_train = float(history["accuracy"].iloc[-1])
    final_val = float(history["val_accuracy"].iloc[-1])
    gap = abs(final_train - final_val)

    if gap <= 0.02:
        pattern_copy = (
            "The training and validation curves remain closely aligned here, so this plot does not show the classic divergence pattern that usually signals obvious overfitting."
        )
    else:
        pattern_copy = (
            "There is some visible separation between the training and validation curves here, so this panel should be reviewed carefully for possible overfitting dynamics."
        )

    return f"""
    <div class="eda-note accuracy-note">
        <p>
            {pattern_copy} Even so, the near-perfect validation trajectory should still be treated as an <strong>internal result</strong>, not proof of broad external generalization. The final recorded values are <strong>{final_train:.2%}</strong> training accuracy and <strong>{final_val:.2%}</strong> validation accuracy, with a final gap of <strong>{gap:.2%}</strong>.
        </p>
    </div>
    """


def _render_codebase_tab() -> None:
    st.markdown(
        _section_intro_html(
            "Open Source",
            "Codebase",
            "Explore the repository behind this Streamlit experience, review the implementation details, and trace the project structure for further collaboration or extension.",
        ),
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([6, 4], gap="large")
    with left_col:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="helper-copy">
                    <span class="section-kicker">Primary Repository</span>
                    <h2>Streamlit codebase</h2>
                    <p>
                        This repository contains the current Streamlit implementation, including the interface layer,
                        project assets, EDA presentation, feedback workflow, and deployment-oriented app structure.
                    </p>
                    <p><a href="{html.escape(STREAMLIT_REPO_URL)}">Open the Streamlit codebase on GitHub</a></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with right_col:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="helper-copy">
                    <span class="section-kicker">Related Build</span>
                    <h2>Original production stack</h2>
                    <p>
                        The earlier production deployment built around FastAPI, Gradio, Docker, and Hugging Face
                        remains available in its own repository for comparison and reference.
                    </p>
                    <p><a href="{html.escape(PRODUCTION_REPO_URL)}">View the original production repository</a></p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_classify_tab(service: PredictionService, project_root: Path) -> None:
    bundle_bytes = _get_demo_bundle_bytes(str(project_root))
    bundle_name = demo_bundle_filename()
    download_link = _download_link_html(bundle_bytes, bundle_name)
    state = st.session_state.streamlit_inference_state
    training_summary = gradio_ui._load_training_summary(project_root)
    banner_path = project_root / "assets" / "banner" / "endometrium_banner.png"

    hero_left, hero_right = st.columns(2, gap="large")
    with hero_left:
        st.markdown(
            f'<div class="hero-shell">{_hero_copy_html(training_summary)}</div>',
            unsafe_allow_html=True,
        )
    with hero_right:
        st.markdown(
            f'<div class="hero-shell">{_hero_banner_html(banner_path)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        _status_strip_html(
            "Model-ready inference flow",
            "Balanced two-class classifier",
            "Downloadable blind-test pack",
            "Research-focused explanation layer",
        ),
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns(2, gap="large")
    uploader_key = f"streamlit_uploader_{st.session_state.streamlit_upload_nonce}"

    with left_col:
        with st.container(border=True):
            st.markdown(_upload_intro_html(download_link), unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Endometrial scan",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=False,
                key=uploader_key,
            )
            if uploaded_file is not None:
                _store_uploaded_image(uploaded_file)

            image = _uploaded_image()
            if image is not None:
                st.image(image, use_container_width=True)
            else:
                st.markdown(_preview_placeholder_html(), unsafe_allow_html=True)

    with right_col:
        with st.container(border=True):
            st.markdown(_result_intro_html(), unsafe_allow_html=True)
            st.markdown(state["summary_html"], unsafe_allow_html=True)
            _render_probability_distribution(state["probabilities"])

    action_left, action_right = st.columns(2, gap="large")
    with action_left:
        if st.button("Run classification", type="primary", use_container_width=True):
            _classify_current_image(service)
            st.rerun()
    with action_right:
        if st.button("Clear", use_container_width=True):
            _reset_inference_state()
            st.rerun()

    state = st.session_state.streamlit_inference_state
    with st.container(border=True):
        st.markdown(state["explanation_html"])
        visual_left, visual_right = st.columns(2, gap="large")
        with visual_left:
            st.markdown(_visual_label_html("Model input used for inference"), unsafe_allow_html=True)
            if state["model_input_image"] is None:
                st.markdown(_visual_placeholder_panel_html("Inference image"), unsafe_allow_html=True)
            else:
                st.markdown(
                    _visual_image_panel_html(state["model_input_image"], "Model input used for inference"),
                    unsafe_allow_html=True,
                )
        with visual_right:
            st.markdown(_visual_label_html("Model attention heatmap"), unsafe_allow_html=True)
            if state["attention_heatmap_image"] is None:
                st.markdown(_visual_placeholder_panel_html("Attention heatmap"), unsafe_allow_html=True)
            else:
                st.markdown(
                    _visual_image_panel_html(state["attention_heatmap_image"], "Model attention heatmap"),
                    unsafe_allow_html=True,
                )
        st.markdown(_metadata_panel_html(state["metadata"]), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="helper-copy" style="text-align:center; margin-top: 1rem;">
            <p>
                This tool supports research, experimentation, and AI-assisted screening workflows.
                Final clinical interpretation should remain with qualified medical experts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_download_tab(project_root: Path) -> None:
    bundle_bytes = _get_demo_bundle_bytes(str(project_root))
    bundle_name = demo_bundle_filename()
    st.markdown(
        _section_intro_html(
            "Demo Assets",
            "Download test images",
            "Download the curated blind-test image bundle when you want to explore the app without sourcing your own scan files.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="download-highlight">
            <p>
                The download bundle contains 20 curated test images from the held-out test split, packaged
                with neutral filenames and randomized ordering so the files do not reveal their class labels upfront.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left_col, right_col = st.columns([6, 4], gap="large")
    with left_col:
        with st.container(border=True):
            st.markdown(
                """
                <div class="download-card">
                    <h3>What is included</h3>
                    <ul>
                        <li>A balanced set of 20 demo scan images</li>
                        <li>Neutral, shuffled scan filenames for blind testing</li>
                        <li>A small <code>README.txt</code> inside the archive describing the bundle</li>
                    </ul>
                    <h3>Suggested use</h3>
                    <ul>
                        <li>Download the bundle</li>
                        <li>Upload any image into the <strong>Classify</strong> tab</li>
                        <li>Compare outputs across different scan examples</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with right_col:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="download-card">
                    <h3>Ready to download</h3>
                    <p>File name: <code>{html.escape(bundle_name)}</code></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button(
                label="Download test image bundle",
                data=bundle_bytes,
                file_name=bundle_name,
                mime="application/zip",
                use_container_width=True,
            )


def _render_feedback_tab(project_root: Path) -> None:
    st.markdown(
        _section_intro_html(
            "Feedback Loop",
            "Share your experience",
            "Tell us what feels clear, what needs refinement, and which improvements would make the project more useful for research, demos, or collaboration.",
        ),
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([6, 4], gap="large")
    form_nonce = st.session_state.streamlit_feedback_form_nonce

    with left_col:
        with st.container(border=True):
            st.markdown(
                """
                <div class="feedback-note">
                    <p>
                        Thoughtful feedback helps us improve the research presentation, interface clarity,
                        explainability workflow, and practical usefulness of the app.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.form(f"feedback_form_{form_nonce}"):
                name = st.text_input("Name or alias", placeholder="Optional", key=f"feedback_name_{form_nonce}")
                role = st.text_input(
                    "Role / affiliation",
                    placeholder="Optional",
                    key=f"feedback_role_{form_nonce}",
                )
                recommendation = st.radio(
                    "Would you recommend this project?",
                    options=["Strongly recommend", "Recommend", "Needs improvement"],
                    index=1,
                    key=f"feedback_recommendation_{form_nonce}",
                )
                rating = st.slider(
                    "Overall experience rating",
                    min_value=1,
                    max_value=5,
                    value=5,
                    key=f"feedback_rating_{form_nonce}",
                )
                thoughts = st.text_area(
                    "Your thoughts",
                    height=140,
                    placeholder="What stood out to you about the project, model explanation, or interface?",
                    key=f"feedback_thoughts_{form_nonce}",
                )
                suggestions = st.text_area(
                    "Recommendations / suggestions",
                    height=120,
                    placeholder="What would you improve, extend, or refine next?",
                    key=f"feedback_suggestions_{form_nonce}",
                )
                submitted = st.form_submit_button("Share feedback", type="primary", use_container_width=True)

            if submitted:
                if not thoughts.strip() and not suggestions.strip():
                    st.error("Please share at least one thought or recommendation before submitting feedback.")
                else:
                    save_feedback(
                        project_root,
                        name=name,
                        role=role,
                        recommendation=recommendation,
                        rating=rating,
                        thoughts=thoughts,
                        suggestions=suggestions,
                    )
                    st.session_state.streamlit_feedback_status_html = gradio_ui._feedback_success_html(recommendation)
                    st.session_state.streamlit_feedback_form_nonce += 1
                    st.rerun()

    with right_col:
        with st.container(border=True):
            st.markdown(st.session_state.streamlit_feedback_status_html, unsafe_allow_html=True)
            st.markdown(
                """
                <div class="feedback-card-copy">
                    <h3>Helpful feedback areas</h3>
                    <ul>
                        <li>Ease of use for first-time visitors</li>
                        <li>Clarity of the prediction and explanation flow</li>
                        <li>Trustworthiness of the interface and research framing</li>
                        <li>Features or improvements you would like to see next</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_eda_tab(project_root: Path) -> None:
    training_summary = gradio_ui._load_training_summary(project_root)
    training_history = gradio_ui._load_training_history(project_root)
    class_distribution_frame = gradio_ui._build_class_distribution_frame(training_summary)
    split_distribution_frame = gradio_ui._build_split_distribution_frame(training_summary)
    test_metrics_frame = gradio_ui._build_test_metrics_frame(training_summary)
    accuracy_curve_frame = gradio_ui._build_curve_frame(
        training_history,
        {"accuracy": "Training Accuracy", "val_accuracy": "Validation Accuracy"},
    )
    loss_curve_frame = gradio_ui._build_curve_frame(
        training_history,
        {"loss": "Training Loss", "val_loss": "Validation Loss"},
    )
    demo_profile_frame = gradio_ui._build_demo_profile_frame(project_root)
    class_chart_limit = gradio_ui._safe_chart_limit(class_distribution_frame, "count", minimum=10.0)
    split_chart_limit = gradio_ui._safe_chart_limit(split_distribution_frame, "count", minimum=10.0)

    st.markdown(
        _section_intro_html(
            "Evidence View",
            "EDA Lab",
            "Review the curation pipeline, partition strategy, optimization traces, and internal evaluation details behind the deployed classifier.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(gradio_ui._eda_overview_html(training_summary), unsafe_allow_html=True)

    top_left, top_right = st.columns(2, gap="large")
    with top_left:
        with st.container(border=True):
            st.markdown(gradio_ui._class_distribution_markdown(training_summary), unsafe_allow_html=True)
            _altair_bar_chart(class_distribution_frame, "class:N", "count:Q", "class:N", class_chart_limit)
    with top_right:
        with st.container(border=True):
            st.markdown(gradio_ui._split_strategy_markdown(training_summary), unsafe_allow_html=True)
            _altair_bar_chart(
                split_distribution_frame,
                "split_class:N",
                "count:Q",
                "class:N",
                split_chart_limit,
                sort=[
                    "Train - Infected",
                    "Train - Uninfected",
                    "Validation - Infected",
                    "Validation - Uninfected",
                    "Test - Infected",
                    "Test - Uninfected",
                ],
                x_label_angle=-20,
            )

    mid_left, mid_right = st.columns(2, gap="large")
    with mid_left:
        with st.container(border=True):
            st.markdown(gradio_ui._accuracy_curves_markdown(training_history), unsafe_allow_html=True)
            _altair_line_chart(
                accuracy_curve_frame,
                ["Training Accuracy", "Validation Accuracy"],
                ["#0e4d73", "#178b76"],
                "epoch:Q",
                "value:Q",
                "series:N",
            )
            st.markdown(_accuracy_interpretation_html(training_history), unsafe_allow_html=True)
    with mid_right:
        with st.container(border=True):
            st.markdown(gradio_ui._loss_curves_markdown(training_history), unsafe_allow_html=True)
            _altair_line_chart(
                loss_curve_frame,
                ["Training Loss", "Validation Loss"],
                ["#0e4d73", "#178b76"],
                "epoch:Q",
                "value:Q",
                "series:N",
            )

    lower_left, lower_right = st.columns([4, 6], gap="large")
    with lower_left:
        with st.container(border=True):
            st.markdown(gradio_ui._research_safeguards_markdown(training_summary), unsafe_allow_html=True)
    with lower_right:
        with st.container(border=True):
            st.markdown(gradio_ui._held_out_evaluation_markdown(training_summary), unsafe_allow_html=True)
            _test_metrics_chart(test_metrics_frame)

    bottom_left, bottom_right = st.columns([4, 6], gap="large")
    with bottom_left:
        with st.container(border=True):
            st.markdown(gradio_ui._demo_profile_markdown(), unsafe_allow_html=True)
            st.dataframe(demo_profile_frame, use_container_width=True, hide_index=True)
    with bottom_right:
        with st.container(border=True):
            st.markdown(gradio_ui._interpretation_note_markdown(training_summary), unsafe_allow_html=True)
            st.markdown(
                """
                - These performance estimates are internal to the current curated archive and should not be interpreted as population-level performance.
                - Group-aware splitting reduces one major source of optimistic bias, but study-level and external validation remain the stronger tests of generalization.
                - The app now exposes its curation and audit safeguards so readers can review the evaluation protocol with greater transparency.
                """
            )


def _render_about_tab(project_root: Path) -> None:
    training_summary = gradio_ui._load_training_summary(project_root)
    assets_dir = project_root / "assets"
    st.markdown(
        _section_intro_html(
            "Project Context",
            "About the project",
            "Understand the medical motivation, data curation approach, research framing, and the people behind the work.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(gradio_ui._project_about_markdown(training_summary), unsafe_allow_html=True)
    st.markdown(gradio_ui.AUTHOR_MARKDOWN, unsafe_allow_html=True)
    columns = st.columns(len(gradio_ui.AUTHOR_PROFILES), gap="large")
    for column, profile in zip(columns, gradio_ui.AUTHOR_PROFILES):
        with column:
            with st.container(border=True):
                image_path = gradio_ui._author_image_path(profile, assets_dir)
                if image_path is not None:
                    st.markdown(_author_image_html(image_path), unsafe_allow_html=True)
                st.markdown(gradio_ui._author_card_markdown(profile), unsafe_allow_html=True)


def _render_future_tab() -> None:
    st.markdown(
        _section_intro_html(
            "Roadmap",
            "Future development",
            "Explore the main enrichment tracks that could strengthen evaluation quality, usability, deployment robustness, and future research value.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(gradio_ui._future_dev_markdown(), unsafe_allow_html=True)
    left_col, right_col = st.columns([6, 4], gap="large")
    with left_col:
        with st.container(border=True):
            st.markdown(
                """
                <div class="helper-copy">
                    <span class="section-kicker">Roadmap Highlights</span>
                    <h2>Where the project can grow next</h2>
                    <p>
                        The roadmap emphasizes stronger evaluation, richer clinical grounding, more robust
                        deployment practice, and a more useful human-review workflow. The aim is not only to
                        improve model performance, but also to improve evidence quality and research usability.
                    </p>
                    <ul>
                        <li>strengthen study-level and external validation</li>
                        <li>improve calibration, uncertainty handling, and explainability</li>
                        <li>explore ROI-focused and multimodal modeling</li>
                        <li>support clinician feedback loops and richer audit trails</li>
                        <li>benchmark stronger architectures under the same evaluation protocol</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with right_col:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="helper-copy">
                    <span class="section-kicker">Repository Guide</span>
                    <h2>Open the full roadmap</h2>
                    <p>
                        The complete future-development document lives in the repository as
                        <code>future development.md</code>.
                    </p>
                    <p><a href="{html.escape(gradio_ui.FUTURE_DEVELOPMENT_URL)}">View the full roadmap on GitHub</a></p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    st.set_page_config(
        page_title="Endometrial Infection Classification App",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_css()
    _ensure_state()

    service = _get_service()
    project_root = service.settings.project_root
    classify_tab, download_tab, eda_tab, about_tab, future_tab, codebase_tab, feedback_tab = st.tabs(
        ["Classify", "Download", "EDA Lab", "About", "Future Dev", "Codebase", "Feedback"]
    )

    with classify_tab:
        _render_classify_tab(service, project_root)

    with download_tab:
        _render_download_tab(project_root)

    with eda_tab:
        _render_eda_tab(project_root)

    with about_tab:
        _render_about_tab(project_root)

    with future_tab:
        _render_future_tab()

    with codebase_tab:
        _render_codebase_tab()

    with feedback_tab:
        _render_feedback_tab(project_root)

    st.markdown(gradio_ui.FOOTER_HTML, unsafe_allow_html=True)
