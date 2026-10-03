"""
PVision AI: UI Components
Reusable Streamlit display modules.
"""

import streamlit as st
import json
from pathlib import Path
from typing import Dict, Any

def render_metric_card(title: str, value: str, subtext: str = ""):
    """Renders a modern metric card in Streamlit."""
    sub_html = f'<div class="metric-sub">{subtext}</div>' if subtext else ''
    card_html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def load_json_metrics(file_path: Path) -> Dict[str, Any]:
    """Safely loads a JSON report file."""
    if not file_path.exists():
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
