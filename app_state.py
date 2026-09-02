import numpy as np
import streamlit as st


def initialize_session_state():
    defaults = {
        # Correction tab
        "spectra_files": None,
        "bkg_files": None,
        "breaks_wn_val": [],
        "off_bkg": None,
        "window_bkg": None,
        "polynom_order_bkg": None,
        # Position tab
        "list_num": np.array([]),
        "Topography": [],
        "Topography_image": False,
        "IR": [],
        "IR_image": False,
        "map_size": [0, 0],
        "map_unit": None,
        "positions_submit": False,
        "spectra_submit": False,
        "topo_submit": False,
        "IR_submit": False,
        "fragment_runs": 0,
        # Plot parameters
        "normalization": "None",
        "arrow_color": "#FFFFFF",
        "arrow_head": 1,
        "arrow_side": "end",
        "arrow_size": 1.0,
        "arrow_width": 1.5,
        "bg_color": None,
        "border_color": None,
        "border_pad": 1.0,
        "border_width": 1.0,
        "box_height": None,
        "box_width": None,
        "textfont_color": "#000000",
        "font_family": "Arial",
        "font_size": 10.0,
        "font_style": "normal",
        "font_textcase": "normal",
        "font_variant": "normal",
        "text_angle": 0,
        "vertical_alignement": "middle",
        "horizontal_alignement": "center",
        "abled_bgColor": False,
        "abled_BorderColor": False,
        "prefix": "",
        "baseline_corr": False,
        "to_plot": np.array([], dtype=int),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, (list, np.ndarray)) else value