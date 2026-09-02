from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import savgol_filter


def find_nearest(array, value):
    array = np.asarray(array)
    return array[(np.abs(array - value)).argmin()]


def find_nearest_idx(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def uncorrect_background(spectrum, background):
    uncorrected_spectrum = deepcopy(spectrum)
    uncorrected_spectrum[:, 1] = spectrum[:, 1] * background[:, 1]
    return uncorrected_spectrum


def offset_background_correction(spectrum, background, apply_offset, offset_wavenumber):
    corrected_spectrum = deepcopy(spectrum)
    if apply_offset:
        offset_section = np.argmin(np.abs(spectrum[:, 0] - offset_wavenumber))
        offset = np.nanmin(spectrum[offset_section:, 1])
        corrected_spectrum[:, 1] = corrected_spectrum[:, 1] - offset
    corrected_spectrum[:, 1] = corrected_spectrum[:, 1] / background[:, 1]
    return corrected_spectrum


def stitching_pieces(spectrum, break_index):
    stitched_spectrum = deepcopy(spectrum)
    beta = spectrum[break_index + 1, 1] / spectrum[break_index - 1, 1]
    break_amplitude = spectrum[break_index + 1, 1]
    stitched_spectrum[:break_index, 1] = spectrum[:break_index, 1] * beta
    stitched_spectrum[break_index, 1] = break_amplitude
    return stitched_spectrum


def break_correction(spectrum, break_values, break_indices, n_delta):
    corrected_spectrum = deepcopy(spectrum)
    for index in break_indices:
        corrected_spectrum = stitching_pieces(corrected_spectrum, index)
    return corrected_spectrum


@dataclass(frozen=True)
class SpectrumCorrectionConfig:
    apply_offset: bool
    offset_wavenumber: Optional[float]
    break_values: np.ndarray
    spectra_divided: bool
    smooth_background: bool
    background_window: Optional[int] = None
    background_polynomial_order: Optional[int] = None


@st.cache_data(ttl=3600, max_entries=1, show_spinner="Break correction")
def correct_spectra(spectra, background, spectra_header, config, n_delta=2):
    corrected_spectra = deepcopy(spectra)
    break_indices = [int(find_nearest_idx(spectra[:, 0], value)) for value in config.break_values]
    if config.smooth_background:
        break_indices.sort()
        smoothed_background = background.copy()
        for index in range(len(break_indices)):
            try:
                smoothed_background[break_indices[index]:break_indices[index + 1], 1] = savgol_filter(
                    background[break_indices[index]:break_indices[index + 1], 1],
                    config.background_window,
                    config.background_polynomial_order,
                )
                smoothed_background[break_indices[index]] = background[break_indices[index]]
                smoothed_background[break_indices[index + 1]] = background[break_indices[index + 1]]
            except IndexError:
                pass
            except ValueError:
                if break_indices[index + 1] - break_indices[index] < config.background_window:
                    smoothed_background[break_indices[index]:break_indices[index + 1], 1] = savgol_filter(
                        background[break_indices[index]:break_indices[index + 1], 1],
                        break_indices[index + 1] - break_indices[index] - 1,
                        config.background_polynomial_order,
                    )
                    smoothed_background[break_indices[index]] = background[break_indices[index]]
                    smoothed_background[break_indices[index + 1]] = background[break_indices[index + 1]]
        smoothed_background[break_indices[-1] + 1:, 1] = savgol_filter(background[break_indices[-1] + 1:, 1], 15, 1)
    else:
        smoothed_background = background.copy()

    for column_index in np.arange(1, spectra.shape[1], 1):
        current_spectrum = spectra[:, (0, column_index)]
        if config.spectra_divided:
            current_spectrum = uncorrect_background(current_spectrum, background)
        corrected_spectrum = offset_background_correction(
            current_spectrum,
            smoothed_background,
            config.apply_offset,
            config.offset_wavenumber,
        )
        corrected_spectra[:, (0, column_index)] = break_correction(
            corrected_spectrum,
            config.break_values,
            break_indices,
            n_delta,
        )
    return pd.DataFrame(corrected_spectra, columns=spectra_header.split(",")).set_index(spectra_header.split(",")[0])