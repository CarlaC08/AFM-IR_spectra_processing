from io import StringIO
import re
import zipfile

import numpy as np
import pandas as pd


def open_series(series_file):
    with zipfile.ZipFile(series_file, "r") as archive:
        file_names = archive.namelist()
        with archive.open(file_names[0], "r") as file:
            binary_data = file.read().decode("ascii").split("Series0")[1].split(" ")
            name = series_file.name.replace(".series", "")
            yunit = [item.split("NameY=")[1].replace('"', "") for item in binary_data if "NameY=" in item][0]
            kind = [item.split("Kind=")[1].replace('"', "") for item in binary_data if "Kind=" in item][0]
        with archive.open(file_names[-1], "r") as file:
            if kind == "Text":
                return pd.DataFrame(
                    data=np.loadtxt(file, delimiter=","),
                    columns=["Wavenumber (cm-1)", name.replace(".Series", "") + " (" + yunit + ")"],
                )
            if kind == "Binary":
                data = np.frombuffer(file.read(), dtype=np.float64)
                half = len(data) // 2
                return pd.DataFrame(
                    {"Wavenumber (cm-1)": data[:half], name + " (" + yunit + ")": data[half:]}
                )
    raise ValueError(f"Unsupported series data kind: {kind}")


def open_spectrum_mirage(path_spectra, path_background, register_type, background_in_file, organisation):
    if register_type == "No":
        if organisation == "Row":
            spectra_file = pd.read_csv(path_spectra, header=None).T.dropna()
            spectra_file.columns = spectra_file.iloc[0]
            spectra_file = spectra_file.drop(index=0).astype(float)
        else:
            spectra_file = pd.read_csv(path_spectra, header=0).dropna()
        if background_in_file == "Yes":
            spectra_header = [column for column in spectra_file.columns if "mV" in column or "cm" in column]
            background_header = [column for column in spectra_file.columns if "Background" in column or "cm" in column]
        else:
            background_file = pd.read_csv(path_background, header=0).dropna()
            spectra_header, background_header = spectra_file.columns, background_file.columns
    else:
        spectra_file = pd.read_csv(path_spectra, header=None).T.dropna()
        background_file = pd.read_csv(path_background, header=0).dropna()
        spectra_file = spectra_file.iloc[2:, :]
        spectra_header = ["Wavenumber"] + ["Spectrum_" + str(i) for i in range(spectra_file.shape[1] - 1)]
        background_header = background_file.columns

    if register_type == "No":
        spectra = spectra_file.loc[:, spectra_header].values
    else:
        spectra = spectra_file.reset_index().drop("index", axis=1).values
    background = spectra_file.loc[:, background_header].values if background_in_file == "Yes" else background_file.loc[:, background_header].values
    spectra_header = ",".join(spectra_header)
    background_header = ",".join(background_header)
    spectra = spectra[(spectra[:, 0] >= 1241) | (spectra[:, 0] <= 1211)]
    background = background[(background[:, 0] >= 1241) | (background[:, 0] <= 1211)]
    return spectra, background, spectra_header, background_header


def open_spectrum_glove_box(path, extension, multiple_file):
    if extension == ["series"]:
        spectra_lines = open_series(path) if not multiple_file else pd.concat([open_series(item) for item in path], axis=1).T.drop_duplicates(keep="first").T
        data, spectra_header = spectra_lines.values, ",".join(spectra_lines.columns.values)
    else:
        spectra_lines = [str(item, encoding="utf-8") for item in path.readlines()]
        header = spectra_lines[0].split(",")[1:]
        spectra_header = ",".join(["Wavenumber"] + [item.split("/")[0].split(".")[-1] for item in header])
        data = _read_csv_lines(spectra_lines, len(spectra_header.split(",")))
    return data[np.argsort(data[:, 0])], spectra_header


def open_background_glove_box(path):
    if ".series" in path.name or ".Series" in path.name:
        spectra_lines = open_series(path)
        data, spectra_header = spectra_lines.values, ",".join(spectra_lines.columns.values)
    else:
        spectra_lines = [str(item, encoding="utf-8") for item in path.readlines()]
        header = spectra_lines[0].split(",")[1:]
        spectra_header = ",".join(["Wavenumber"] + [item.split("/")[0].split(".")[-1] for item in header])
        data = _read_csv_lines(spectra_lines, len(spectra_header.split(",")))
    return data[np.argsort(data[:, 0])], spectra_header


def _read_csv_lines(lines, column_count):
    data = np.zeros((len(lines) - 1, column_count))
    for row_index, line in enumerate(lines[1:]):
        values = line.split(",")
        for column_index, value in enumerate(values):
            if value not in ("\n", "", "\r\n"):
                data[row_index, column_index] = float(value)
    return data


def open_spectrum_nano2(path):
    if ".irb" in path.name:
        infos = StringIO(path.getvalue().decode("utf-16")).read()
        params, spectra_info = re.split("<Table>|</Table>", infos)[0:2]
        values = re.split("<double>|</double>", spectra_info)[1::2]
        wavenumbers = np.linspace(
            float(re.split("<StartWavenumber>|</StartWavenumber>", params)[1]),
            float(re.split("<EndWavenumber>|</EndWavenumber>", params)[1]),
            len(values),
        )
        data = np.column_stack((wavenumbers, [float(value) for value in values]))
        header = ["cm-1" + re.split("<Units>|</Units>", params)[1]]
    else:
        content = StringIO(path.getvalue().decode("utf-8")).read()
        spectra_lines = pd.DataFrame([line.split(",") for line in content.split("\n")])
        spectra_lines.columns = spectra_lines.iloc[0]
        spectra_lines = spectra_lines.drop(columns=[column for column in spectra_lines.columns if "deg" in column])[1:-1].astype(float)
        header = ",".join(["Wavenumber"] + [column.split("/")[0].replace(" ", ".") for column in spectra_lines.columns[1:]])
        data = spectra_lines.values
    return data, header