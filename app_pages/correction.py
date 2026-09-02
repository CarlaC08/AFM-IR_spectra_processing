import os
import runpy
from pathlib import Path


previous_page = os.environ.get("AFM_IR_PAGE")
os.environ["AFM_IR_PAGE"] = "correction"
try:
	runpy.run_path(Path(__file__).parents[1] / "IR_spectra_processing.py", run_name="__main__")
finally:
	if previous_page is None:
		os.environ.pop("AFM_IR_PAGE", None)
	else:
		os.environ["AFM_IR_PAGE"] = previous_page