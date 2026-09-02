from pathlib import Path
import runpy


runpy.run_path(Path(__file__).with_name("IR_spectra_processing.py"), run_name="__main__")