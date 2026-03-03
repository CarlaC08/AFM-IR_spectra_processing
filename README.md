### Global information  
AFM-IR_spectra_processing is a streamlit app, made in the AFM-IR team of the Institut de Chimie Physique of Paris-Saclay University, to help users of Musiics plateform (https://www.icp.universite-paris-saclay.fr/musiics/) to process their data.
To run it you need streamlit (https://streamlit.io/) in your python environment.  
Once you've downloaded the script, open a terminal under python, and use the lines code :  
cd [path_to_script]  
streamlit run IR_spectra_processing.py  
This app is protected by an MIT License, to cite this work please refer to the DOI : https://doi.org/10.5281/zenodo.18744084

### Contents of the app

In this app, you'll find 2 tabs : "Laser breaks correction" and "Visualisation of IconIR spectra on map".  

#### Laser breaks correction  
This 1st tab allows to correct the break due to a transition from a laser to an other (mutliple QCL are needed to have a wavenumber range of 900 – 1900 cm-1).  
For this tab, mutliple instrument are available :
 - GloveBox : from Bruker, it corresponds to the IconIR in specific environment
 - IconIR : from Bruker
 - OPTIR Mirage : from Photothermal
 - Nano IR2 : from Anasys Instrument

#### Visualisation of IconIR spectra on map
The 2nd tab **ONLY** works for the IconIR system from Bruker (i.e. IconIR and GloveBox). It allows to plot the position of the spectra on the corresponding map, and to visualise and apply basic operation to the spectra.  
This tab makes it easier to create figures thanks to multiple personalization elements.
