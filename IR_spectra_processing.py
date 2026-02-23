# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:39:46 2025
@break correction : julien rojas
@application: carla collange
"""
# %% Packages

import streamlit as st
import plotly.express as px
from streamlit_plotly_events import plotly_events
import numpy as np
from cycler import cycler
from scipy.signal import savgol_filter
from scipy import stats
import pandas as pd
from copy import deepcopy
from pathlib import Path
from itertools import cycle
import cv2
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator
import seaborn as sns
import re
import zipfile
from io import StringIO
from sklearn.preprocessing import normalize
import os as os
from skimage import io

#%% Définitions de variables

# Initialisation for the correction tab
if 'spectra_files'  not in st.session_state : st.session_state.spectra_files = None
if 'bkg_files'  not in st.session_state : st.session_state.bkg_files = None
if 'breaks_wn_val' not in st.session_state : st.session_state.breaks_wn_val = []

# Initialisation for the position tab
if 'list_num' not in st.session_state : st.session_state.list_num=np.array([])
if 'Topography' not in st.session_state : st.session_state.Topography = []
if 'Topography_image' not in st.session_state : st.session_state.Topography_image=False
if 'IR' not in st.session_state : st.session_state.IR = []
if 'IR_image' not in st.session_state : st.session_state.IR_image=False
if 'map_size' not in st.session_state : st.session_state.map_size=[0,0]
if 'map_unit' not in st.session_state : st.session_state.map_unit=None
if 'positions_submit' not in st.session_state : st.session_state.positions_submit=False
if 'spectra_submit' not in st.session_state : st.session_state.spectra_submit=False
if 'topo_submit' not in st.session_state : st.session_state.topo_submit=False
if 'IR_submit' not in st.session_state : st.session_state.IR_submit=False
if 'fragment_runs' not in st.session_state : st.session_state.fragment_runs=0

# Needed for the plot
st.session_state.colorscales = [i for j in [[k, k+'_r'] for k in px.colors.named_colorscales()] for i in j]
symbols_names = [i for i in SymbolValidator().values[2::3] if '-dot' not in i]
ls_option = ['dash', 'dashdot', 'dot', 'longdash', 'longdashdot', 'solid']

#%% Fonctions corrections

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def Open_spectrum_mirage(Path_spectra, Path_bkg, type_register, bkg_in_file, organisation) :
    if type_register=='No':
        if organisation=='Row':
            Spectra_file = pd.read_csv(Path_spectra, header=None).T.dropna()
            Spectra_file.columns = Spectra_file.iloc[0]
            Spectra_file = Spectra_file.drop(index=0).astype(float)
        else : Spectra_file = pd.read_csv(Path, header=0).dropna()
        if bkg_in_file=='Yes' :
            Spectra_header, bkg_header = [i for i in Spectra_file.columns if 'mV' in i or 'cm' in i], [i for i in Spectra_file.columns if 'Background' in i or 'cm' in i]
            sorted_arr_spec, sorted_arr_bkg = Spectra_file.loc[:,Spectra_header].values, Spectra_file.loc[:,bkg_header].values
        else :
            Bkg_file = pd.read_csv(Path_bkg, header=0).dropna()
            Spectra_header, bkg_header = Spectra_file.columns, Bkg_file.columns
            sorted_arr_spec, sorted_arr_bkg = Spectra_file.loc[:,Spectra_header].values, Bkg_file.loc[:,bkg_header].values
    else :
        Spectra_file = pd.read_csv(Path_spectra, header=None).T.dropna() ; Bkg_file = pd.read_csv(Path_bkg, header=0).dropna()
        Spectra_file = Spectra_file.iloc[2:,:]
        Spectra_header, bkg_header = ['Wavenumber'] + ['Spectrum_'+str(i) for i in range(Spectra_file.shape[1]-1)], Bkg_file.columns
        sorted_arr_spec, sorted_arr_bkg = Spectra_file.reset_index().drop('index', axis=1).values, Bkg_file.loc[:,bkg_header].values
    Spectra_header, bkg_header = ','.join(Spectra_header), ','.join(bkg_header)
    sorted_arr_spec = sorted_arr_spec[(sorted_arr_spec[:,0]>=1241) | (sorted_arr_spec[:,0]<=1211)]
    sorted_arr_bkg = sorted_arr_bkg[(sorted_arr_bkg[:,0]>=1241) | (sorted_arr_bkg[:,0]<=1211)]
    return sorted_arr_spec, sorted_arr_bkg, Spectra_header, bkg_header

def Open_series(series_file):
    with zipfile.ZipFile(series_file, 'r') as zip:
        file_names = zip.namelist()
        with zip.open(file_names[0],'r') as f:
            binary_data = f.read().decode('ascii').split('Series0')[1].split(' ')
            name, yunit, kind =  [series_file.name.replace('.series','')][0], [i.split('NameY=')[1].replace('"', '') for i in binary_data if 'NameY=' in i][0], [i.split('Kind=')[1].replace('"', '') for i in binary_data if 'Kind=' in i][0]
        with zip.open(file_names[-1],'r') as f:
            if kind == 'Text':
                spectrum = pd.DataFrame(data = np.loadtxt(f,delimiter=','), columns=['Wavenumber (cm-1)',name.replace('.Series', '')+' ('+yunit+')'])
            elif kind == 'Binary':
                d = np.frombuffer(f.read(), dtype=np.float64)
                half = len(d) // 2
                spectrum = pd.DataFrame({'Wavenumber (cm-1)': d[:half], name+' ('+yunit+')': d[half:]})
    return spectrum

def Open_spectrum_GloveBox(Path, extension, multiple_file) :
    if extension == ['series'] :
        if multiple_file==False: Spectra_lines = Open_series(Path)
        else : Spectra_lines = pd.concat([Open_series(i) for i in Path], axis=1).T.drop_duplicates(keep='first').T
        Data, Spectra_header = Spectra_lines.values, ','.join(Spectra_lines.columns.values)
    else:
        Spectra_lines = [str(i, encoding='utf-8') for i in Path.readlines()]
        header = Spectra_lines[0].split(',')[1:]
        Spectra_header = ','.join(["Wavenumber"] + ['Spectrum.'+i.split('/')[0].split('.')[-1] for i in header])
        Data = np.zeros((len(Spectra_lines)-1,len(Spectra_header.split(','))))
        for i in range(1,len(Spectra_lines)) :
            Line = Spectra_lines[i].split(',')
            for j in range(len(Line)) :
                if (Line[j] != '\n') and (Line[j] != '') and (Line[j] != '\r\n'): Data[i-1,j] = float(Line[j])
    sorted_arr = Data[np.argsort(Data[:, 0])]
    return sorted_arr, Spectra_header

def Open_bkg_GloveBox(Path) :
    if ('.series' in Path.name) or ('.Series' in Path.name)  :
        Spectra_lines = Open_series(Path)
        Data, Spectra_header = Spectra_lines.values, ','.join(Spectra_lines.columns.values)
    else:
        Spectra_lines = [str(i, encoding='utf-8') for i in Path.readlines()]
        header = Spectra_lines[0].split(',')[1:]
        Spectra_header = ','.join(["Wavenumber"] + ['Spectrum.'+i.split('/')[0].split('.')[-1] for i in header])
        Data = np.zeros((len(Spectra_lines)-1,len(Spectra_header.split(','))))
        for i in range(1,len(Spectra_lines)) :
            Line = Spectra_lines[i].split(',')
            for j in range(len(Line)) :
                if (Line[j] != '\n') and (Line[j] != '') and (Line[j] != '\r\n'): Data[i-1,j] = float(Line[j])
    sorted_arr = Data[np.argsort(Data[:, 0])]
    return sorted_arr, Spectra_header

def Open_spectrum_nano2(Path) :
    if '.irb' in Path.name :
        infos = StringIO(Path.getvalue().decode("utf-16")).read()
        params, spectra_info = re.split('<Table>|</Table>', infos)[0:2]
        file = pd.DataFrame(re.split('<double>|</double>', spectra_info)[1::2], columns=['Background'])        
        wn = np.linspace(float(re.split('<StartWavenumber>|</StartWavenumber>', params)[1]), float(re.split('<EndWavenumber>|</EndWavenumber>', params)[1]), file.shape[0])
        sorted_arr = pd.DataFrame([wn, [float(i.replace('<double>','').replace('</double>','')) for i in file.iloc[:,0]]]).T.values
        Spectra_header = ['cm-1'+re.split('<Units>|</Units>', params)[1]]
    else :
        Spectra_lines_read = StringIO(Path.getvalue().decode("utf-8")).read()
        Spectra_lines =[i.split(',') for i in Spectra_lines_read.split('\n')]
        Spectra_lines = pd.DataFrame(Spectra_lines)
        Spectra_lines.columns = Spectra_lines.iloc[0]
        Spectra_lines=Spectra_lines.drop(columns=[i for i in Spectra_lines.columns if 'deg' in i])[1:-1].astype(float)
        header = Spectra_lines.columns[1:]
        Spectra_header = ','.join(["Wavenumber"] + [i.split('/')[0].replace(' ','.') for i in header])
        sorted_arr = Spectra_lines.values
    return sorted_arr, Spectra_header
    
def uncorrect_background(Spectrum, Background):
    Uncorrected_Spectrum = deepcopy(Spectrum)
    Uncorrected_Spectrum[:,1] = Spectrum[:,1]*Background[:,1]
    return Uncorrected_Spectrum    

def Offset_background_correction(Spectrum, Background, Wv_nbr):
    i_Section = np.argmin(np.abs(Spectrum[:,0] - Wv_nbr))
    Offset, i_Offset = np.nanmin(Spectrum[i_Section:,1]), i_Section + np.argmin(Spectrum[i_Section:,1])
    x_Offset = Spectrum[i_Offset,0]
    Spect_offcorr = deepcopy(Spectrum)
    Spect_offcorr[:, 1] = Spect_offcorr[:, 1] - Offset
    Spectr_bkgcorr = deepcopy(Spect_offcorr)
    Spectr_bkgcorr[:,1] = Spect_offcorr[:,1]/Background[:,1]
    return Spectr_bkgcorr

def Stitching_peaces(Spectrum, X_Break, i_Break) :
    New_spectrum = deepcopy(Spectrum)
    beta = Spectrum[i_Break+1,1]/Spectrum[i_Break-1,1]
    Y_Break = Spectrum[i_Break+1,1]
    New_spectrum[:i_Break,1] = Spectrum[:i_Break,1]*beta
    New_spectrum[i_Break,1] = Y_Break
    return New_spectrum   

def Break_correction(Spectrum, X_Breaks, i_Break, n_delta) :
    Spectrum_stitched = deepcopy(Spectrum)
    for i, X_Break in enumerate(X_Breaks):
        Spectrum_stitched = Stitching_peaces(Spectrum_stitched, X_Break, i_Break[i])
    return Spectrum_stitched

@st.cache_data(ttl=3600, max_entries=1, show_spinner='Break correction')
def Spectrum_correction(Spec, Bkg, header_spec, _header_bkg, system, off_bkg, X_Breaks, n_delta=2, Bkg_divided = False):
    Spec_corrected = deepcopy(Spec)
    i_Break = [int(find_nearest_idx(Spec[:,0], i)) for i in X_Breaks]
    for i in np.arange(1, Spec.shape[1],1):
        if Bkg_divided: Spec[:,(0,i)] = uncorrect_background(Spec[:,(0,i)], Bkg)
        Spectr_bkgcorr = Offset_background_correction(Spec[:,(0,i)], Bkg, Wv_nbr=off_bkg)
        Spec_corrected[:,(0,i)] = Break_correction(Spectr_bkgcorr, X_Breaks, i_Break, n_delta)   
    return pd.DataFrame(Spec_corrected, columns=header_spec.split(',')).set_index(header_spec.split(',')[0])
#%% Fonctions plot

@st.cache_data(max_entries=5, show_spinner=False)
def plot_png(image, map_size, map_unit, height_px, width_px, origin):
    img  = px.imshow(cv2.cvtColor(image, cv2.COLOR_RGBA2BGR), origin=origin, labels=dict(x="x ("+map_unit+')', y="y ("+map_unit+')'), x=np.linspace(-float(map_size[0])/2, float(map_size[0])/2, image.shape[1]), y=np.linspace(-float(map_size[1])/2, float(map_size[1])/2, image.shape[0]), height=height_px, width=width_px)
    return img

@st.cache_data(max_entries=5, show_spinner=False)
def plot_txtcsv(image, color, map_size, map_unit, map_max, map_min, height_px, width_px, origin, colorbar_title):
    img  = px.imshow(image, origin=origin, color_continuous_scale=color, zmin=map_min, zmax=map_max, labels=dict(x="x ("+map_unit+')', y="y ("+map_unit+')', color=colorbar_title), x=np.linspace(-float(map_size[0])/2, float(map_size[0])/2, image.shape[1]), y=np.linspace(float(map_size[1])/2, -float(map_size[1])/2, image.shape[0]), height=height_px, width=width_px)
    return img

def color_selected_points(selected):
    st.session_state.tempo_df = pd.DataFrame([np.array([1]*len(selected)), sns.color_palette(st.session_state.choose_cmap, n_colors=len(selected)).as_hex()], index=['value', 'colors']).T
    st.bar_chart(st.session_state.tempo_df, y=['value'], color='colors', use_container_width=True, height=75, y_label=None)

def color_change() :
    if st.session_state.choose_cmap[0] == '#' : st.session_state.colors = [st.session_state.choose_cmap]
    else : st.session_state.colors = px.colors.sample_colorscale(st.session_state.choose_cmap, [n/(len(selected) -1) for n in range(len(selected))])
    st.session_state.positions['color'].loc[selected] = st.session_state.colors
    st.session_state.to_plot = selected

def no_color_change() :
    st.session_state.colors = st.session_state.positions['color'].loc[selected].values
    st.session_state.to_plot = selected

#%% Fonctions

def toast_appearance():
    st.markdown(
        """
        <style>
            div[data-testid=toastContainer] {
                padding: 1% 4% 65% 2%;
                align-items: center;}
        
            div[data-testid=stToast] {
                padding: 20px 10px 40px 10px;
                margin: 10px 400px 200px 10px;
                background-color: #CECECE;
                width: 20%;}
            
            [data-testid=toastContainer] [data-testid=stMarkdownContainer] > p {
                font-size: 20px; font-style: normal; font-weight: 400;
                foreground-color: #ffffff;}
        </style>
        """, unsafe_allow_html=True)

def get_positions(path):
    file_info = pd.read_xml(path)[['X', 'Y', 'Name', 'Size']]
    positions = file_info[['X', 'Y', 'Name']].dropna().reset_index().drop(columns='index'); img_info = re.split(', | ', file_info['Size'].dropna().values[0].replace('(','').replace(')',''))
    img_size = [img_info[0],img_info[2]]; img_unit = img_info[1]
    positions['Spectrum No'] = [float(i.split('.')[-1]) for i in positions['Name']]
    positions['X'] = positions['X'].str.replace(' um', '').astype(float)
    positions['Y'] = positions['Y'].str.replace(' um', '').astype(float)
    positions = positions.drop(columns='Name')
    positions['color'], positions['marker_style'] = 'blue', 'circle'
    return positions, img_size, img_unit

System_breaks_predefined = {'Nano2' : [1675,1477,1171], 'IconIR': [1706,1411,1209], 'Nano1' : [1712,1420,1100], 'Nano2S' : [1712,1420,1100], 'Mirage' : [1433,1205], 'GloveBox' : [1389,989]}

#%% Application

st.set_page_config(layout='wide')
st.title('IR spectra processing')
correctionTab, visuTab = st.tabs(["Laser breaks correction", "Visualisation of IconIR and GloveBox spectra on map"])

with correctionTab:
    c1,c2,c3,c4 = st.columns(4)
    with c1 : system = st.radio('Choose the system', ['GloveBox', 'IconIR', 'Mirage', 'Nano2'], key='system')
    with c2 : divided = st.radio('Spectra are already divided by the background', [True, False], key='divided')
    if system =='Mirage' :
        multiple_file=False
        c5,c6,c7,c8 = st.columns(4)
        with c6 : type_register = st.radio('Is it a hyperspectra file ?', ['Yes', 'No'], key='type_register')
        with st.form('Mirage_file', clear_on_submit=True) :
            if type_register == 'No' :
                with c7 : bkg_in_file = st.radio('Is the background in the same file as the spectra ?', ['Yes', 'No'], key='bkg_in_file',index=1)
                with c8 : organisation = st.radio('Is your file organised with wavenumber in a row or in a column ?', ['Row', 'Column'], key='organisation')        
                if bkg_in_file =='Yes' : st.session_state.spectra_files = st.file_uploader("Import your file", accept_multiple_files=False, type='csv'); st.session_state.bkg_files = 0
                else :
                    c9,c10 = st.columns(2)
                    with c9 : st.session_state.spectra_files = st.file_uploader("Import your IR file", accept_multiple_files=False, type=['csv', 'txt'])
                    with c10 : st.session_state.bkg_files = st.file_uploader("Import your background file", accept_multiple_files=False, type=['csv', 'txt'])
            else :
                st.session_state.bkg_in_file, st.session_state.organisation = 'No', 'Row' 
                c9,c10 = st.columns(2)
                with c9 : st.session_state.spectra_files = st.file_uploader("Import your IR file", accept_multiple_files=False, type=['csv', 'txt'])
                with c10 : st.session_state.bkg_files = st.file_uploader("Import your background file", accept_multiple_files=False, type=['csv', 'txt'])  
            st.form_submit_button('Submit')
                
    elif system =='Nano2' :
        multiple_file=False
        with  st.form('Nano2_file', clear_on_submit=True) :
            c5,c6 = st.columns(2)
            with c5 : st.session_state.spectra_files = st.file_uploader("Import your IR file", accept_multiple_files=False, type=['csv', 'txt'])
            with c6 : st.session_state.bkg_files = st.file_uploader("Import your background file", accept_multiple_files=False, type=['csv', 'irb', 'txt'])
            st.form_submit_button('Submit')

    else :
        with c3 : extension = st.radio('Select the extension typeof the SPECTRA files', ['series', 'txt/csv']).split('/')
        if extension==['series']: multiple_file=True
        else : multiple_file=False
        with  st.form('IconIR_file', clear_on_submit=True) :
            c5,c6 = st.columns(2)
            with c5 : st.session_state.spectra_files = st.file_uploader("Import your IR file", accept_multiple_files=multiple_file, type=extension)
            with c6 : st.session_state.bkg_files = st.file_uploader("Import your background file", accept_multiple_files=False, type=['series', 'txt', 'csv'])
            st.form_submit_button('Submit')
    
    if (st.session_state.spectra_files==None) or (st.session_state.bkg_files==None) : pass
    else :
        if system == 'GloveBox' or system == 'IconIR': st.session_state.Spec, st.session_state.header_spec = Open_spectrum_GloveBox(st.session_state.spectra_files, extension, multiple_file); st.session_state.Bkg, st.session_state.header_bkg = Open_bkg_GloveBox(st.session_state.bkg_files)
        elif system == 'Nano2': st.session_state.Spec, st.session_state.header_spec = Open_spectrum_nano2(st.session_state.spectra_files); st.session_state.Bkg, st.session_state.header_bkg = Open_spectrum_nano2(st.session_state.bkg_files)
        elif system == 'Mirage': st.session_state.Spec, st.session_state.Bkg, st.session_state.header_spec, st.session_state.header_bkg = Open_spectrum_mirage(st.session_state.spectra_files, st.session_state.bkg_files, st.session_state.type_register, st.session_state.bkg_in_file, st.session_state.organisation)    
        
        if len(st.session_state.Bkg)>st.session_state.Spec.shape[0] : st.session_state.Bkg = np.array([i for i in st.session_state.Bkg if i[0] in st.session_state.Spec[:,0]])

        with st.expander('Show the background'):
            fig_bkg = px.line(st.session_state.Bkg, x=0, y=1)
            fig_bkg.update_layout(title_text = 'Background', xaxis_title_text="Wavenumber (cm-1)",yaxis_title_text="Amplitude (mV)"); fig_bkg.update_xaxes(autorange="reversed")
            st.plotly_chart(fig_bkg)
        c_l, c_r = st.columns(2)
        
        with c_l:
            if system == 'GloveBox' :
                breaks_values = st.radio('What are the values of laser breaks (cm-1) ?', [str([1389,989]).replace(']', '').replace('[', ''), str([1402,989]).replace(']', '').replace('[', ''), 'Other'])
                if breaks_values == 'Other': st.session_state.breaks_values_enter = st.text_input('Please enter the value(s) (in cm-1). If multiple values, separate them with a comma.', value = str(System_breaks_predefined[system]).replace(']', '').replace('[', ''))
                else : st.session_state.breaks_values_enter = breaks_values
            elif system == 'IconIR' :
                breaks_values = st.radio('What are the values of laser breaks (cm-1) ?', [str([1706,1411,1209]).replace(']', '').replace('[', ''), str([1390,990]).replace(']', '').replace('[', ''), 'Other'])
                if breaks_values == 'Other': st.session_state.breaks_values_enter = st.text_input('Please enter the value(s) (in cm-1). If multiple values, separate them with a comma.', value = str(System_breaks_predefined[system]).replace(']', '').replace('[', ''))
                else : st.session_state.breaks_values_enter = breaks_values
            elif system == 'Mirage' :
                breaks_values = st.radio('What are the values of laser breaks (cm-1) ?', [str([1433,1205]).replace(']', '').replace('[', ''), 'Other'])
                if breaks_values == 'Other': st.session_state.breaks_values_enter = st.text_input('Please enter the value(s) (in cm-1). If multiple values, separate them with a comma.', value = str(System_breaks_predefined[system]).replace(']', '').replace('[', ''))
                else : st.session_state.breaks_values_enter = breaks_values
            elif system == 'Nano2' :
                breaks_values = st.radio('What are the values of laser breaks (cm-1) ?', [str([1675,1477,1171]).replace(']', '').replace('[', ''), 'Other'])
                if breaks_values == 'Other': st.session_state.breaks_values_enter = st.text_input('Please enter the value(s) (in cm-1). If multiple values, separate them with a comma.', value = str(System_breaks_predefined[system]).replace(']', '').replace('[', ''))
                else : st.session_state.breaks_values_enter = breaks_values
            st.session_state.breaks_values_use = np.array(re.split(',|, ', st.session_state.breaks_values_enter), float)        
        with c_r : off_bkg = st.number_input("Wavenumber of the background's offset", min_value=min(st.session_state.Spec[:, 0]), max_value=max(st.session_state.Spec[:, 0]), key='off_bkg')
        spectra_test_selection = st.selectbox('Spectra to use for the test',st.session_state.header_spec.split(',')[1:])      
        idx_selection = [idx for idx in range(len(st.session_state.header_spec.split(','))) if st.session_state.header_spec.split(',')[idx] == spectra_test_selection][0]
        st.session_state.spectra_test = Spectrum_correction(st.session_state.Spec[:, [0,idx_selection]], st.session_state.Bkg, 'Wavenumber (cm-1),After', st.session_state.header_bkg, st.session_state.system, st.session_state.off_bkg, st.session_state.breaks_values_use, n_delta=2, Bkg_divided= st.session_state.divided)    
        st.session_state.spectra_test['Before'] = st.session_state.Spec[:, idx_selection]        
        fig_before = px.line(st.session_state.spectra_test, color_discrete_sequence=['blue','red'])
        fig_before.update_layout(title_text = 'Before vs after break correction', xaxis_title_text="Wavenumber (cm-1)",yaxis_title_text="Amplitude (mV)"); fig_before.update_xaxes(autorange="reversed")
        st.plotly_chart(fig_before)
        c11, c12 = st.columns(2)
        with c11 : correct = st.button('Correct the laser break !')
        with c12 : container = st.container()
        if correct : st.session_state.spectra_corrected = Spectrum_correction(st.session_state.Spec, st.session_state.Bkg, st.session_state.header_spec, st.session_state.header_bkg, st.session_state.system, st.session_state.off_bkg, st.session_state.breaks_values_use, n_delta=2, Bkg_divided= st.session_state.divided)
        if 'spectra_corrected' in st.session_state :
            with container.form('Save_form') :
                st.text_input('Enter the file path :', key='savepath')
                if multiple_file==False : value = st.session_state.spectra_files.name.split('.csv')[0]
                else : value = st.session_state.spectra_files[0].name
                Name = st.text_input('Enter the name of the file', value=value)
                spectra_to_exclude = st.multiselect('Select spectra to exclude', [str(i) for i in st.session_state.spectra_corrected.columns], default=None, key='spectra_to_exclude')
                if st.form_submit_button('Save the corrected spectra'): st.session_state['spectra_corrected'].to_csv(st.session_state.savepath + '/' + Name +"_Break_Corrected.csv", columns=st.session_state.spectra_corrected.columns.drop(spectra_to_exclude), header=True, index=True, encoding='utf-8'); toast_appearance(); st.toast('Corrected spectra have been saved', icon=':material/check:', duration="infinite")
        if 'spectra_corrected' in st.session_state :
            fig = px.line(st.session_state.spectra_corrected,height=600)
            fig.update_layout(height=800, legend_title=None)
            fig.update_xaxes(autorange="reversed")
            st.plotly_chart(fig)

with visuTab :
    with st.expander('Upload necessary files', expanded=True):
        c1,c2 = st.columns(2) ; c3,c4 = st.columns(2)
        with c1:
            with st.form("Spectra", clear_on_submit=True):
                spectra_files = st.file_uploader("Import all spectra files", accept_multiple_files=True, type=['csv', 'series'], help="Upload your spectra file (.csv or .series). Don't upload multiple extension at the same type.")
                submitted = st.form_submit_button("Submit")
                if submitted and spectra_files is not None:
                    if spectra_files[0].name.endswith('csv'):
                        st.session_state.spec_file=[pd.read_csv(i, index_col=0) for i in spectra_files]
                        if len(st.session_state['spec_file']) > 1 : st.session_state.spectra_initial = pd.concat(st.session_state.spec_file, axis=1).T
                        else : st.session_state.spectra_initial = st.session_state.spec_file[0].T
                    else :
                        st.session_state.spectra_initial = pd.concat([Open_series(i) for i in spectra_files], axis=1).T.drop_duplicates(keep='first')
                        st.session_state.spectra_initial.columns=st.session_state.spectra_initial.iloc[0]
                        st.session_state.spectra_initial = st.session_state.spectra_initial.drop(labels = 'Wavenumber (cm-1)', axis=0)
                    if 'Spectrum.' in st.session_state.spectra_initial.index[0]: st.session_state.spectra_initial.index = [int(i.split('.')[1]) for i in st.session_state.spectra_initial.index]
                    else : st.session_state.spectra_initial.index = [int(re.sub(r"\D", "",i.split('.')[-1])) for i in st.session_state.spectra_initial.index]
                    st.session_state.spectra_initial = st.session_state.spectra_initial.sort_index()
                    st.session_state.spectra_initial.index.names = ['Spectrum No']
                    st.session_state.spectra = st.session_state.spectra_initial.copy()  
                    st.session_state.to_plot=np.array([]); st.session_state.spectra_submit=True
        with c2:
            with st.form("Positions", clear_on_submit=True):
                positions_files = st.file_uploader("Import all measurements files", accept_multiple_files=True, type='xml', help="Upload the measurements files corresponding to your spectra. They can be found in the different subfolder of the folder 'IRMeasurement', under the name '[subfolder-name].Measurement.xml'")
                submitted = st.form_submit_button("Submit")
                if submitted and positions_files is not None:
                    st.session_state.pos_file = [get_positions(i) for i in positions_files]
                    if len(st.session_state.pos_file) > 1 : st.session_state.positions = pd.concat([i[0] for i in st.session_state.pos_file], axis=0)
                    else : st.session_state.positions = st.session_state.pos_file[0][0]
                    st.session_state.positions = st.session_state.positions.set_index('Spectrum No')
                    st.session_state.positions = st.session_state.positions.sort_values('Spectrum No').drop_duplicates()
                    st.session_state.map_size, st.session_state.map_unit = st.session_state.pos_file[0][1:]
                    st.session_state.positions_submit=True         
        with c3 :
            with st.form("Topomap", clear_on_submit=True):
                topo_file = st.file_uploader("Import the topography", type=['txt','csv','png','tiff'])
                submitted = st.form_submit_button("Submit")
                if submitted and topo_file is not None:
                    if topo_file.name.endswith('.csv'): st.session_state['Topography'] = pd.read_csv(topo_file); st.session_state['Topography_image'] = False
                    elif topo_file.name.endswith('.png') : st.session_state['Topography'] = cv2.imdecode(np.asarray(bytearray(topo_file.read()), dtype=np.uint8), 1); st.session_state['Topography_image'] = True
                    elif topo_file.name.endswith('.tiff') or topo_file.name.endswith('.tif'): st.session_state['Topography'] = io.imread(topo_file); st.session_state['Topography_image'] = False
                    elif topo_file.name.endswith('.txt'): st.session_state['Topography'] = np.loadtxt(topo_file, delimiter=';'); st.session_state['Topography_image'] = False
                    st.session_state.topo_submit=True
        with c4 :
            with st.form("IRmap", clear_on_submit=True):
                IR_file = st.file_uploader("Import the IR map", type=['txt','csv','png','tiff'])
                submitted = st.form_submit_button("Submit")
                if submitted and IR_file is not None:
                    if IR_file.name.endswith('.png'): st.session_state['IR'] = cv2.imdecode(np.asarray(bytearray(IR_file.read()), dtype=np.uint8), 1); st.session_state['IR_image'] = True
                    elif IR_file.name.endswith('.csv'): st.session_state['IR'] = pd.read_csv(IR_file); st.session_state['IR_image'] = False
                    elif IR_file.name.endswith('.tiff') or IR_file.name.endswith('.tif'): st.session_state['IR'] = io.imread(topo_file); st.session_state['IR_image'] = False
                    elif IR_file.name.endswith('.txt'): st.session_state['IR'] = np.loadtxt(IR_file, delimiter=';'); st.session_state['IR_image'] = False
                    st.session_state.IR_submit=True
        st.write('Size of the map (topography and/or IR) : ',str(st.session_state.map_size[0]),'x',str(st.session_state.map_size[1]),' ',st.session_state.map_unit)
    
    if (st.session_state.topo_submit==False and st.session_state.IR_submit==False) or st.session_state.spectra_submit==False or st.session_state.positions_submit==False : pass
    else :
        if st.session_state.spectra_initial[st.session_state.spectra_initial.index.duplicated(keep=False)].reset_index().shape[0] != 0:
            if st.button('Reset the rows selection'): st.session_state.spectra = st.session_state.spectra_initial.copy()
        test = st.session_state.spectra[st.session_state.spectra.index.duplicated(keep=False)].reset_index()
        if test.shape[0] != 0:
            duplicated_index = test['index'].unique()
            st.session_state.selection = st.session_state.spectra_initial.loc[duplicated_index]
            st.data_editor(st.session_state.selection.reset_index(), disabled=True)
            selected_indices = st.multiselect('Choose the spectra to keep (one/same index)', st.session_state.selection.reset_index().index)
            if st.button('Selection is finished.'):
                st.session_state.spectra = st.session_state.spectra_initial.groupby(st.session_state.spectra_initial.index).mean()
                st.session_state.spectra.loc[duplicated_index] = st.session_state.selection.iloc[selected_indices]
        try : st.session_state.positions=st.session_state.positions.loc[st.session_state.spectra.index.values]
        except KeyError : st.error("Your positions doesn't seems to match with the spectra. Re-upload the matching measurement file."); st.stop()
        else : st.session_state.positions=st.session_state.positions.loc[st.session_state.spectra.index.values]
        # Plots parameters and spectra operation
        with st.sidebar :
            with st.popover('See all the different colorscales'): st.write(px.colors.sequential.swatches_continuous()); st.write(px.colors.diverging.swatches_continuous())
            st.header('Plot parameters')
            if (st.session_state.IR_submit==True) and (st.session_state.topo_submit==False) : default_image='IR'
            elif st.session_state.topo_submit==True : default_image='Topography'
            st.select_slider("Map to use:", ["Topography", "IR"], key='map', value = default_image)
            with st.expander('Map aspect'):
                c_w, c_h = st.columns(2, vertical_alignment='bottom')
                with c_w : width_px = st.number_input('Width in pixels of the map', min_value=10, value=800, key='width_px')
                with c_h : height_px = st.number_input('Height in pixels of the map', min_value=10, value=800, key='height_px')
                if st.session_state[st.session_state.map+'_image']==False:
                    with c_w : map_max = st.number_input('Upper limit of the colorbar', value=np.nanmax(st.session_state[st.session_state.map]), key='map_max')
                    with c_h : map_min = st.number_input('Lower limit of the colorbar', value=np.nanmin(st.session_state[st.session_state.map]), key='map_min')
                st.radio('Select the origin of the map', ['upper', 'lower'], horizontal=True, key='origin')
                st.number_input('Marker size', min_value=1, value=8, step=1, key='marker_size')
                st.multiselect('Choose the markestyle(s) to use', symbols_names, key='marker_select', default='circle')
                for i in range(len(st.session_state.marker_select)): st.session_state.positions['marker_style'].iloc[i::len(st.session_state.marker_select)]=st.session_state.marker_select[i]
            with st.expander('Plot and line(s) aspect'):
                c_e1, c_e2  = st.columns(2)
                width_spec = c_e1.number_input('Width in pixels of the spectra graph', min_value=10, value=800, key='width_spec')
                xleft = c_e1.number_input('Select the left x boundary', value=np.nanmax(st.session_state.spectra.columns), key='xleft')
                ybottom = c_e1.number_input('Select the bottom y boundary', value=0., step=0.01, key='ybottom')
                xitcks_step = c_e1.number_input("x ticks' step (cm-1)", min_value=1, value=200, key="x_ticks_step")
                height_spec = c_e2.number_input('Height in pixels of the spectra graph', min_value=10, value=600, key='height_spec')
                xright = c_e2.number_input('Select the right x boundary', value=np.nanmin(st.session_state.spectra.columns), key='xright')
                ytop = c_e2.number_input('Select the top y boundary', value=np.nanmax(np.ma.masked_where(st.session_state.spectra == np.inf, st.session_state.spectra)), step=0.01, key='ytop')
                legend_show = c_e2.radio('Show legend', [True, False], index=0, key='legend_show', horizontal=True)
                if legend_show==True:
                    leg_orient = c_e1.radio('Legend orientation', ['h', 'v'], format_func=lambda x: {'h': "Horizontal",'v': "Vertical"}.get(x), key='leg_orient')
                    if leg_orient=='h': c_e2.number_input('Legend position in y', value=-0.75, key='y_leg')         
                offset = st.number_input('Offset between each spectrum', min_value=0., value=0., key='offset',step=1.,format="%.2f")
                c_color1, c_color2 = st.columns(2)
                with c_color1 :
                    bkg_color = st.color_picker('Background plot color', value='#FFFFFF', key = 'bkg_color')
                    grid_color = st.color_picker('Grid line color', value='#9E9E9E', key = 'grid_color')
                with c_color2 :
                    font_color = st.color_picker('Font color', value='#000000', key = 'font_color')
                    color_container = st.container()
                st.multiselect('Choose the linestyle(s) to use', ls_option, 'solid', key='line_styles_option')
                st.toggle('Add the marker of the corresponding positions in the legend.', key='marker_appear')          
            with st.expander('Spectra selection'):
                st.radio('Select spectra by :', ['Range', 'Multiselection', 'Selection on map', 'All'], key='selection_tool')
                # Range selection
                if st.session_state.selection_tool=='Range' :                    
                    df_range = pd.DataFrame({'Range' : 1, 'StartSpec' : 0, 'EndSpec' : 1}, index=[1]).set_index('Range')
                    with st.popover('Enter range(s) here', width='stretch', help='Add and remove range of spectra to select (ex: spectra 1 to 10 in range 1 and 20 to 40 in range 2)').container(width=400):
                        df_range_edit = st.data_editor(df_range, num_rows="dynamic",
                        column_config={"Range": st.column_config.NumberColumn('Range n°', min_value=1,step=1),
                        'StartSpec' : st.column_config.NumberColumn('From spectra n°', min_value=int(st.session_state.spectra.index[0]), max_value=int(st.session_state.spectra.index[-1]), step=1, required=True),
                        'EndSpec' : st.column_config.NumberColumn('to spectra n°', min_value=int(st.session_state.spectra.index[0]), max_value=int(st.session_state.spectra.index[-1]), step=1, required=True)}, hide_index=True)
                    choose_cmap = st.selectbox('Colorscale for the spectra/marker of the selected positions (_r is the reversed)', st.session_state.colorscales, key='choose_cmap')
                    c3, c4 = st.columns([0.7,0.3])
                    with c3 :
                        if st.button('Change colors and plot', width='stretch') :
                            selected = np.hstack([np.arange(df_range_edit.to_dict()['StartSpec'][i], df_range_edit.to_dict()['EndSpec'][i]+1) for i in df_range_edit.index])
                            selected = st.session_state.positions.index.intersection(selected)
                            st.session_state.colors = px.colors.sample_colorscale(st.session_state.choose_cmap, [n/(len(selected) -1) for n in range(len(selected))])
                            st.session_state.positions['color'].loc[selected] = st.session_state.colors
                            st.session_state.to_plot = selected
                    with c4 :
                        if st.button('Plot', help="This button doesn't change the color of the spectra", width='stretch') :
                            selected = np.hstack([np.arange(df_range_edit.to_dict()['StartSpec'][i], df_range_edit.to_dict()['EndSpec'][i]+1) for i in df_range_edit.index])
                            selected = st.session_state.positions.index.intersection(selected)
                            st.session_state.colors = st.session_state.positions['color'].loc[selected].values
                            st.session_state.to_plot = selected
                # Multiselection
                elif st.session_state.selection_tool=='Multiselection' :
                    mutliselect_spectra = np.array(st.multiselect('Spectra n°', st.session_state.spectra.index))
                    choose_cmap = st.selectbox('Colorscale for the spectra/marker of the selected positions (_r is the reversed)', st.session_state.colorscales, key='choose_cmap')
                    c3, c4 = st.columns([0.7,0.3])
                    with c3 :
                        if st.button('Change colors and plot') :
                            selected = mutliselect_spectra
                            selected = st.session_state.positions.index.intersection(selected)
                            st.session_state.colors = px.colors.sample_colorscale(st.session_state.choose_cmap, [n/(len(selected) -1) for n in range(len(selected))])
                            st.session_state.positions['color'].loc[selected] = st.session_state.colors
                            st.session_state.to_plot = selected
                    with c4 :
                        if st.button('Plot', help="This button doesn't change the color of the spectra") :
                            selected = mutliselect_spectra
                            selected = st.session_state.positions.index.intersection(selected)
                            st.session_state.colors = st.session_state.positions['color'].loc[selected].values
                            st.session_state.to_plot = mutliselect_spectra
                # Map selection
                elif st.session_state.selection_tool=='Selection on map' : selection_container = st.container(key='selection_container')
                # Select all
                elif st.session_state.selection_tool=='All' :
                    choose_cmap = st.selectbox('Colorscale for the spectra/marker of the selected positions (_r is the reversed)', st.session_state.colorscales, key='choose_cmap')
                    c3, c4 = st.columns([0.7,0.3])
                    with c3 :
                        if st.button('Change colors and plot') :
                            selected = st.session_state.positions.index
                            st.session_state.colors = px.colors.sample_colorscale(st.session_state.choose_cmap, [n/(len(selected) -1) for n in range(len(selected))])
                            st.session_state.positions['color'].loc[selected] = st.session_state.colors
                            st.session_state.to_plot = selected
                    with c4 :
                        if st.button('Plot', help="This button doesn't change the color of the spectra") :
                            selected = st.session_state.positions.index
                            st.session_state.colors = st.session_state.positions['color'].loc[selected].values
                            st.session_state.to_plot = selected

            with st.expander('Operations on spectra'):
                st.write('Normalization'); st.radio('Normalization', ['None', 'Divided by max amplitude', 'Vectorial normalization'], key='normalization', label_visibility='collapsed')
                if st.session_state.normalization == 'Divided by max amplitude' : wn_norm = st.number_input('Enter a wavenumber', min_value = min(st.session_state.spectra.columns), max_value = max(st.session_state.spectra.columns), step=st.session_state.spectra.columns[1]-st.session_state.spectra.columns[0], key='wn_norm')
                st.divider()
                st.toggle('Savitsky-Golay filter', key='savgol_operation')
                if st.session_state.savgol_operation : st.session_state.win_len = st.number_input('Window length', min_value=3, step=2, value=11); st.session_state.polyorder = st.number_input('Polynome order', min_value=1, step=1, value=2); st.session_state.deriv = st.number_input('Derivation order', min_value=0, step=1)
                st.divider()
                mean = st.toggle('Mean of selected spectrum'); st.divider()
                st.toggle('Ratio analysis', key='ratio_analysis')
                if st.session_state.ratio_analysis :
                    ratio_cmap = st.selectbox('Colorscale for the ratio (_r is the reversed)', st.session_state.colorscales, key='ratio_cmap'); st.number_input('Wavenumber 1 (cm-1)', min_value=np.nanmin(st.session_state.spectra.columns), max_value=np.nanmax(st.session_state.spectra.columns), key='wn_1'); st.number_input('Wavenumber 2 (cm-1)', min_value=np.nanmin(st.session_state.spectra.columns), max_value=np.nanmax(st.session_state.spectra.columns), key='wn_2')
                    with st.popover('Select spectra for the ratio') : select_ratio = st.multiselect('Select spectra for the ratio', [int(i) for i in st.session_state.positions.index], default=[int(i) for i in st.session_state.positions.index])
        ########################################
        # Topography map
        if st.session_state.map=='Topography' :
            if st.session_state.Topography_image==True : img = plot_png(st.session_state.Topography, st.session_state.map_size, st.session_state.map_unit, st.session_state.height_px, st.session_state.width_px, st.session_state.origin)
            else : img = plot_txtcsv(st.session_state.Topography,'YlOrBr_r', st.session_state.map_size, st.session_state.map_unit, st.session_state.map_max, st.session_state.map_min, st.session_state.height_px, st.session_state.width_px, st.session_state.origin, 'Height (nm)')
        # IR map
        elif st.session_state.map=='IR' :
            if st.session_state.IR_image==True : img = plot_png(st.session_state.IR, st.session_state.map_size, st.session_state.map_unit, st.session_state.height_px, st.session_state.width_px, st.session_state.origin)
            else : img = plot_txtcsv(st.session_state.IR,'hot', st.session_state.map_size, st.session_state.map_unit, st.session_state.map_max, st.session_state.map_min, st.session_state.height_px, st.session_state.width_px, st.session_state.origin, 'IR signal')
        # Ratio analysis
        if st.session_state.ratio_analysis :
            # & Savitsky-Golay
            if st.session_state.savgol_operation :
                ratio_savgol = pd.DataFrame((savgol_filter(st.session_state.spectra, st.session_state.win_len, st.session_state.polyorder, st.session_state.deriv)), index=st.session_state.dots_ratio.index, columns=st.session_state.dots_ratio.columns)
                st.session_state.z = ratio_savgol[st.session_state.wn_1]/ratio_savgol[st.session_state.wn_2]
            else : st.session_state.z = st.session_state.spectra[st.session_state.wn_1]/st.session_state.spectra[st.session_state.wn_2]
            dots = img.add_scatter(x=st.session_state.positions['X'].loc[select_ratio], y=st.session_state.positions['Y'].loc[select_ratio], mode='markers', marker_size=st.session_state.marker_size, marker_line_width=1, marker_line_color='black', uirevision=True, hovertext=st.session_state.positions.index, hovertemplate= '%{text}', text  = ['Spectrum n° {} : {}'.format(int(i), round(st.session_state.z.loc[i], 3)) for i in select_ratio], marker_symbol=st.session_state.positions['marker_style'].loc[select_ratio], marker=dict(color = st.session_state.z.loc[select_ratio], colorscale=st.session_state.ratio_cmap, colorbar=dict(x=+1.2, title='Ratio')))
            dots.update_layout(hovermode='closest')
            if st.session_state.selection_tool=='Selection on map' : selected_points = plotly_events(img, select_event=True, override_height=height_px)
            else : plotly_events(dots, False, False, override_height=height_px)       

        # No ratio analysis
        else :
            dots = img.add_scatter(x=st.session_state.positions['X'], y=st.session_state.positions['Y'], mode='markers', marker_size=st.session_state.marker_size, marker_color=st.session_state.positions['color'], marker_line_width=1, marker_line_color='black', uirevision=True, hovertext=st.session_state.positions.index, marker_symbol=st.session_state.positions['marker_style'], name='positions')
            dots.update_layout(hovermode='closest')
            if st.session_state.selection_tool=='Selection on map' : selected_points = plotly_events(img, select_event=True, override_height=height_px)
            else : st.plotly_chart(dots, use_container_width=False)       
        
        # Spectra selection on map
        if st.session_state.selection_tool=='Selection on map' :
            if selected_points!=[] :
                try : selected = img.to_dict()["data"][1]['hovertext'][np.array([int(i['pointNumber']) for i in selected_points])]
                except TypeError : pass
                else :    
                    with selection_container:
                        if len(selected)==1 : choose_cmap = st.color_picker('Pick a color', key='choose_cmap')
                        else : choose_cmap = st.selectbox('Colorscale for the spectra/marker of the selected positions (_r is the reversed)', st.session_state.colorscales, key='choose_cmap')
                        c3, c4 = st.columns([0.7, 0.3])
                        with c3 : st.button('Plot and change colors', on_click=color_change, key='plot_colors')
                        with c4 : plot_nocolors = st.button('Plot', on_click=no_color_change, key='plot_nocolors', help="This button doesn't change the color of the spectra")

        if st.session_state.to_plot.size != 0 :
            if 'colors' not in st.session_state : st.session_state.colors = st.session_state.positions.loc[st.session_state.to_plot]['color'].values
            # Max amplitude normalization
            if st.session_state.normalization == 'Divided by max amplitude' :
                st.session_state.spectra_norm = (st.session_state.spectra.T/st.session_state.spectra[st.session_state.wn_norm]).T            
                st.session_state.df_toplot=st.session_state.spectra_norm.loc[st.session_state.to_plot].T
            # Vectorial normalization
            elif st.session_state.normalization == 'Vectorial normalization' :
                st.session_state.spectra_norm = pd.DataFrame(normalize(st.session_state.spectra, norm='l2', axis=1), index = st.session_state.spectra.index, columns=st.session_state.spectra.columns)
                st.session_state.df_toplot=st.session_state.spectra_norm.loc[st.session_state.to_plot].T
            # No normalization
            else : st.session_state.df_toplot=st.session_state.spectra.loc[st.session_state.to_plot].T
            # Mean of the selected spectra
            if mean :
                color_container.color_picker('Color of the mean spectrum and std', key = 'line_color')
                if st.session_state.savgol_operation : st.session_state.df_final = savgol_filter(st.session_state.df_toplot.mean(axis=1), st.session_state.win_len, st.session_state.polyorder, st.session_state.deriv); st.session_state.df_final_std = savgol_filter(st.session_state.df_toplot.std(axis=1), st.session_state.win_len, st.session_state.polyorder, st.session_state.deriv)
                else : st.session_state.df_final = st.session_state.df_toplot.mean(axis=1); st.session_state.df_final_std = st.session_state.df_toplot.std(axis=1)
                X = list(st.session_state.df_toplot.index.values)
                spectra = go.Figure([go.Scatter(x=X,y=st.session_state.df_final,line_color=st.session_state.line_color,mode='lines', showlegend=False), 
                go.Scatter(x=X+X[::-1], y=list(st.session_state.df_final+st.session_state.df_final_std)+list(st.session_state.df_final[::-1]-st.session_state.df_final_std[::-1]),
                    fill='toself', fillcolor=st.session_state.line_color, opacity=0.5,line=dict(color='rgba(255,255,255,0)'),hoverinfo="skip",showlegend=False)])
            # Selected spectra
            else :
                # Savitsky-Golay
                if st.session_state.savgol_operation : st.session_state.df_final = pd.DataFrame((savgol_filter(st.session_state.df_toplot.T, st.session_state.win_len, st.session_state.polyorder, st.session_state.deriv).T), index=st.session_state.df_toplot.index, columns=st.session_state.df_toplot.columns)
                # Raw spectra
                else : st.session_state.df_final = st.session_state.df_toplot
                # Spectra offset
                if st.session_state.offset != 0:
                    for i in range(np.shape(st.session_state.df_toplot)[1]):st.session_state.df_toplot.iloc[:,i]=st.session_state.df_toplot.iloc[:,i] + i*st.session_state.offset
                ##### Line and marker style
                line_styles = cycle(st.session_state.line_styles_option)
                st.session_state.marker_to_plot=st.session_state.positions.loc[st.session_state.df_toplot.columns.values, 'marker_style']
                if st.session_state.marker_appear : mark_app, leg_itm_size = True, 'constant'
                else : mark_app, leg_itm_size = False, 'trace'                            
                ###########################
                spectra = px.line(st.session_state.df_final, color_discrete_sequence = st.session_state.colors, height=st.session_state.height_spec, width=st.session_state.width_spec, markers=mark_app)
                if st.session_state.legend_show == True :
                    if st.session_state.leg_orient=='h' : spectra.update_layout(legend=dict(title=None, xanchor="left", yanchor="bottom", itemsizing=leg_itm_size, orientation='h', y=st.session_state.y_leg, x=0, indentation=0, entrywidth=35))
                    else : spectra.update_layout(legend=dict(title=None, yanchor="top", itemsizing=leg_itm_size, orientation='v', indentation=0, entrywidth=35))
                else : spectra.update_layout(showlegend=False)
                spectra.update_traces(marker=dict(size=0.1))
                for d in spectra.data : d.line["dash"], d.marker['symbol'] = next(line_styles), st.session_state.marker_to_plot.loc[int(d.name)]
            
            spectra.update_layout(template=None, margin= {'l': 70,'r': 1, 't': 0}, xaxis_title='Wavenumber [cm-1]',yaxis_title='Amplitude [a.u.]', paper_bgcolor=st.session_state.bkg_color, plot_bgcolor=st.session_state.bkg_color, font_color=st.session_state.font_color,yaxis_gridcolor=st.session_state.grid_color, xaxis_gridcolor=st.session_state.grid_color, yaxis_zerolinecolor=st.session_state.grid_color, height=st.session_state.height_spec, width=st.session_state.width_spec)
            spectra.update_xaxes(dtick=st.session_state.x_ticks_step, range=[xleft, xright], ticks='outside', title_standoff = 0, gridcolor=st.session_state.grid_color, tickcolor=st.session_state.grid_color, zeroline=True, zerolinecolor=st.session_state.grid_color); spectra.update_yaxes(range=[ybottom, ytop], gridcolor=st.session_state.grid_color, ticks='outside', tickcolor=st.session_state.grid_color)
            # Plot the spectra
            st.plotly_chart(spectra, use_container_width=False)
                        
            with st.container() :
                savepath = st.text_input('Enter the file path :')
                Name = st.text_input('Enter the name of the file')
                if st.session_state.normalization != 'None' : st.write('The current parameters are :\n','- Applied spectra offset : ', str(st.session_state.offset),'\n- Normalized by : ', st.session_state.normalization)
                else : st.write('The current parameters are :\n','- Applied spectra offset : ', str(st.session_state.offset),'\n- Not normalized')
                cn, cm, cmall, cfigMAP, cfigIR = st.columns(5)
                if st.session_state.normalization == 'Vectorial normalization' : st.session_state.type_norm = 'Vectorial'
                elif st.session_state.normalization == 'Divided by max amplitude' : st.session_state.type_norm = str(st.session_state.wn_norm)
                with cn :
                    if st.button('Save the normalized spectra'): st.session_state.spectra_norm.to_csv(savepath +'/' + Name + '_Normalized__'+ st.session_state.type_norm +'_offset_' + str(st.session_state.offset) + '.csv', header=True, index=True, encoding='utf-8')
                with cm :
                    if st.button('Save the normalized mean and std of the selected spectra'):
                        file_to_save = pd.concat([st.session_state.df_toplot.mean(axis=1), st.session_state.df_toplot.std(axis=1)], axis=1)
                        file_to_save['Selected spectrum'] = pd.Series(st.session_state.df_toplot.columns.values, index=file_to_save.index[:len(st.session_state.df_toplot.columns.values)])
                        file_to_save.to_csv(savepath + '/' + Name + '_Normalized__'+ st.session_state.type_norm +"_Mean_Std.csv", header=['Mean', 'Standard deviation', 'n° of the selected spectrum'], index=True, encoding='utf-8')
                with cmall :
                    if st.button('Save the normalized mean and std for all spectra'): pd.concat([st.session_state.spectra_norm.mean(axis=0), st.session_state.spectra_norm.std(axis=0)], axis=1).to_csv(savepath + '/' + Name + '_Normalized__'+ st.session_state.type_norm +"_Mean_Std_all.csv", header=['Mean', 'Standard deviation'], index=True, encoding='utf-8')
                with cfigMAP:
                    if st.button('Save map figure as html') : dots.write_html(savepath +'/' + Name + '_'+ st.session_state.map +'_map' + '.html')
                with cfigIR:
                    if st.button('Save IR spectra figure as html') : spectra.write_html(savepath +'/' + Name + '_spectra' + '.html')