import streamlit as st
from PIL import Image
import pandas as pd
import os
import numpy as np 
from detecto import core, utils, visualize
from torchvision import transforms
from detecto import core, utils, visualize
import sys
dir = os.path.dirname
import requests
import cv2
dir = os.path.dirname
src_path = dir(dir(os.path.abspath(__file__)))
sys.path.append(src_path)
from utils import dashboard_tb as dash
#import utils.dashboard_tb as dash
path = os.path.abspath(__file__)

st.set_option('deprecation.showPyplotGlobalUse', False)

dash.configuracion()

menu = st.sidebar.selectbox('Menu:',
                            options=["Welcome", "Visualization", "Flask_API" , "Model Prediction", "“Models From SQL Database”" ])

st.title("CLASIFICACIÓN Y DETECCIÓN DE RESIDUOS SÓLIDOS")

path_model = src_path  + os.sep + "models" + os.sep

if menu == 'Welcome':
    dash.menu_inicio()

if menu == 'Visualization':
    dash.menu_visualization()

if menu == 'Flask_API':
    dash.menu_api()

if menu == 'Model Prediction':
    dash.subir_img(path_model)

