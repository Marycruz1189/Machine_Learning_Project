import streamlit as st
from PIL import Image
import pandas as pd
import os 
import io
import requests
import sys
from tensorflow.python.keras.preprocessing.image_dataset import load_image 
import torch
import cv2
import numpy as np 
from detecto import core, utils, visualize
from torchvision import transforms
from detecto.core import Model
from detecto.visualize import detect_video
from detecto.utils import reverse_normalize, normalize_transform, _is_iterable
from torchvision import transforms

import ffmpy

import matplotlib.pyplot as plt
dir = os.path.dirname
path = os.path.abspath(__file__)
src_path = dir(dir(os.path.abspath(__file__)))

#sys.path.append(src_path)

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import utils.visualization_tb as vis
st.set_option('deprecation.showPyplotGlobalUse', False)

def configuracion():
    st.set_page_config(page_title='Waste Management', page_icon=':electric_plug:', layout="wide")

def menu_inicio():
    
    st.header('Bienvenidos al sistema de clasificación y detección por tipos de residuos sólidos')
    fullpath1 = (dir(src_path) + os.sep+ "resources" + os.sep + "separacion-residuos.jpg")
    image = Image.open(fullpath1)
    st.image (image,use_column_width=True)
    with st.beta_expander("De que me hablas?"):
        st.write("""
            OBJETIVO:
            Implementar un sistema de reconocimiento 
            automatizado que utiliza un algoritmo de aprendizaje 
            profundo para clasificar los desechos por tipo de residuo. 
            La segregación eficiente de los desechos sólidos ayuda a
            reducir la cantidad de dispuestos incorrectamente, lo que 
            mejora la tasa de reciclaje y protege el suelo de la 
            contaminación""")
   
    st.write("""
            PERFIL DE LA AUTORA:
            Mary Cruz Meza Rivas
            Investigadora en Residuos sólidos y data scientist""" )


def menu_visualization(): 
    st.subheader('DESCRIPCIÓN DE LA BASE DE DATOS') 
    fullpath1 = (dir(src_path) + os.sep+ "reports" + os.sep + "muestra_data.png")
    image = Image.open(fullpath1)
    st.image (image,use_column_width=True)
    

def menu_api(): 
    url= "http://localhost:6060/token_id?token_id=Y4290783D"
    json_api = requests.get(url).json()
    st.write(json_api)


def opciones_filtros():
    st.sidebar.subheader("Filtros:")
    #clasificacion = st.sidebar.selectbox(
        #'Selecciona el modelo de clasificación que te interese:',
        #options= ['cnn', 'resnet34', 'resnetV2', 'vgg16'])

    detector = st.sidebar.selectbox(
        'Selecciona el detector  que te interese:',
        options= ['imagen', 'video'])
       
    return detector
def detect_video(model, input_file, output_file, fps=30, score_filter=0.6):
    video = cv2.VideoCapture(input_file)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scaled_size = 800
    scale_down_factor = min(frame_height, frame_width) / scaled_size
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height))
    transform_frame = transforms.Compose([  
        transforms.ToPILImage(),
        transforms.Resize(scaled_size),
        transforms.ToTensor(),
        normalize_transform(),
    ])
    while True:
        ret, frame = video.read()
        if not ret:
            break
        transformed_frame = frame  
        predictions = model.predict(transformed_frame)
        for label, box, score in zip(*predictions):
            if score < score_filter:
                continue        
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)      
            cv2.putText(frame, '{}: {}'.format(label, round(score.item(), 2)), (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        out.write(frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    video.release()
    out.release()
    cv2.destroyAllWindows()


path_model = src_path  + os.sep + "models" + os.sep
#path_save = src_path  + os.sep + "reports" + os.sep + 'output_vid.avi'

def mostrar_video(archivo):
    cap = cv2.VideoCapture(archivo)
    ret, frame = cap.read()
    while(1):
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
            cap.release()
            cv2.destroyAllWindows()
            break
        cv2.imshow('frame',frame)

def subir_img(path_model):
    st.subheader('DETECCIÓN DE IMAGENES POR TIPO DE RESIDUO SÓLIDO') 
    st.sidebar.subheader("Filtros:")
    detector = st.sidebar.selectbox(
        'Selecciona el detector  que te interese:',
        options= ['imagen', 'video'])
    if detector == 'imagen':
        slide_img = st.file_uploader("Sube imagen", type=["png", "jpg", "jpeg"])
        if slide_img is not None:
            img = Image.open(slide_img)
            #st.image(img, caption='Uploaded Image.', use_column_width=True)
        #image = Image.open(slide_img)
            if st.button("Predict"): 
                model_detecto = core.Model.load(path_model +'model_2_data_aug.pth', ['battery', 'biological', 'cardboard', 'glass', 'metal', 'paper', 'plastic'])
                #imagen = utils.read_image(image)
                predictions = model_detecto.predict(img)
                labels, boxes, scores = predictions
                vision = (visualize.show_labeled_image(img, boxes, labels)) 
                st.pyplot(vision, figsize=(5, 5), caption='Detección de residuos')
            
    else:
        slide_video = st.file_uploader("Sube video", type=["mp4"])
        #temporary_location = False
        if slide_video is not None:
            #video = Image.open(slide_video)
            #g = io.BytesIO(slide_video.read())  ## BytesIO Object
            #temporary_location = "testout_simple.mp4"
            if st.button("Predict"): 
                model_detecto = core.Model.load(path_model +'model_2_data_aug.pth', ['battery', 'biological', 'cardboard', 'glass', 'metal', 'paper', 'plastic'])
                detect_video(model_detecto, slide_video, 'output_vid.avi')
                #clip = moviepy.VideoFileClip("output_vid.avi")
                #clip.write_videofile("output_vid.mp4")
                #ff = ffmpy.FFmpeg(
                #inputs={'inpit_vid.avi': None},
                #outputs={'output.mp4': None}
                # )
                #ff.run()

                video_file = open("output_vid.avi", 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)












