import pandas as pd
import glob
import os 
import cv2


def make_imag_df(path):
    dictionary = {}
    for paths in glob.glob(path):
        dictionary.setdefault('set', []).append(paths.split('\\')[-3])
        dictionary.setdefault('name', []).append(paths.split('\\')[-2])
        dictionary.setdefault('id', []).append(paths.split('\\')[-1][:-4])
        dictionary.setdefault('img', []).append(paths.split('\\')[-1])
       
        
    images = pd.DataFrame(dictionary)
    return images


def creat_colum(dataframe):
    dataframe['path']= dataframe.name + "/" + dataframe.img
    dataframe['label'] = dataframe['name'].map({'battery': 0, 'biological': 1, 'cardboard': 2, 'glass':3, 'metal':4, 'paper':5, 'plastic':6})
    
    return dataframe

def import_imag(path_train,dataframe):
    imagenes = []
    
    for i in dataframe.path:
        x = path_train + i
        imagenes_rs = cv2.imread(x, 1)
        imagenes.append(cv2.resize(imagenes_rs, (224, 224)))
        
    dataframe["nombre_img"] = imagenes   
    
    return dataframe