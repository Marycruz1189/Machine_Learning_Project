import pandas as pd 
import os 
import matplotlib as plt 



def concatenar(dataframe1, dataframe2):
    """
    Une las bases de datos de train y test 
    """
    dataframe = pd.concat([dataframe1,dataframe2], ignore_index=True)
    return dataframe 


def guardar_plot (path, imgname):
    """
    Guarda la imagen con formato .PNG en el directorio indicado
    """
    results_dir = os.path.join(path) 
    if not os.path.isdir(results_dir): 
        os.makedirs(results_dir) 
    plt.savefig(results_dir + imgname,bbox_inches='tight') 


def clean_data_json (path, imgname):
    """
    selecciona la información para ser exportada en el json
    """
def import_imag(path_train,dataframe):
    imagenes = []
    for i in dataframe.path:
        x = path_train + i
        imagenes_rs = cv2.imread(x, 1)
        imagenes.append(cv2.resize(imagenes_rs, (224, 224)))
    dataframe["nombre_img"] = imagenes   
    return dataframe


def colsize(dataframe):
 dataframe['size'] = '(224,224)'
 return dataframe

def json_df(dataframe):
    dataframe = dataframe.loc[:,['img','size','label']]
    return dataframe 


