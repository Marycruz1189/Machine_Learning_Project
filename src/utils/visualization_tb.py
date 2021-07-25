import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image



def plot_img (train_ds):
    plt.figure(figsize=(20, 10))
    for images, labels in train_ds.take(1):
        for i in range(18):
            ax = plt.subplot(3, 6, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(train_ds.class_names[labels[i]])
            plt.axis("off")

def comparativa():
    resultados = {'modelo': ['CNN', 'ResnetV2', 'Vgg16', 'Resnet34' ], 'accuracytrain': ['0.93','0.20','0.92','0.95',], 'accuracytest': ['0.44', '0.16', '0.60', 'Nan'] }

    comparativa = pd.DataFrame(resultados)
    return comparativa

def spent_time():
    time = pd.DataFrame({'Time': [0.01, 0.60 , 0.30, 0.09 ],
                   'Task': ['Buscar base de datos', 'Entrenamiento de model-clasi', 'Entrenamiento de model-detec', 'Flask/Streamlit']})
    plot = time.plot.pie(y='Time', autopct='%1.2f%%',shadow=True, figsize=(5, 5))
    labels = 'Buscar base de datos', 'Entrenamiento de model-clasi', 'Entrenamiento de model-detec', 'Flask/Streamlit'
    plt.legend(labels, bbox_to_anchor=(1,0), loc="lower right", 
                          bbox_transform=plt.gcf().transFigure)
    plt.show()
    return plot



