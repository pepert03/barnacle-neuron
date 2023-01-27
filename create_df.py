import pandas as pd
import numpy as np
import os
import random as rd
import matplotlib.pyplot as plt


def create_df():
    df = pd.DataFrame(columns=['data', 'label'])
    for folder in os.listdir('data'):
        if folder != 'data.csv':
            for file in os.listdir('data/' + folder):
                img = plt.imread('data/' + folder + '/' + file)
                gray = (np.mean(img, axis=2).reshape(1,-1)*255).astype(np.uint8).tolist()[0]
                df.loc[len(df)] = [gray, folder]
    return df


def save_image(path,n):
    img=plt.imread(path)
    gray=(np.mean(img,axis=2).reshape(1,-1)*255).astype(np.uint8).tolist()[0]
    df=pd.DataFrame(columns=['data','label'])
    df.loc[0]=[gray,n]
    df.to_csv(f"{n}.csv",index=False)


df = create_df()
df.to_csv("data/data.csv", index=False)