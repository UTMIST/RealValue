import tensorflow as tf
import numpy as np
import os,pandas as pd

def tester():
    #bedroom_path,bathroom_path,frontal_path,kitchen_path,postal_3,sqft
    df = pd.read_csv('CA.txt', delimiter = "\t")
    toronto_dataset = pd.read_excel('TorontoDataset.xlsx').loc[:,["Zip Code"]]
    print(toronto_dataset)
    print(toronto_dataset.columns)
    main_dataframe = df.loc[:,["T0A","54.766","-111.7174"]]
    main_dataframe = main_dataframe.rename(columns={"T0A": "Zip Code", "54.766": "Long", "-111.7174":"Lat"})
    print(main_dataframe)
    temp_try = pd.merge(toronto_dataset,main_dataframe,how="inner")
    print(temp_try)
    return True

if __name__=='__main__':
    tester()
