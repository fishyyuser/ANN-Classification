from tensorflow.keras.models import load_model
import pickle

def load_preprocessor(file_path:str):
    with open(file_path,'rb')as file:
        preprocessor=pickle.load(file)
    return preprocessor

def load_ann_model(file_path:str):
    model=load_model(file_path)
    return model