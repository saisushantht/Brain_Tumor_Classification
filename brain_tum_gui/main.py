
# import tkinter module
import tkinter
from tkinter import *
from tkinter import font 
from tkinter.ttk import *
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from typing import Counter
from textblob import TextBlob
from googletrans import Translator
import warnings
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
warnings.filterwarnings("ignore")
# creating main tkinter window/toplevel
master = Tk()
master.geometry("500x200")
# master.state("zoomed")
master.title("Brain tumor classification using MRI images")
text = tkinter.Label( text="Brain tumor classification using MRI images", font=("Helvetica", 12), height=2, anchor='n', fg='#f00')
def askopenfile():
    global filename
    global df
    global name
    filename=askopenfilename( filetypes =(("JPG Files","*.jpg"),))
    result['text']="File uploaded"
b=Button(text="Upload Image", command=askopenfile)
result = tkinter.Label(text="you will see result here!",font=('Courier', 12), height=40, anchor='nw')
text.grid(row=0, columnspan=4)

def Close():
    master.destroy()
def predict():
    labels=['No tumor','meningioma tumor','glioma tumor','pituitary tumor']
    model = load_model('model.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(filename)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)

    
    result['text']=labels[np.argmax(prediction)]
b1=Button(text="predict", command=predict)
b2=Button(text="Close application", command=Close)
b.grid(row=1,column=0)
b1.grid(row=2,column=0)
b2.grid(row=3, column=0)
result.grid(rowspan=6,columnspan=4)

# infinite loop which can be terminated 
# by keyboard or mouse interrupt
mainloop()