import streamlit as st
from PIL import Image
import pandas as pd

# Path to your local image file
image_path = "/Users/justinlee/Desktop/AutoML Webpage/ML_webapp/python_image.png"

# Open the image file
image = Image.open(image_path)

with st.sidebar:
    st.image(image)
    st.title("Auto ML Web Application")
    choice = st.radio('Options', ["Upload", "Profiling", "Machine Learning", "Download"])
    st.info("This applocation allows you to build an automated ML pipline using Streamlit, Pandas Profiling and Pycaret. And it is damn amazing!")


if choice == "Upload":
    st.title("Upload Your Data for Modelling!")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())
        

if choice == "Profiling":
    pass

if choice == "Machine Learning":
    pass

if choice == "Download":
    pass
