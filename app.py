import streamlit as st
from PIL import Image
import pandas as pd
import os
from pydantic_settings import BaseSettings

# Machine Learning stuff
from pycaret.regression import setup, compare_models, pull, save_model

# Import Profiling capability
from ydata_profiling import ProfileReport

# Path to your local image file
image_path = "/Users/justinlee/Desktop/AutoML Webpage/ML_webapp/python_image.png"

# Open the image file
image = Image.open(image_path)

with st.sidebar:
    st.image(image)
    st.title("Auto ML Web Application")
    choice = st.radio('Options', ["Upload", "Profiling", "Machine Learning", "Download"])
    st.info("This applocation allows you to build an automated ML pipline using Streamlit, Pandas Profiling and Pycaret. And it is damn amazing!")
        
if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload Your Data for Modelling!")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    if df is not None:
        profile_report = ProfileReport(df)
        profile_report.to_file("report.html")
    with open("report.html", "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=1000, scrolling=True)

if choice == "Machine Learning":
    st.title("Machine Learning on progress")
    target = st.select_box("Select Target Column", df.columns)
    
    
if choice == "Download":
    pass
