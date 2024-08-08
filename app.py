import streamlit as st
from PIL import Image
import pandas as pd
import os

from pydantic_settings import BaseSettings

# Import Profiling capability
import pandas_profiling
from streamlit_pandas_profiling import st_profiling_report

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
        st.dataframe(df.head(100))
        

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)


if choice == "Machine Learning":
    pass

if choice == "Download":
    pass
