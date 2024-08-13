import streamlit as st
from PIL import Image
import pandas as pd
import os
from pydantic_settings import BaseSettings

# Importing the necessary libraries for machine learning tasks
from pycaret.regression import setup, compare_models, pull, save_model

# Importing the profiling tool to perform EDA
from ydata_profiling import ProfileReport

# Define the path to the local image file (used in the sidebar)
image_path = "/Users/justinlee/Desktop/AutoML Webpage/ML_webapp/python_image.png"

# Load the image from the specified path
image = Image.open(image_path)
 
# Sidebar setup
with st.sidebar:
    # Display the image in the sidebar
    st.image(image)
    # Title for the sidebar
    st.title("Auto ML Web Application")
    # Radio buttons for selecting different functionalities
    choice = st.radio('Options', ["Upload", "Profiling", "Machine Learning", "Download"])
    # Info text for guidance
    st.info("This application allows you to build an automated ML pipeline using Streamlit, Pandas Profiling, and Pycaret. And it is damn amazing!")

# Check if the sourcedata.csv file already exists
if os.path.exists("sourcedata.csv"):
    # If it exists, load the data into a dataframe
    df = pd.read_csv("sourcedata.csv", index_col=None)

# Functionality 1: Upload
if choice == "Upload":
    # Title for the Upload page
    st.title("Upload Your Data for Modelling!")
    # File uploader widget
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        # If a file is uploaded, load it into a dataframe
        df = pd.read_csv(file, index_col=None)
        # Save the uploaded file locally
        df.to_csv("sourcedata.csv", index=None)
        # Display the dataframe
        st.dataframe(df)

# Functionality 2: Profiling
if choice == "Profiling":
    # Title for the Profiling page
    st.title("Automated Exploratory Data Analysis")
    if df is not None:
        # Generate the profile report if data is available
        profile_report = ProfileReport(df)
        # Save the profile report as an HTML file
        profile_report.to_file("report.html")
        # Open the HTML file and read the content
        with open("report.html", "r", encoding="utf-8") as f:
            html = f.read()
        # Render the profile report in the Streamlit app
        st.components.v1.html(html, height=1000, scrolling=True)
    else:
        # If no data is available, prompt the user to upload a dataset
        st.write("No data available for profiling. Please upload a dataset first.")

# Functionality 3: Machine Learning
if choice == "Machine Learning":
    # Title for the Machine Learning page
    st.title("Machine Learning in Progress")
    # Dropdown to select the target column for ML
    target = st.selectbox("Select The Target Column", df.columns)
    
    # Uncomment the below lines if the target column contains boolean values
    # Convert boolean target to numeric if selected
    # if df[target].dtype == 'bool':
    #     df[target] = df[target].astype(int)
    
    if st.button('Run Machine Learning'):
        # Initialize the ML setup with the selected target column
        setup(data=df, target=target, verbose=False)
        # Pull the setup dataframe and display the ML experiment settings
        setup_df = pull()
        st.info("This is the ML Experiment settings")
        st.dataframe(setup_df)
        
        # Compare models to find the best one
        best_model = compare_models()
        # Pull the comparison results and display them
        compare_df = pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        
        # Save the best model to a file
        save_model(best_model, "best_model")

# Functionality 4: Download
if choice == "Download": 
    # Allow the user to download the saved model
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
