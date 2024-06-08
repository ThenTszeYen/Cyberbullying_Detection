### Cyberbullying Detection in Social Media
## Project Overview
This repository contains the code for my final year project on using transformer models for detecting cyberbullying in social media.
## Project Structure
1. In 'Project Outline' file:
   1. Step1_data_preparation
   2. Step2_text_preprocessing
   3. Step3_finetune_transformers
2. cb_detection_app.py
## Environment
All the finetuning process are done in Google Colab because of the GPU availability.
## Deployment
The best model will be deployed using Streamlit. The deployment code can be found in cb_detection_app.py.
Due to resource limitation in streamlit cloud, can download the file and run through command in local:
streamlit run cb_detection_app.py
