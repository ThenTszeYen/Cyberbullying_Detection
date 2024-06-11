# Cyberbullying Detection in Social Media
## Project Overview
This repository contains the code for my final year project on using transformer models for detecting cyberbullying in social media.
## Project Structure
1. In 'Project Outline' file:
   1. Step1_data_preparation: Extract files using 'BratReader'.
   2. Step2_text_preprocessing: Text cleaning.
   3. Step3_finetune_transformers: Finetuning DistilBERT, AlBERT and RoBERTa Tiny with different hyperparameter settings.
2. app.py: Streamlit app for cyberbullying detection with justification from Local Interpretable Model-agnostic Explanations (LIME).
3. requirements.txt: The Python packages or library with the depencies used.
4. Dockerfile: To build and deploy containerized application.
## Environment
All the finetuning process are done in Google Colab because of the GPU availability.
## Deployment
The best model will be deployed using Streamlit and Cloud Run from Google Cloud Platform for demonstation. The deployment code can be found in 'app.py'.

Application for demonstation:
https://streamlit-dbhou3h2uq-as.a.run.app

Due to resource limitation in streamlit cloud, can download the file and run through command in local:
streamlit run app.py

