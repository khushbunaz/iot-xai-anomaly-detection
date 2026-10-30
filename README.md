# iot-xai-anomaly-detection
MSC Anomaly Detection Project
# Explainable AI-Driven Anomaly Detection for IoT Networks (N-BaIoT)
MSc AI project, University of Bedfordshire, CIS144-6. Khushbunaz Dalal (2529717)

Files
- Anomaly detection_MSC_Project.ipynb: full pipeline (data loading, models, evaluation, SHAP)
- iot_artefact_app.py: Gradio dashboard

Run the dashboard in Colab
!pip install gradio plotly shap -q
%run iot_artefact_app.py
(needs X_test.npy, scaler.pkl, model files in the working folder)

Data: N-BaIoT, UCI Machine Learning Repository
