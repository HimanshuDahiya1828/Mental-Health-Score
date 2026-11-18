# Student Social Media & Mental Health Prediction — Streamlit Deployment

## Included Files (Ready for Deployment)
- `app.py` — interactive Streamlit web application
- `Mental_Health_Regression_Model.pkl` — pre-trained & tuned Random Forest Regressor
- `Mental_Health_Classification_Model.pkl` — pre-trained & tuned Random Forest Classifier
- `model_metadata.json` — feature schema, defaults, and dropdown options
- `requirements.txt` — pinned Python dependencies matching the trained models
- `train_models.py` — model training script (supports `.csv`, `.xlsx`, and `.xls`)
- `Student Social Media And Mental Health Impact.csv` — dataset file

## Deploy Directly to Streamlit Community Cloud
1. Extract this folder.
2. Upload all files inside this folder (`app.py`, `requirements.txt`, `Mental_Health_Regression_Model.pkl`, `Mental_Health_Classification_Model.pkl`, `model_metadata.json`, and `README.md`) to your GitHub repository (`Mental-Health-Score`).
3. Go to [share.streamlit.io](https://share.streamlit.io/), sign in with GitHub, and click **Create app**.
4. Select your repository (`HimanshuDahiya1828/Mental-Health-Score`), branch (`main`), and main file path (`app.py`).
5. Click **Deploy**.
