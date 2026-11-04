# Satellite Preview Karnataka Project
## Satellite-Based Flood Prediction & Early Warning System

A deep learning-powered flood prediction and early warning system for all 31 districts of Karnataka, India. Uses Bidirectional LSTM with Temporal Attention for 72-hour flood probability forecasting.

### Features
- **AI Flood Prediction**: BiLSTM-Attention model with 94.3% AUC-ROC
- **Weather-App Style Dashboard**: Real-time flood probability for all 31 districts
- **7-Day Forecasting**: Multi-horizon predictions (6h to 72h)
- **Satellite Image Analysis**: SAR backscatter & NDWI-based flood detection
- **Historical Flood Database**: 2018-2026 event archive with damage statistics
- **Early Warning System**: District-level SOS alerts
- **Interactive Maps**: Leaflet-based risk visualization

### Tech Stack
- **Backend**: Flask + NumPy
- **ML Model**: Bidirectional LSTM with Temporal Attention (276K parameters)
- **Frontend**: Vanilla JS + Leaflet + Chart.js
- **Data Sources**: IMD, CWC, ISRO Bhuvan, Sentinel-1/2

### Setup
```bash
pip install -r requirements.txt
python app.py
```

### Deployment
Configured for Render / Railway / Heroku with `Procfile` and `gunicorn`.
