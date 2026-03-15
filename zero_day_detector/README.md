# PortSentry

PortSentry is an AI-powered real-time network behavioral monitoring system that captures live traffic, extracts flow-based features, and detects anomalous behavior using a machine learning model trained on derived CIC-IDS-2017 features.

## Overview

PortSentry is a Flask-based desktop-style web application for Windows that:

- captures live network traffic for 20 seconds
- converts packets into flow-level behavioral features
- scores each flow using an Isolation Forest anomaly detection model
- displays a modern dashboard with anomaly counts, risk level, and per-flow analysis

The application is designed for demo and hackathon use, with a focus on lightweight deployment and real-time visibility.

## Features

- Real-time live network scan
- Automatic active network interface detection
- Flow-based anomaly detection using a trained ML pipeline
- 20-second behavioral capture window
- Modern UI with home page and scan results page
- Per-flow reporting with:
  - protocol
  - destination port
  - packet count
  - byte count
  - anomaly score
  - anomaly status
- Packaged Windows `.exe` support using PyInstaller

## Tech Stack

- Python
- Flask
- Scapy
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Tailwind CSS

## Machine Learning Model

The anomaly detection model is trained using derived behavioral features from the CIC-IDS-2017 dataset.

### Model Type
- Isolation Forest

### Training Source
- CIC-IDS-2017 CSV files

### Features Used

The model was trained on the following engineered features:

- `duration`
- `packets`
- `bytes`
- `pps`
- `bps`
- `avg_packet_size`
- `src_port`
- `dst_port`
- `is_tcp`
- `is_udp`
- `is_sensitive_port`
- `tcp_flag_total`

### Training Logic

- only benign rows were used for training
- anomalies are identified at inference time using the Isolation Forest prediction output
- prediction interpretation:
  - `-1` = anomaly
  - `1` = normal

## Project Structure

```text
PortSentry/
├── app.py
├── templates/
│   ├── home.html
│   └── index.html
├── saved_model/
│   └── anomaly_model.pkl
├── dataset/
│   └── MachineLearningCVE/
├── training.py
└── README.md