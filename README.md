# Intrusion Detection System (IDS) with Machine Learning

## Project Overview

This project implements an Intrusion Detection System (IDS) using Machine Learning techniques to detect malicious network activities based on traffic patterns. The system combines machine learning-based attack detection with real-time packet monitoring to identify suspicious behavior in network traffic.

## Objective

To design and implement an Intrusion Detection System (IDS) that uses machine learning algorithms to detect malicious network activity based on network traffic patterns.

## Tools and Technologies Used

* Python 3.11
* Scikit-learn
* TensorFlow
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Wireshark
* Scapy
* Joblib

## Dataset

The project uses the NSL-KDD dataset:

* KDDTrain+.txt
* KDDTest+.txt

The dataset contains normal and attack network traffic records used for training and testing intrusion detection models.

## Machine Learning Workflow

### Data Preprocessing

* Loaded NSL-KDD dataset
* Converted attack labels into binary classes:

  * Normal = 0
  * Attack = 1
* Applied one-hot encoding to categorical features
* Prepared training and testing datasets

### Model Training

A Gradient Boosting Classifier was trained using the processed NSL-KDD dataset.

Parameters:

* n_estimators = 200
* learning_rate = 0.1
* random_state = 42

### Model Performance

Results on the NSL-KDD test dataset:

* Accuracy: 80.74%
* ROC-AUC Score: 0.946

Confusion Matrix:

| Actual \ Predicted | Normal | Attack |
| ------------------ | ------ | ------ |
| Normal             | 9444   | 267    |
| Attack             | 4075   | 8758   |

## Real-Time Traffic Monitoring

The project uses Scapy for real-time packet capture and network monitoring.

Features:

* Live packet capture
* Source IP tracking
* Packet counting
* Suspicious activity detection
* Real-time alert generation

Example Alert:

⚠ ALERT: Suspicious Activity Detected!

Source IP: 10.54.12.143

Packets: 101

## Visualizations

Generated visualizations include:

1. Attack Distribution Graph
2. Confusion Matrix Heatmap
3. Feature Importance Graph

All generated graphs are stored in the screenshots folder.

## Project Structure

IDS/

├── dataset/

├── models/

├── screenshots/

├── src/

├── main.py

└── README.md

## Key Features

* Network traffic preprocessing
* Machine learning-based attack classification
* Gradient Boosting intrusion detection model
* Real-time packet capture using Scapy
* Intrusion alert generation
* Network traffic visualization

## Future Enhancements

* Real-time ML prediction on live traffic
* Multi-class attack classification
* Web dashboard for monitoring
* Database logging of alerts
* Advanced anomaly detection techniques
