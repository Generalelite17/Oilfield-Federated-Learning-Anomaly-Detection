# Federated Learning for Anomaly Detection in Distributed Oilfield Automation Systems

## Project Goal

This project evaluates whether federated learning can effectively detect anomalies in distributed industrial automation networks compared to centralized machine learning models.

The system simulates geographically distributed oilfield automation nodes that locally train anomaly detection models and share model updates with a central aggregation server to build a global detection model while preserving data locality.

## Baseline Model

The baseline implementation uses a centralized anomaly detection approach.
Network telemetry data is collected and processed on a single machine where
an Isolation Forest model is trained to identify anomalous network behavior.

This centralized approach represents a traditional architecture commonly used
in industrial monitoring systems where all telemetry is aggregated into a
central analytics platform.

## Proposed Extension

The proposed approach introduces a federated learning framework where multiple
distributed automation nodes collaboratively train anomaly detection models
without sharing raw network telemetry data.

Each node performs local training on its own dataset partition and sends model
updates to a central aggregation server. The server combines these updates
using the Federated Averaging (FedAvg) algorithm to produce an improved
global model.
## Dataset

The centralized baseline model uses structured network-flow telemetry data from the dataset:

csv_data/BenignTraffic.pcap_Flow.csv

For the federated learning simulation, the MNIST dataset is used to demonstrate distributed model training and aggregation across multiple simulated clients.

## Experimental Design

This project evaluates anomaly detection performance using two approaches:

### Baseline: Centralized Training

In the baseline system, all network telemetry data is collected and processed on a centralized machine. The model is trained using the full dataset and performs anomaly detection on aggregated network traffic.

Workflow:

Network Traffic Data
        ↓
Centralized Processing
        ↓
Isolation Forest Training
        ↓
Anomaly Detection

Advantages:
- Simpler architecture
- No communication overhead
- Direct access to full dataset

Limitations:
- Requires centralizing sensitive network data
- Not scalable for distributed environments

### Proposed Method: Federated Learning

The proposed system distributes model training across multiple simulated nodes representing geographically distributed oilfield automation systems.

Each node trains a local model using its own dataset partition. Instead of sharing raw data, nodes send model updates to a central aggregation server. The server aggregates these updates using the Federated Averaging (FedAvg) algorithm to produce an updated global model.

Workflow:

Client 1 Dataset
Client 2 Dataset
Client 3 Dataset
        ↓
Local Training
        ↓
Server Aggregation (FedAvg)
        ↓
Global Model Update

### Baseline vs Proposed Comparison

| Feature | Baseline Model | Federated Model |
|-------|----------------|----------------|
Training Architecture | Centralized | Distributed |
Data Location | All data centralized | Data remains local |
Privacy | Lower | Higher |
Communication | None | Requires aggregation rounds |
Scalability | Limited | Suitable for distributed systems |