# Federated Learning for Anomaly Detection in Distributed Oilfield Automation Systems

## Project Overview

This project evaluates whether federated learning can effectively detect anomalies in distributed industrial automation networks compared to traditional centralized machine learning approaches.

The system simulates geographically distributed oilfield automation nodes that locally train models and share updates with a central server using the Federated Averaging (FedAvg) algorithm. This work demonstrates the trade-off between centralized performance and real-world deployability in distributed industrial systems.
---

## Key Contributions

- Implemented a **federated learning framework** using Flower
- Simulated **non-IID distributed oilfield environments**
- Compared **centralized vs federated anomaly detection performance**
- Extended from **binary detection → multi-class intrusion detection**

---

## Dataset

This project uses the **CIC IoT-DIAD 2024 dataset** as a proof-of-concept for structured OT-like communication patterns.

### Attack Classes:
- Modbus (Normal Traffic)
- DDoS TCP SYN Flood
- Port Scanning
- SQL Injection

> Note: The dataset approximates industrial communication behavior but does not fully reflect real-world oilfield environments.

---

## Models

### Centralized Model (Baseline)
- Isolation Forest
- Trained on fully aggregated dataset
- Represents traditional industrial monitoring architecture

### Federated Learning Model (Proposed)
- Neural Network (PyTorch)
- Trained across 10 distributed clients
- Aggregated using FedAvg
- Supports non-IID data distribution

---

## Experimental Results

### Final Performance (Experiment 2)

| Metric | Centralized Model | Federated Model |
|-------|------------------|----------------|
| Accuracy | 1.00 | 0.94 |
| Precision | 1.00 | 0.91 |
| Recall | 1.00 | 0.90 |
| Macro F1-score | 1.00 | 0.90 |

---

## Key Observations

- Federated learning achieves **strong performance without centralizing data**
- Centralized model acts as an **upper-bound benchmark**
- Non-IID data introduces realistic performance constraints
- Model converges rapidly within early communication rounds

---

## Project Structure
- federated_learning_model_nn/
- centralized_model_nn/
- baseline_isolation_forest/


## How to Run

1. Install dependencies  
   pip install -r requirements.txt  

2. Run federated learning simulation  
   python federated_learning_model_nn/run_flower_simulation.py  

3. Run centralized model  
   python centralized_model_nn/train_centralized_nn.py  

## Limitations
- Dataset is not from real oilfield environments
- Centralized model may overfit due to full data visibility
- Limited number of clients (simulation)

## Future Work
- Increase number of clients
- Evaluate additional attack types
- Introduce statistical validation (multiple runs)
- Explore trust-based aggregation methods

## Author
Emmanuel Cardenas
University of Texas Permian Basin

## Final Step
Run these commands in your terminal:

git add README.md
git commit -m "Finalize README"
git push origin main