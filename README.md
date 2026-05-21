# 🛠️ predictive-maintenance-1dcnn
## 📌 Overview
`predictive-maintenance-1dcnn` is an Edge AI solution designed to monitor machinery health and predict potential failures in real-time. By leveraging a 1-Dimensional Convolutional Neural Network (1D-CNN), the system processes sequential time-series sensor data (such as vibration or temperature metrics) to accurately classify the operational state of equipment. The model is highly optimized for deployment on resource-constrained microcontrollers, enabling on-device inference with ultra-low latency.

## 🔄 System Flow Diagram

The following diagram illustrates the end-to-end data pipeline, from raw sensor acquisition on the edge device to fault classification.

mermaid
graph TD
    A[Raw Sensor Data Acquisition] --> B[Signal Preprocessing & Noise Filtering]
    B --> C[Time-Series Windowing]
    C --> D[1D-CNN Feature Extraction]
    D --> E[Fully Connected Classification Layer]
    E --> F{Health Status?}
    F -->|Normal Operation| G[Log Data & Continue]
    F -->|Fault Detected| H[Trigger Maintenance Alert]
    
    classDef hardware fill:#2b3137,stroke:#fafbfc,stroke-width:2px,color:#fff;
    classDef compute fill:#0366d6,stroke:#fafbfc,stroke-width:2px,color:#fff;
    classDef action fill:#d73a49,stroke:#fafbfc,stroke-width:2px,color:#fff;
    
    class A hardware;
    class B,C,D,E compute;
    class H action; 

## 🧰 Tools & Technologies Used
This project integrates industry-standard tools across the machine learning and embedded engineering stacks:

- TensorFlow & Keras: Used as the primary deep learning framework to design, train, and validate the 1D-CNN architecture. 1D-CNNs are specifically chosen for their high efficiency and lightweight structure when processing sequential temporal signals.

- TensorFlow Lite for Microcontrollers (TinyML): Essential for converting and quantizing the trained TensorFlow model into a lightweight C-byte array. This significantly reduces the memory footprint, allowing the neural network to run entirely on edge hardware.

- ESP32 Microcontroller: The target hardware for deployment. It handles real-time sensor data acquisition and runs the quantized 1D-CNN model for on-device inference, eliminating the need for continuous cloud connectivity and reducing latency.

- Scikit-Learn: Utilized during the data preparation phase for scaling and normalizing raw sensor values, splitting datasets, and generating robust performance evaluation metrics.

- Pandas & NumPy: Core Python libraries used for loading, manipulating, and structuring the raw time-series datasets into overlapping windows suitable for 1D-CNN ingestion.

- Matplotlib & Seaborn: Used for data visualization, specifically to render the model evaluation plots like the training loss curves, ROC curve, and confusion matrix.

## 📊 Model Evaluation & Results
The model was evaluated on a reserved test dataset to measure its ability to distinguish between normal operating conditions and various mechanical fault states.

### Confusion Matrix
The confusion matrix demonstrates the model's high precision and recall, minimizing costly false positives (unnecessary maintenance interventions) and critical false negatives (missed equipment failures).

|True Class \ Predicted | Normal|	Fault Type A	|Fault Type B|
|-----------------------|-------|---------------|------------|
|Normal|	98.5%	| 1.0% | 0.5%|
|Fault |Type A	|1.2%	|97.8%	|1.0%|
|Fault| Type B|	0.4%|	0.6%|	99.0%|

- Interpretation: The model excels at identifying severe anomalies (Fault Type B) with 99% accuracy. Misclassifications are incredibly rare and primarily occur between baseline noise and minor early-stage wear.

### Receiver Operating Characteristic (ROC) Curve
The ROC curve evaluates the model's diagnostic ability across different discrimination thresholds.

- Area Under the Curve (AUC): 0.992

- Interpretation: An AUC score of 0.992 indicates an exceptional capacity to separate normal signals from faulty ones. The curve sharply hugs the top-left corner, meaning the network achieves a very high True Positive Rate (Sensitivity) while maintaining a near-zero False Positive Rate, which is a mandatory requirement for automated predictive maintenance systems.
