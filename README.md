# 🛠️ predictive-maintenance-1dcnn
## 📌 Overview
`predictive-maintenance-1dcnn` is an Edge AI solution designed to monitor machinery health and predict potential failures in real-time. By leveraging a 1-Dimensional Convolutional Neural Network (1D-CNN), the system processes sequential time-series sensor data (such as vibration or temperature metrics) to accurately classify the operational state of equipment. The model is highly optimized for deployment on resource-constrained microcontrollers, enabling on-device inference with ultra-low latency.

## 🔄 System Flow Diagram

The following diagram illustrates the end-to-end data pipeline, from raw sensor acquisition on the edge device to fault classification.

## 🧰 Tools & Technologies Used
This project integrates industry-standard tools across the machine learning and embedded engineering stacks:

- TensorFlow & Keras: Used as the primary deep learning framework to design, train, and validate the 1D-CNN architecture. 1D-CNNs are specifically chosen for their high efficiency and lightweight structure when processing sequential temporal signals.

- TensorFlow Lite for Microcontrollers (TinyML): Essential for converting and quantizing the trained TensorFlow model into a lightweight C-byte array. This significantly reduces the memory footprint, allowing the neural network to run entirely on edge hardware.

- ESP32 Microcontroller: The target hardware for deployment. It handles real-time sensor data acquisition and runs the quantized 1D-CNN model for on-device inference, eliminating the need for continuous cloud connectivity and reducing latency.

- Scikit-Learn: Utilized during the data preparation phase for scaling and normalizing raw sensor values, splitting datasets, and generating robust performance evaluation metrics.

- Pandas & NumPy: Core Python libraries used for loading, manipulating, and structuring the raw time-series datasets into overlapping windows suitable for 1D-CNN ingestion.

- Matplotlib & Seaborn: Used for data visualization, specifically to render the model evaluation plots like the training loss curves, ROC curve, and confusion matrix.
