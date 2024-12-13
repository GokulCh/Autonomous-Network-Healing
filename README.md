# **Autonomous Network Healing**
This repository contains the implementation and research for an **Autonomous Network Healing System**, designed to diagnose and resolve network anomalies in real-time using machine learning and reinforcement learning techniques. The system simulates network conditions using NS3 and evaluates performance across various scenarios, aligning with modern trends in cloud and edge computing.

---

## **Project Objectives**
- **Anomaly Detection:** Identify network irregularities using unsupervised learning models such as autoencoders and clustering algorithms.
- **Corrective Actions:** Automate network recovery using reinforcement learning (RL) for real-time decision-making.
- **Simulation:** Use NS3 to replicate realistic network scenarios, test the system under stress, and validate its effectiveness.
- **Performance Metrics:** Optimize detection latency, recovery times, and overall network performance.

---

## **Repository Structure**
Here’s an overview of the main folders and their purpose:

```plaintext
Autonomous_Network_Healing/
│
├── datasets/               # Contains raw, processed, and simulated network data
│   ├── processed_data/
│   │   ├── processed_test_features.csv
│   │   ├── processed_test_labels.csv
│   │   ├── processed_train_features.csv
│   │   └── processed_train_labels.csv
│   ├── raw_data/
│   └── simulation_data/
├── models/                 # Trained models, checkpoints, and logs
│   ├── autoencoder/
│   │   ├── network_anomaly_autoencoder.keras
│   │   ├── evaluation_on_train_data_output.txt
│   │   ├── evaluation_on_train_data_output2.txt
│   │   ├── evaluation_on_train_data_output3.txt
│   │   └── error_dist_on_test_data.txt
│   ├── clustering/
│   │   ├── kmeans_train_output.txt
│   │   ├── kmeans_test_output.txt
│   │   └── tf_kmeans_centers.npy
│   └── reinforcement_learning/
│       ├── network_healing_rl_model.h5
│       ├── network_healing_rl_model_old.h5
│       ├── output.txt
│       └── simple_network_healing_model.keras
├── simulations/            # NS3 simulation configs, results, and metrics
│   ├── ns3_configs/
│   └── results/
├── scripts/                # Python scripts for preprocessing, training, and testing
│   ├── autoencoder/
│   │   ├── evaluate_autoencoder.py
│   │   ├── train_autoencoder.py
│   │   ├── train_autoencoder_old.py
│   │   └── train_autoencoder_old2.py
│   ├── clustering/
│   │   ├── evaluate_cluster.py
│   │   └── train_cluster.py
│   ├── data_processing/
│   │   └── data_processing.py
│   ├── reinforcement_learning/
│   │   ├── evaluate_rl_agent.py
│   │   ├── train_rl_agent.py
│   │   └── train_rl_agent_old.py
│   └── simulation/
│       ├── run_ns3_simulation.py
│       └── run_ns3_simulation_old.py
├── reports/                # Project documentation, reports, and presentations
│   ├── ~$tonomous_Network_Healing.doc
│   ├── Autonomous_Network_Healing.doc
│   └── ceidp_paper-template.doc
├── docs/                   # Documentation files for setup, tools, and guides
├── .gitignore              # Specifies files to exclude from version control
└── README.md               # Project overview and instructions
```

### **1. `datasets/`**
Contains all datasets used for training, testing, and validating the models.
- **`processed_data/`**: Cleaned and preprocessed data for model training and testing.
  - `processed_test_features.csv`
  - `processed_test_labels.csv`
  - `processed_train_features.csv`
  - `processed_train_labels.csv`
- **`raw_data/`**: Original datasets.
- **`simulation_data/`**: Data generated from NS3 simulations.

### **2. `models/`**
Stores machine learning models and their training logs.
- **`autoencoder/`**:
  - `network_anomaly_autoencoder.keras`: Pre-trained autoencoder model for anomaly detection.
  - `evaluation_on_train_data_output.txt`: Evaluation output on training data.
  - `evaluation_on_train_data_output2.txt`: Additional evaluation output on training data.
  - `evaluation_on_train_data_output3.txt`: Another evaluation output on training data.
  - `error_dist_on_test_data.txt`: Error distribution on test data.
- **`clustering/`**:
  - `kmeans_train_output.txt`: Output from K-means training.
  - `kmeans_test_output.txt`: Output from K-means testing.
  - `tf_kmeans_centers.npy`: Trained K-means cluster centers.
- **`reinforcement_learning/`**:
  - `network_healing_rl_model.h5`: Trained RL model for network healing.
  - `network_healing_rl_model_old.h5`: Older version of the RL model.
  - `output.txt`: Output from RL model training.
  - `simple_network_healing_model.keras`: Simple network healing model.

### **3. `simulations/`**
Contains configuration files and results from NS3 simulations.
- **`ns3_configs/`**: XML or text files for defining simulation scenarios.
- **`results/`**: Outputs and performance metrics from simulations.

### **4. `scripts/`**
Houses Python scripts for data preprocessing, model training, and simulation execution.
- **`autoencoder/`**:
  - `evaluate_autoencoder.py`: Script to evaluate the autoencoder model.
  - `train_autoencoder.py`: Script to train the autoencoder model.
  - `train_autoencoder_old.py`: Older version of the autoencoder training script.
  - `train_autoencoder_old2.py`: Another older version of the autoencoder training script.
- **`clustering/`**:
  - `evaluate_cluster.py`: Script to evaluate clustering.
  - `train_cluster.py`: Script to train a clustering model.
- **`data_processing/`**:
  - `data_processing.py`: Script for data preprocessing.
- **`reinforcement_learning/`**:
  - `evaluate_rl_agent.py`: Script to evaluate the RL agent.
  - `train_rl_agent.py`: Script to train the RL agent.
  - `train_rl_agent_old.py`: Older version of the RL agent training script.
- **`simulation/`**:
  - `run_ns3_simulation.py`: Automates NS3 simulations.
  - `run_ns3_simulation_old.py`: Older version of the NS3 simulation script.

### **5. `reports/`**
Contains all project documentation and presentations.
- `~$tonomous_Network_Healing.doc`: Temporary file.
- `Autonomous_Network_Healing.doc`: Main project report.
- `ceidp_paper-template.doc`: Paper template for CEIDP.

### **6. `docs/`**
Includes general documentation for understanding and replicating the project.

---

## **How to Get Started**

### **1. Set Up the Environment**
1. Install Python (>=3.8) and required libraries:
   ```bash
   pip install -r requirements.txt
   ```
2. Install NS3:

### **2. Preprocess the Data**
- Run the preprocessing scripts:
  ```bash
  python scripts/data_processing/data_processing.py
  ```

### **3. Train the Models**
- Train the anomaly detection model:
  ```bash
  python scripts/autoencoder/train_autoencoder.py
  ```
- Train the clustering model:
  ```bash
  python scripts/clustering/train_cluster.py
  ```
- Train the RL agent:
  ```bash
  python scripts/reinforcement_learning/train_rl_agent.py
  ```

### **4. Simulate and Evaluate**
- Run NS3 simulations:
  ```bash
  python scripts/simulation/run_ns3_simulation.py
  ```
---

## **Tools and Technologies**
- **NS3:** Network simulator for generating and testing anomaly scenarios.
- **TensorFlow & PyTorch:** Frameworks for training machine learning models.
- **Python Libraries:** pandas, scikit-learn, Matplotlib, NumPy, etc.
- **Datasets:** UNSW-NB15.

---

This project builds on research in self-healing networks, anomaly detection, and reinforcement learning. This project is experimental and may exhibit instability. Proceed with caution.

---