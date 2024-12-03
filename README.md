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
├── models/                 # Trained models, checkpoints, and logs
├── simulations/            # NS3 simulation configs, results, and metrics
├── scripts/                # Python scripts for preprocessing, training, and testing
├── reports/                # Project documentation, reports, and presentations
├── docs/                   # Documentation files for setup, tools, and guides
├── utilities/              # Supplementary tools for logging, configs, and visualization
└── .gitignore              # Specifies files to exclude from version control
```

### **1. `datasets/`**
Contains all datasets used for training, testing, and validating the models.
- **`raw_data/`**: Original datasets, such as MAWI, CAIDA, and UNSW-NB15.
  - Example: `mawi_traffic_archive.csv`
- **`processed_data/`**: Cleaned and preprocessed data for model training and testing.
  - Example: `training_data.csv`
- **`simulation_data/`**: Data generated from NS3 simulations, such as traffic logs and anomaly results.
  - Example: `simulated_anomalies_traffic.csv`

### **2. `models/`**
Stores machine learning models and their training logs.
- **`anomaly_detection/`**:
  - `autoencoder_model.h5`: Pre-trained autoencoder model for anomaly detection.
  - `clustering_model.pkl`: Trained clustering model (e.g., K-means).
  - `training_logs/`: Logs generated during training, such as loss curves and metrics.
- **`reinforcement_learning/`**:
  - `rl_agent.pkl`: Trained RL agent for corrective actions.
  - `training_logs/`: Logs detailing RL agent's training progress.

### **3. `simulations/`**
Contains configuration files and results from NS3 simulations.
- **`ns3_configs/`**: XML or text files for defining simulation scenarios.
  - Example: `traffic_congestion_scenario.xml`
- **`results/`**: Outputs and performance metrics from simulations.
  - Examples:
    - `anomaly_detection_results.csv`
    - `performance_metrics.json`

### **4. `scripts/`**
Houses Python scripts for data preprocessing, model training, and simulation execution.
- **`data_processing/`**:
  - `clean_data.py`: Script for cleaning and normalizing raw datasets.
  - `feature_engineering.py`: Creates features like entropy and time-windowed metrics.
  - `split_data.py`: Splits data into training and testing sets.
- **`anomaly_detection/`**:
  - `train_autoencoder.py`: Script to train the autoencoder model.
  - `train_clustering.py`: Script to train a clustering model.
  - `evaluate_anomaly_detection.py`: Tests anomaly detection performance.
- **`reinforcement_learning/`**:
  - `train_rl_agent.py`: Script to train the RL agent.
  - `evaluate_rl_agent.py`: Tests the corrective action model.
- **`simulation/`**:
  - `run_ns3_simulation.py`: Automates NS3 simulations.
  - `analyze_simulation_data.py`: Processes simulation outputs for evaluation.

### **5. `reports/`**
Contains all project documentation and presentations.
- **`preliminary_report.docx`**: Initial report outlining the project's objectives, assumptions, and framework.
- **`final_report.docx`**: Comprehensive report with results and analyses.
- **`presentation.pptx`**: Slide deck summarizing the project for presentations.
- **`references/`**:
  - `citations.bib`: Bibliographic references for the project.
  - `related_work.pdf`: Key papers and resources.

### **6. `docs/`**
Includes general documentation for understanding and replicating the project.
- `project_plan.md`: High-level timeline and milestones.
- `setup_guide.md`: Instructions for setting up the environment and dependencies.
- `tools_and_technologies.md`: Overview of tools like NS3, TensorFlow, and PyTorch.

### **7. `utilities/`**
Supplementary tools and scripts.
- **`logging/`**:
  - `log_parser.py`: Script to process logs for debugging.
- **`config/`**:
  - `config.yaml`: Configuration file for setting hyperparameters and paths.
  - `hyperparameters.json`: Stores model-specific parameters for training.
- **`visualization/`**:
  - `plot_metrics.py`: Generates performance plots (e.g., accuracy, precision-recall curves).
  - `visualize_anomalies.py`: Visualizes detected anomalies.
  - `visualize_corrective_actions.py`: Displays the RL agent's decision-making.

---

## **How to Get Started**

### **1. Set Up the Environment**
1. Install Python (>=3.8) and required libraries:
   ```bash
   pip install -r requirements.txt
   ```
2. Install NS3:
   - Follow the setup instructions in `docs/setup_guide.md`.

### **2. Preprocess the Data**
- Run the preprocessing scripts:
  ```bash
  python scripts/data_processing/clean_data.py
  python scripts/data_processing/feature_engineering.py
  ```

### **3. Train the Models**
- Train the anomaly detection model:
  ```bash
  python scripts/anomaly_detection/train_autoencoder.py
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
- Analyze simulation results:
  ```bash
  python scripts/simulation/analyze_simulation_data.py
  ```

### **5. Visualize Results**
- Generate performance plots:
  ```bash
  python utilities/visualization/plot_metrics.py
  ```

---

## **Tools and Technologies**
- **NS3:** Network simulator for generating and testing anomaly scenarios.
- **TensorFlow & PyTorch:** Frameworks for training machine learning models.
- **Python Libraries:** pandas, scikit-learn, Matplotlib, NumPy.
- **Datasets:** MAWI, CAIDA, UNSW-NB15.

---

This project builds on research in self-healing networks, anomaly detection, and reinforcement learning. This project is experimental and may exhibit instability. Proceed with caution.

---