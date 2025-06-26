# ML for Season Recognition

## Overview
This project implements three machine learning models: **Multi-Layer Perceptron (MLP)** , **1D Convolutional Neural Network (1D-CNN)**, and **2D Convolutional Neural Network (2D-CNN)** to classify seasons (winter, spring, summer, autumn) using 24-hour time-series data of power load, wind generation, and solar generation in Denmark. The dataset, sourced from Open Power System Data (opsd_raw.csv), spans hourly measurements from 2015 to 2020. The goal is to predict seasons based on daily energy profiles, leveraging temporal and spatial patterns in energy consumption and generation.

## Dataset
The dataset contains 50,400 hourly records with three key features:

- **Power Load (DK_load_actual_entsoe_transparency):** Total electricity load in Denmark (MW), sourced from the ENTSO-E Transparency Platform.
- **Wind Generation (DK_wind_generation_actual):** Actual wind power generation in Denmark (MW).
- **Solar Generation (DK_solar_generation_actual):** Actual solar power generation in Denmark (MW).

Each feature is aggregated into 24-hour time-series arrays per day, forming a (n_days, 3, 24) input matrix (3 features × 24 hours).

## Data Preprocessing

- **Data Loading and Selection:** Extracted relevant columns (utc_timestamp, DK_load_actual_entsoe_transparency, DK_wind_generation_actual, DK_solar_generation_actual) and renamed them to timestamp, power_load, wind_generation, and solar_generation.
- **Handling Missing Values:**
  -  **Power load:** 2 missing values, imputed using forward and backward fill (ffill and bfill) due to their edge placement, preserving temporal trends.
  - **Wind generation:** 2 missing values, imputed using forward and backward fill.
  - **Solar generation:** 11 missing values, filled with monthly-hourly means to reflect seasonal daylight patterns.
- **Data Validation:** Verified each day has exactly 24 hourly records (hours 0–23).
- **Feature Engineering:** Grouped data by date to form 24-hour arrays for each feature, resulting in a (n_days, 3, 24) dataset.
- **Season Labeling:** Assigned seasons based on months
- **Data Splitting:** Split into 70% training, 15% validation, and 15% test sets using stratified sampling to maintain season proportions.
- **Scaling:** Standardized each feature independently using StandardScaler, fitted only on training data to prevent data leakage.
- **Data Transformation:**
  - MLP: Flattened the 3×24 matrix into a 72-dimensional vector (3 features × 24 hours).
  - 1D-CNN: Used the (n_days, 3, 24) input as 3 channels and 24 time steps.
  - 2D-CNN: Reshaped the 24-hour series into a 4×6 "image" (4 segments of 6 hours), resulting in (n_days, 3, 4, 6).



## Exploratory Data Analysis (EDA)
EDA revealed distinct seasonal patterns (see Figure 1 in the report):

- Power Load: Peaks in winter (~5000 MW at 17:00) due to heating and lighting demands, lowest in summer.
- Wind Generation: Highest in winter, likely due to stormier weather, lowest in summer.
- Solar Generation: Peaks in summer with a bell-shaped curve (highest midday, zero at night), with earlier sunrises (e.g., 05:00 vs. 08:00 in winter).

Visualizations included average 24-hour profiles per season and single-day profiles, confirming these trends.

## Models
Three models were developed to classify seasons:

### MLP:

- Architecture: Two hidden layers with ReLU activations, followed by an output layer (4 classes).
- Input: Flattened 3×24 matrix (72 features).
- Hyperparameters (tuned via grid search):
  - Hidden sizes: 128 (layer 1), 64 (layer 2)
  - Learning rate: 0.001


- Optimizer: Adam (chosen for adaptive learning rate and faster convergence vs. SGD).
- Loss: Cross-Entropy
- Test Accuracy: **85.08%**
- Training Curves: Stable learning, with some summer-spring confusion due to weather similarities (see Figure 2 in the report).


### 1D-CNN:

- Architecture: One Conv1d layer (3 input channels), MaxPool1d (kernel size 2), two fully connected layers, ReLU activations, and dropout for regularization.
- Input: (n_days, 3 channels, 24 time steps).
- Hyperparameters (tuned via grid search):
  - Hidden sizes: 256 (layer 1), 128 (layer 2)
  - Number of filters: 16
  - Kernel size: 3
  - Dropout rate: 0.3
  - Learning rate: 0.001


- Optimizer: Adam
- Loss: Cross-Entropy
- Test Accuracy: **86.98%**
- Training Curves: Stable loss and accuracy trends (see Figure 3 in the report).


### 2D-CNN:

- Architecture: Two Conv2d layers with batch normalization, ReLU, MaxPool2d (kernel size 2), one fully connected layer, and dropout.
- Input: (n_days, 3 channels, 4 height, 6 width).
- Hyperparameters (tuned via grid search):
  - Number of filters: 16 (conv1), 32 (conv2)
  - Hidden size: 32
  - Dropout rate: 0.0
  - Learning rate: 0.01


- Optimizer: Adam
- Loss: Cross-Entropy
- Test Accuracy: **87.30%**
- Training Curves: Steady loss decrease but fluctuating validation accuracy (0.80–0.85), suggesting potential overfitting (see Figure 4 in the report).



## Training and Evaluation

- Training: Each model was trained for 100 epochs with a batch size of 32. Training loss and validation accuracy were monitored.
- Hyperparameter Tuning: Grid search optimized hyperparameters (code included but commented out due to long runtime; best parameters hardcoded).
- Evaluation Metrics:
  - Accuracy on validation and test sets.
  - Classification report (precision, recall, F1-score) per season.
  - Confusion matrix (see Figure 5 in the report) for per-class performance.


## Results:
- 2D-CNN: Highest test accuracy (87.30%), excelling in winter (F1: 0.92) and summer (F1: 0.88) due to the 4×6 transformation capturing intra-day patterns (e.g., morning, afternoon).
- 1D-CNN: 86.98% test accuracy, best for spring (recall: 0.80) and strong in autumn (F1: 0.90), leveraging temporal patterns.
- MLP: 85.08% test accuracy, limited by lack of temporal structure but strong in summer (F1: 0.85).


### Per-Class Performance:
- Winter: 2D-CNN best (F1: 0.92, recall: 0.90), capturing high power load and wind generation patterns.
- Spring: 1D-CNN best (recall: 0.80), handling transitional patterns better; 2D-CNN struggles (16 summer misclassifications vs. 14 for 1D-CNN).
- Summer: 2D-CNN best (F1: 0.88, recall: 0.94), leveraging solar generation peaks emphasized by the 4×6 reshape.
- Autumn: 1D-CNN and 2D-CNN strong (F1: 0.90 and 0.89, respectively), capturing wind generation patterns.

## Dependencies

- Python 3.x
- Libraries: pandas, numpy, matplotlib, scikit-learn, torch, seaborn

Install dependencies using:
```pip install pandas numpy matplotlib scikit-learn torch seaborn```

## Usage

1. Clone the repository:
   ```git clone https://github.com/poinka/ML-for-season-recognition.git```

2. Navigate to the project directory:
   ```cd ML-for-season-recognition```


3. Open and run the Jupyter notebook

The notebook is structured into tasks:
- Task 1: Data preprocessing and EDA.
- Task 2: MLP implementation and evaluation.
- Task 3: 1D-CNN implementation and evaluation.
- Task 4: 2D-CNN implementation and evaluation.


Outputs include plots, metrics, and prediction CSV files.

## Notes

- The 2D-CNN’s slight edge (87.30% vs. 86.98% for 1D-CNN) suggests the 4×6 reshape captures intra-day patterns (e.g., morning, afternoon), though improvements are modest.
- Spring is the most challenging season due to its transitional nature, with 1D-CNN performing best.
- The 2D-CNN’s fluctuating validation accuracy indicates potential overfitting, which could be addressed with early stopping or learning rate scheduling.
- Grid search is computationally intensive; best hyperparameters are hardcoded for convenience.

## Future Improvements

- Experiment with alternative 2D reshape configurations for the 2D-CNN.
- Incorporate additional features (e.g., temperature, humidity) to improve spring classification.
- Explore recurrent neural networks (e.g., LSTM) for enhanced temporal modeling.
- Implement early stopping or learning rate scheduling to mitigate 2D-CNN overfitting.
