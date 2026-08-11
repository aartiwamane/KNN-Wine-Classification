# KNN-Wine-Classification
A machine learning case study using K-Nearest Neighbors (KNN) for wine classification, including data preprocessing, feature scaling, train-test splitting, K-value tuning, accuracy evaluation, and visualization.
# KNN Wine Classification

A machine learning case study implementing the **K-Nearest Neighbors (KNN)** algorithm for wine classification using Python and Scikit-learn.

## Project Overview

This project demonstrates the complete workflow of a classification problem, starting from loading and analyzing the dataset to training a KNN model and evaluating its performance for different values of **K**.

The project also uses feature scaling because KNN is a distance-based algorithm.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib
* K-Nearest Neighbors (KNN)
* StandardScaler
* Train-Test Split
* Accuracy Score

## Project Workflow

The implementation follows these steps:

1. **Load the Dataset**

   * Read the wine dataset using Pandas.
   * Display the first and last records.

2. **Analyze the Dataset**

   * Remove missing values.
   * Check dataset shape.
   * Display statistical summary.

3. **Separate Features and Target**

   * Independent variables → `X`
   * Target variable → `Class`

4. **Split the Dataset**

   * Split the dataset into training and testing sets.
   * Use `stratify=Y` to maintain the class distribution.

5. **Feature Scaling**

   * Apply `StandardScaler` to normalize the features.
   * Fit the scaler only on training data and transform both training and testing data.

6. **KNN Model Training**

   * Train KNN models with different values of `K`.
   * Values of K from **1 to 20** are tested.

7. **Model Evaluation**

   * Calculate classification accuracy for each K value.
   * Compare the performance of different K values.

8. **Visualization**

   * Plot K values against their corresponding accuracy.
   * Identify the K value that provides better performance.

##  Hyperparameter Tuning

The project tests:

```text
K = 1, 2, 3, ..., 20
```

The accuracy corresponding to each K value is calculated and displayed in the form:

```text
1 : 94.38
2 : 92.13
3 : 95.51
...
```

An accuracy graph is also generated to visualize the effect of different K values.

## Project Structure

```text
KNN-Wine-Classification/
│
├── ClassificationKNN.py
├── WinePredictor.csv
└── README.md
```

##  How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project directory

```bash
cd KNN-Wine-Classification
```

### 3. Install required libraries

```bash
pip install pandas scikit-learn matplotlib
```

### 4. Run the Python program

```bash
python ClassificationKNN.py
```

## Output

The program displays:

* Dataset records
* Dataset shape
* Statistical summary
* Training and testing dataset sizes
* Accuracy for each K value
* Accuracy vs. K graph

## 🎯 Learning Objectives

This project was created as a practical exercise to understand:

* KNN classification
* Data preprocessing
* Missing-value handling
* Feature scaling
* Train-test splitting
* Stratified sampling
* Hyperparameter tuning
* Model evaluation
* Data visualization

## Author

**Aarti Wamane**

M.Tech Computer Engineering
