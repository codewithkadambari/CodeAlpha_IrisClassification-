# Iris Flower Classification - CodeAlpha

This project is part of the CodeAlpha Data Science Internship. It involves building a machine learning model to classify iris flowers into three species: Setosa, Versicolor, and Virginica, based on their sepal and petal measurements.

## Project Overview
The Iris dataset is a classic in machine learning. It contains 150 samples with four features each:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

## Implementation Steps
1. **Data Exploration**: Analyzed feature distributions and relationships using Seaborn pairplots.
2. **Preprocessing**: Split data into training (80%) and testing (20%) sets and standardized features using `StandardScaler`.
3. **Model Selection**: Used **Logistic Regression** for classification.
4. **Evaluation**: Achieved high accuracy on the test set. Generated a confusion matrix and classification report.

## Results
- **Accuracy**: Typically > 95% (depends on the random split).
- **Confusion Matrix**: Shows precise classification for all three species.

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn
   ```
2. Run the script:
   ```bash
   python iris_classification.py
   ```

## Visualizations
- `iris_pairplot.png`: Shows how different features distinguish the species.
- `confusion_matrix.png`: Details the model's performance on the test set.
