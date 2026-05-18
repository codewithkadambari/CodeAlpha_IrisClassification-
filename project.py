import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------
# CODEALPHA INTERNSHIP PROJECT
# TASK 1 : IRIS FLOWER CLASSIFICATION
# -----------------------------------

def main():

    print("\n===================================")
    print(" IRIS FLOWER CLASSIFICATION PROJECT ")
    print("===================================\n")

    # -----------------------------------
    # STEP 1 : LOAD DATASET
    # -----------------------------------

    iris = load_iris()

    iris_df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    # Add species names
    iris_df['species'] = iris.target

    species_map = {
        0: 'Setosa',
        1: 'Versicolor',
        2: 'Virginica'
    }

    iris_df['species_name'] = iris_df['species'].map(species_map)

    print(" Dataset Loaded Successfully\n")

    print("First 5 Rows:\n")
    print(iris_df.head())

    print("\n-----------------------------------")
    print("Dataset Information")
    print("-----------------------------------\n")

    print(iris_df.info())

    print("\n-----------------------------------")
    print("Statistical Summary")
    print("-----------------------------------\n")

    print(iris_df.describe())

    # -----------------------------------
    # STEP 2 : DATA VISUALIZATION
    # -----------------------------------

    sns.set_style("whitegrid")

    # ---------------- SCATTER PLOT ----------------

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=iris_df,
        x='sepal length (cm)',
        y='petal length (cm)',
        hue='species_name',
        s=100
    )

    plt.title("Scatter Plot")
    plt.savefig("scatter_plot.png")
    plt.show()

    # ---------------- HISTOGRAM ----------------

    iris_df.hist(figsize=(10, 8))

    plt.suptitle("Histogram of Features")
    plt.savefig("histogram.png")
    plt.show()

    # ---------------- PAIRPLOT ----------------

    pair_plot = sns.pairplot(
        iris_df[
            [
                'sepal length (cm)',
                'sepal width (cm)',
                'petal length (cm)',
                'petal width (cm)',
                'species_name'
            ]
        ],
        hue='species_name',
        palette='Set1'
    )

    pair_plot.fig.suptitle(
        "Pairplot of Iris Dataset",
        y=1.02
    )

    plt.savefig("pairplot.png")
    plt.show()

    # ---------------- CORRELATION HEATMAP ----------------

    plt.figure(figsize=(8, 6))

    correlation = iris_df.iloc[:, 0:4].corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap='coolwarm'
    )

    plt.title("Correlation Heatmap")

    plt.savefig("correlation_heatmap.png")
    plt.show()

    # -----------------------------------
    # STEP 3 : PREPARE DATA
    # -----------------------------------

    X = iris_df[
        [
            'sepal length (cm)',
            'sepal width (cm)',
            'petal length (cm)',
            'petal width (cm)'
        ]
    ]

    y = iris_df['species']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Feature Scaling

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------------
    # STEP 4 : TRAIN MODEL
    # -----------------------------------

    print("\nTraining Model...\n")

    model = LogisticRegression(max_iter=200)

    model.fit(X_train, y_train)

    # -----------------------------------
    # STEP 5 : PREDICTION
    # -----------------------------------

    y_pred = model.predict(X_test)

    # -----------------------------------
    # STEP 6 : ACCURACY
    # -----------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    print("-----------------------------------")
    print(f"Model Accuracy : {accuracy * 100:.2f}%")
    print("-----------------------------------\n")

    # -----------------------------------
    # STEP 7 : CLASSIFICATION REPORT
    # -----------------------------------

    target_names = [
        'Setosa',
        'Versicolor',
        'Virginica'
    ]

    print("Classification Report:\n")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=target_names
        )
    )

    # -----------------------------------
    # STEP 8 : CONFUSION MATRIX
    # -----------------------------------

    plt.figure(figsize=(8, 6))

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")

    plt.show()

    # -----------------------------------
    # FINAL MESSAGE
    # -----------------------------------

    print("\n===================================")
    print(" PROJECT COMPLETED SUCCESSFULLY ")
    print("===================================\n")

    print("Generated Graph Files:\n")

    print("1. scatter_plot.png")
    print("2. histogram.png")
    print("3. pairplot.png")
    print("4. correlation_heatmap.png")
    print("5. confusion_matrix.png\n")


# -----------------------------------
# MAIN FUNCTION
# -----------------------------------

if __name__ == "__main__":
    main()