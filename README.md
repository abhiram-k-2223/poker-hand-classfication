# Poker Hand Classification

This project aims to classify poker hands using machine learning models. The dataset used is a poker hand dataset, and the target variable is 'CLASS', which indicates the type of poker hand. The workflow includes data loading, cleaning, preprocessing, and training models using Support Vector Classification (SVC) and Logistic Regression.

## Project Description

### Data Loading

The dataset is loaded using pandas, and an initial exploration is done using the `head()` method to understand the structure and content of the data.

### Data Cleaning

1. **Handling Missing Values**: Any missing values in the dataset are dropped to ensure clean data for model training.
2. **Data Type Conversion**: Specific columns ('S3', 'C3', 'S4', 'C4', 'S5', 'C5') are converted to integers to ensure correct data types for modeling.

### Data Exploration

1. **Unique Values**: The unique values in the 'S2' column are explored to understand the range of values.
2. **Dataset Information**: The structure and types of the dataset columns are reviewed using `df.info()` to ensure correctness after cleaning.

### Data Splitting

The dataset is split into training and testing sets using a 70-30 split. The target variable 'CLASS' is separated from the feature set.

### Model Training and Evaluation

#### Support Vector Classification (SVC)

1. **Model Training**: An SVC model with a linear kernel and regularization parameter C=1 is trained on the training data.
2. **Prediction**: The trained SVC model is used to predict the 'CLASS' on the test data.
3. **Evaluation**: The model's performance is evaluated using the accuracy score, indicating the proportion of correct predictions.

#### Logistic Regression

1. **Model Training**: A logistic regression model with multinomial classification, using the 'lbfgs' solver and a high iteration limit (100,000), is trained on the training data.
2. **Prediction**: The trained logistic regression model is used to predict the 'CLASS' on the test data.
3. **Evaluation**: The model's performance is evaluated using the accuracy score, similar to the SVC model.

### Comparison of Models

Both models' accuracy scores are compared to determine which model performs better on the poker hand classification task.

## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/poker-hand-classification.git
    cd poker-hand-classification
    ```

2. **Install dependencies**:
    ```bash
    pip install pandas scikit-learn
    ```

3. **Run the Jupyter notebook or script**:
    - If using a Jupyter notebook, open `multiclass_classification2.ipynb` and run the cells sequentially.
    - If using a Python script, ensure all code is included in a script and run:

4. **Output**:
    - Accuracy scores of both SVC and Logistic Regression models, indicating their performance on the test set.

This project demonstrates how to preprocess data, train multiple classification models, and evaluate their performance for the task of poker hand classification.
