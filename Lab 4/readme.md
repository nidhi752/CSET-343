# Lab Assignment 4: Logistic Regression for Breast Cancer Classification

## Course Details

* **Course:** AI in Healthcare
* **Course Code:** CSET343
* **Program:** B.Tech
* **Semester:** VII
* **Lab Assignment:** 4

---

## Objective

The objective of this lab is to implement and evaluate **Logistic Regression** for a binary classification problem using the **Breast Cancer Wisconsin (Diagnostic) Dataset**.

The lab focuses on:

* Data acquisition and exploration
* Data cleaning and preprocessing
* Outlier detection and handling
* Data visualization
* Feature standardization
* Logistic Regression model training
* Model evaluation using multiple classification metrics
* ROC curve and AUC analysis
* Interpretation of model performance in a healthcare context

---

## Dataset

The **Breast Cancer Wisconsin (Diagnostic)** dataset available through `scikit-learn` is used in this experiment.

The dataset contains numerical features extracted from digitized images of breast mass samples.

### Target Encoding

For easier medical interpretation, the target variable is represented as:

```text
Outcome = 0 → Benign
Outcome = 1 → Malignant
```

Malignant cases are treated as the positive class because detecting cancer correctly is the more clinically important classification task.

---

## Dataset Features

The dataset contains **30 numerical features**, including measurements related to:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Concave points
* Symmetry
* Fractal dimension

These measurements are available as mean values, standard-error values, and worst-case values.

---

## Libraries Used

The project uses the following Python libraries:

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## Task 1: Data Acquisition and Exploration

The Breast Cancer Wisconsin dataset is loaded using:

```python
from sklearn.datasets import load_breast_cancer
```

The following exploratory steps are performed:

* Dataset dimensions and structure are inspected.
* Summary statistics including mean, median, standard deviation, minimum, and maximum are calculated.
* Missing values are checked.
* Duplicate records are detected and removed if present.
* Outliers are detected using the **Interquartile Range (IQR)** method.
* Histograms are plotted for important features.
* Box plots are used to compare features across benign and malignant classes.
* A correlation heatmap is generated.
* Class distribution is visualized and analyzed.

---

## Outlier Handling

Outliers are detected using the IQR rule:

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Since extreme medical measurements may contain useful diagnostic information, outlier rows are not simply deleted.

Instead, **IQR-based capping** is applied.

The IQR limits are calculated only from the training data and then applied to both training and testing data to prevent **data leakage**.

---

## Class Distribution

The dataset contains more benign cases than malignant cases.

Approximately:

```text
Benign    ≈ 63%
Malignant ≈ 37%
```

This represents a moderate class imbalance.

Therefore, accuracy alone is not sufficient for evaluating the model. Metrics such as **precision, recall, F1-score, confusion matrix, and ROC-AUC** are also considered.

For medical diagnosis, **recall for malignant cases** is especially important because a false negative could result in a cancer case being incorrectly classified as benign.

---

## Task 2: Data Preprocessing

### Train-Test Split

The dataset is divided into:

```text
80% Training Data
20% Testing Data
```

A **stratified split** is used so that the proportion of benign and malignant samples remains approximately equal in both datasets.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
```

---

## Feature Standardization

The features have significantly different numerical scales.

Therefore, `StandardScaler` is used:

```python
from sklearn.preprocessing import StandardScaler
```

Standardization transforms features approximately to:

```text
Mean = 0
Standard Deviation = 1
```

This is important because Logistic Regression and its regularization terms are sensitive to differences in feature scale.

---

## Task 3: Logistic Regression

The model is trained using:

```python
from sklearn.linear_model import LogisticRegression
```

The standardized training data is used to fit the classifier.

---

## Model Evaluation

The trained model is evaluated using the following metrics:

### Accuracy

Measures the percentage of correctly classified samples.

### Precision

Measures how many samples predicted as malignant were actually malignant.

```text
Precision = TP / (TP + FP)
```

### Recall

Measures how many actual malignant cases were correctly detected.

```text
Recall = TP / (TP + FN)
```

Recall is particularly important in cancer diagnosis because a low recall value means the model is missing malignant cases.

### F1-Score

The F1-score balances precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## Confusion Matrix

The confusion matrix contains:

```text
True Positive  → Malignant correctly identified
True Negative  → Benign correctly identified
False Positive → Benign predicted as malignant
False Negative → Malignant predicted as benign
```

In a medical diagnosis system, **false negatives are particularly dangerous** because they may delay further diagnosis or treatment.

---

## ROC Curve and AUC

The **Receiver Operating Characteristic (ROC)** curve is used to evaluate model performance across different classification thresholds.

The graph compares:

```text
True Positive Rate
vs.
False Positive Rate
```

The **Area Under the Curve (AUC)** summarizes the model's ability to distinguish malignant samples from benign samples.

```text
AUC ≈ 1.0 → Excellent classification
AUC ≈ 0.5 → Similar to random guessing
```

---

## Regularization

The notebook also compares:

* **L1 Regularization**
* **L2 Regularization**

### L1 Regularization

L1 can reduce some coefficients exactly to zero and can therefore perform a form of feature selection.

### L2 Regularization

L2 reduces the magnitude of model coefficients and helps control overfitting.

Regularization is particularly useful because several features in the Breast Cancer dataset are strongly correlated.

---

## Model Interpretation

Logistic Regression coefficients are analyzed to understand which features influence the prediction.

For the encoding used in this experiment:

```text
Positive coefficient → increases predicted probability of malignancy
Negative coefficient → decreases predicted probability of malignancy
```

Since all features are standardized, coefficient magnitudes can be compared more meaningfully.

However, coefficient values should not automatically be interpreted as causal relationships because many features in the dataset are correlated.

---

## Files

```text
Lab_Assignment_4_Logistic_Regression_Colab.ipynb
README.md
```

---

## How to Run

1. Open **Google Colab**.
2. Select **File → Upload Notebook**.
3. Upload:

```text
Lab_Assignment_4_Logistic_Regression_Colab.ipynb
```

4. Select:

```text
Runtime → Run all
```

5. The notebook will automatically load the dataset, perform preprocessing, train the model, and display all required evaluation results and graphs.

No manual dataset download is required because the dataset is loaded directly from `scikit-learn`.

---

## Conclusion

This experiment demonstrates the complete application of Logistic Regression to a healthcare classification problem.

The workflow covers data exploration, cleaning, outlier handling, visualization, preprocessing, model training, and evaluation.

The model achieves strong classification performance on the Breast Cancer Wisconsin dataset. However, in medical diagnosis, model performance should not be judged only through accuracy. Particular attention must be given to **malignant-class recall and false negatives**, because incorrectly classifying a malignant tumor as benign can have serious clinical consequences.

The experiment also demonstrates how standardization and regularization improve the reliability and interpretability of Logistic Regression models.
