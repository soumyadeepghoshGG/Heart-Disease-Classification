# ML-metrics-Generator


## Overview

Classes `ClassificationModelEvaluator` and `RegressionModelEvaluator` are designed for evaluating and comparing the performance of various classification and regression models respectively. They provide fast, easy and simultaneous training and testing of different algorithms, reporting and comparing their performances to find most suited algorithm for the task.

  
## ClassificationModelEvaluator Methods
-  `display_classification_metrics`: Displays a DataFrame with training and testing metrics of algorithms used.
-  `display_classification_report`: Displays classification reports of the algorithms used.
-  `show_confusion_matrix`: Shows confusion matrices for each algorithm.
-  `show_roc_curve`: Shows ROC curves with AUC for each algorithm.
-  `display_cross_validation_report`: Displays a cross-validation comparison chart. (Optional parameter: `n` for the number of folds)


## RegressionModelEvaluator Methods
-  `display_regression_metrics`: Displays a DataFrame with training and testing metrics of algorithms used.
  

## Parameters
(same for both classes)
-  `features`: Feature matrix.
-  `target`: Target variable.
-  `*args, **kwargs`: Classification models to evaluate.


## Usage

```bash
# Example Usage
from  MLmetricsGen  import  ClassificationModelEvaluator

from  sklearn.linear_model  import  LogisticRegression
from  sklearn.naive_bayes  import  GaussianNB


# Models to test
models  = [LogisticRegression(), GaussianNB()]

# Assuming you have your feature matrix X and target variable y
evaluator  =  ClassificationModelEvaluator(X, y, *models)

# Display a report with training and testing metrics
report  =  evaluator.display_report()

# Display classification reports for each algorithm
evaluator.display_classification_report()

# Show confusion matrices for each algorithm
evaluator.show_confusion_matrix()

# Show ROC curves for each algorithm
evaluator.show_roc_curve()

# Display cross-validation comparison chart (optional:  n=10)
cv_report  =  evaluator.display_cross_validation_report()
```