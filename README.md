# Heart Disease Dataset 

## Installation

To run the project, you'll need to have the following Python packages installed. You can install them using the following command:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn xgboost
```

## About Dataset

### Context
This dataset dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.

- **Dataset Characteristics:** Multivariate
- **Subject Area:** Health and Medicine
- **Associated Tasks:** Classification
- **Feature Type:** Categorical, Integer, Real
- **Number of Records:** 1025
- **Number of Features:** 13

### Features

| Feature    | Description                                                                                                      |
|------------|------------------------------------------------------------------------------------------------------------------|
| **age**     | The age of the patient in years.                                                                                 |
| **sex**     | The sex of the patient (1 = male, 0 = female).                                                                   |
| **cp**      | The type of chest pain the patient experienced (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic). |
| **trestbps**| The resting blood pressure of the patient in mm Hg.                                                              |
| **chol**    | The serum cholesterol level of the patient in mg/dl.                                                             |
| **fbs**     | The fasting blood sugar level of the patient, measured in mg/dl (1 = high, 0 = low).                            |
| **restecg** | The resting electrocardiographic results of the patient (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy)  |
| **thalach** | The maximum heart rate achieved by the patient during exercise.                                                    |
| **exang**   | Whether the patient experienced exercise-induced angina (1 = yes, 0 = no).                                       |
| **oldpeak** | The ST depression induced by exercise relative to rest.                                                            |
| **slope**   | The slope of the ST segment during peak exercise (1 = upsloping, 2 = flat, 3 = downsloping).  |
| **ca**      | The number of major vessels colored by fluoroscopy (0-3).   |
| **thal**    | The type of thallium scan performed on the patient (1 = fixed defect, 2 = reversible defect, 3 = normal).        |

**NOTE**<br>
1. Resting electrocardiographic (ECG or EKG) is a non-invasive diagnostic test that records the electrical activity of the heart while the patients at rest. The test is performed using an electrocardiogram machine, which records the electrical signals produced by the heart through electrodes placed on the chest, arms, and legs.
2. ST depression induced by exercise relative to rest. Oldpeak, also known as ST depression, is a common parameter measured during an exercise stress test to evaluate the presence and severity of coronary artery disease. It represents the amount of ST segment depression that occurs on an electrocardiogram (ECG) during exercise compared to rest.
3. The number of major vessels (0-3) colored by fluoroscopy is a parameter that is used to assess the severity of coronary artery disease (CAD) in patients who undergo coronary angiography.

### Project Target and Evaluation Metrics:
Primary objective of this project is to find the best classification algorithm for our dataset. I have listed below the classifiers I have tried in this project. In the context of our medical application, the goal is to develop a robust and accurate model that minimizes both false positives and false negatives. Achieving high sensitivity is crucial to ensure the identification of all positive cases, minimizing the risk of missing critical instances. Simultaneously, precision is of utmost importance to reduce the repercussions of false positives, especially in scenarios where misclassifications could have severe consequences. The project aims to strike a balance between precision and recall, acknowledging the inherent trade-off between the two. The precision-recall trade-off is a key consideration, as increasing sensitivity often comes at the cost of reduced precision and vice versa. The choice of the appropriate threshold for classification will be critical in optimizing the model's performance based on the specific needs of our medical task. Our evaluation metrics, including sensitivity, specificity, precision, recall, F1 score, and ROC-AUC, will guide the assessment of the model's effectiveness in achieving these objectives.

### Algorithms Implemented
- Decision Tree
- K-Nearest Neighbor
- Logistic Regression
- Linear Discriminant Analysis
- Naive Bayes Classifier
- Random Forest
- Support Vector Machine
- XG Boost

## Reference
By R. Detrano, A. Jánosi, W. Steinbrunn, M. Pfisterer, J. Schmid, S. Sandhu, K. Guppy, S. Lee, V. Froelicher. 1989<br>
Published in American Journal of Cardiology.<br>
[International application of a new probability algorithm for the diagnosis of coronary artery disease.](https://www.semanticscholar.org/paper/International-application-of-a-new-probability-for-Detrano-J%C3%A1nosi/a7d714f8f87bfc41351eb5ae1e5472f0ebbe0574)

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - see the [LICENSE](https://github.com/soumyadeepghoshGG/Heart-Disease-Classification/blob/main/License.txt) file for details.

## Contact
For questions or issues, please contact me (Soumyadeep Ghosh) via mail: soumyadeepghosh57@gmail.com