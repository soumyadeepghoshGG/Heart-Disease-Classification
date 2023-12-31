# Importing Packages 
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# Ignore all warnings
import warnings
warnings.filterwarnings(action = 'ignore')




# Train models
class ModelTrainer:
    def __init__(self, features, target, *args, **kwargs):
        self.X = features
        self.y = target
        self.kwargs = kwargs
        self.args = args
        self.fitted_models = None  # Store the trained models


    def _model_name_extractor(self):
        model_name = []

        for arg in self.args:
            model_name.append(arg.__class__.__name__)
        for name, _ in self.kwargs.items():
            model_name.append(name)

        return model_name


    def _trainTestSplit(self):
        return train_test_split(self.X, self.y, test_size=0.3, random_state=9)


    def _model_train(self):
        if self.fitted_models is None:
            x_train, _, y_train, _ = self._trainTestSplit()
            self.fitted_models = []

            for arg in self.args:
                self.fitted_models.append(arg.fit(x_train, y_train))

            for _, model in self.kwargs.items():
                self.fitted_models.append(model.fit(x_train, y_train))

        return self.fitted_models
    



# Classifier Performance
class ClassificationModelEvaluator(ModelTrainer):
    # Use this Method to get a DataFrame with training and testing metrics of algorithms used
    def display_classification_metrics(self):
        x_train, x_test, y_train, y_test = self._trainTestSplit()
        fitted_models = self._model_train()
        index = self._model_name_extractor()

        # Empty DataFrame to store report
        report = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
        
        for name, model in zip(index, fitted_models):
            # Performance for training data
            accuracy_train = accuracy_score(y_train, model.predict(x_train))
            precision_train = precision_score(y_train, model.predict(x_train))
            recall_train = recall_score(y_train, model.predict(x_train))
            f1_train = f1_score(y_train, model.predict(x_train))

            # Performance for testing data
            accuracy_test = accuracy_score(y_test, model.predict(x_test))
            precision_test = precision_score(y_test, model.predict(x_test))
            recall_test = recall_score(y_test, model.predict(x_test))
            f1_test = f1_score(y_test, model.predict(x_test)) 

            # Create a DataFrame with metrics
            train_index = name.title() + " Training Set"
            report.loc[train_index] = [accuracy_train, precision_train, recall_train, f1_train]
            test_index = name.title() + " Testing Set"
            report.loc[test_index] = [accuracy_test, precision_test, recall_test, f1_test]
        
        return report
        

    # Use this Method to get classification report of the algorithms used
    def display_classification_report(self):
        _, x_test, _, y_test = self._trainTestSplit()
        fitted_models = self._model_train()
        index = self._model_name_extractor()

        for name, model in zip(index, fitted_models):
            print(f"Classification Report for {name}:")
            print(classification_report(y_test, model.predict(x_test)))

    
    # Use this Method for Confusion Matrix of algorithms used
    def show_confusion_matrix(self):
        _, x_test, _, y_test = self._trainTestSplit()
        fitted_models = self._model_train()
        index = self._model_name_extractor()

        for name, model in zip(index, fitted_models):
            conf_matrix = confusion_matrix(y_test, model.predict(x_test))

            # Generating Heatmap for Confusion Matrix
            plt.figure(figsize=(10, 7))
            plt.title(f"Confusion Matrix for {name}", fontdict={"fontweight": "bold", "color": 'k', "fontsize": 14})
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlOrBr')
            plt.xlabel("Predicted Classes", fontdict={"fontsize": 12})
            plt.ylabel("Actual Classes", fontdict={"fontsize": 12})
            plt.show()


    # Use this method to get ROC curve of algorithms used
    def show_ruc_curve(self):
        _, x_test, _, y_test = self._trainTestSplit()
        fitted_models = self._model_train()
        index = self._model_name_extractor()

        for name, model in zip(index, fitted_models):
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
            roc_auc = auc(fpr, tpr)

            # Plot for ROC with AUC
            plt.style.use('classic')
            plt.figure(figsize=(10, 7))
            plt.title(f'\n\nReceiver Operating Characteristic Curve for {name}\n')
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.show()


    # Use this method to get cross validation score for the algorithms used
    def display_cross_validation_report(self, n=10):
        print("CROSS VALIDATION USING F1 SCORES\n\n")
        fitted_models = self._model_train()
        index = self._model_name_extractor()
        
        # Empty DataFrame to store scores
        cols = [f"CV{i}" for i in range(1, n+1)]+['CV Mean', 'CV Std Dev']
        report = pd.DataFrame(columns=cols)

        k_fold = StratifiedKFold(n_splits=n, shuffle=True, random_state=7)

        for name, model in zip(index, fitted_models):
            cv_results = cross_val_score(model, self.X, self.y, cv=k_fold, scoring='f1', error_score='raise')
            cv_results = np.append(cv_results, [np.mean(cv_results), np.std(cv_results)])
            report.loc[name] = cv_results

        # Plot the CV-Mean Scores of different algorithms
        plt.style.use("dark_background")
        plt.figure(figsize=[10,7])
        plt.title("Cross-Validation Comparison Chart")
        sns.barplot(report["CV Mean"], palette="cubehelix")
        plt.ylabel("Cross-Validation Mean")
        plt.xticks(rotation=90)
        plt.show()

        return report




# Regression Performance
class RegressionModelEvaluator(ModelTrainer):
    # Us this method to get an extensive report of model performance
    def display_regression_metrics(self):
        x_train, x_test, y_train, y_test = self._trainTestSplit()
        fitted_models = self._model_train()
        index = self._model_name_extractor()

        # Empty DataFrame to store report
        report = pd.DataFrame(columns=["Mean Squared Error", "Root Mean Squared Error", "Mean Absolute Error", "Mean Absolute Percentage Error", "R^2 Score"])
        
        training_actual = y_train
        testing_actual = y_test

        for name, model in zip(index, fitted_models):
            training_predicted = model.predict(x_train)
            testing_predicted = model.predict(x_test)

            # Performance of Training Data according to our model:
            MSE_train = mean_squared_error(training_actual, training_predicted)
            RMSE_train = np.sqrt(mean_squared_error(training_actual, training_predicted))
            MAE_train = mean_absolute_error(training_actual, training_predicted)
            MAPE_train = mean_absolute_percentage_error(training_actual, training_predicted)
            R_Squared_score_train = r2_score(training_actual, training_predicted)

            # Performance of Testing Data according to our model:
            MSE_test = mean_squared_error(testing_actual, testing_predicted)
            RMSE_test = np.sqrt(mean_squared_error(testing_actual, testing_predicted))
            MAE_test = mean_absolute_error(testing_actual, testing_predicted)
            MAPE_test = mean_absolute_percentage_error(testing_actual, testing_predicted)
            R_Squared_score_test = r2_score(testing_actual, testing_predicted)
            
            # Putting the performance data into a DataFrame to print it.
            train_index = name.title() + " Training Set"
            report.loc[train_index] = [MSE_train, RMSE_train, MAE_train, MAPE_train, R_Squared_score_train]
            test_index = name.title() + " Testing Set"
            report.loc[test_index] = [MSE_test, RMSE_test, MAE_test, MAPE_test, R_Squared_score_test]

        return report