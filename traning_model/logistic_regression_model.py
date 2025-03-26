import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from traning_model.model_interface import ModelInterface


class LogisticRegressionChained(ModelInterface):
    """
    This class implements a chained approach for multi-stage classification using logistic regression models.
    It predicts in a sequential manner where predictions from one stage (y2) are used as the basis for the next stage (y3),
    and predictions from the next stage (y3) are used for the final stage (y4).
    """

    def __init__(self):
        """
        Initializes the three logistic regression models (one for each stage of the prediction).
        """
        # Logistic Regression models for each stage (y2, y3, y4)
        self.lr_type2 = LogisticRegression(max_iter=1000, random_state=42)
        self.lr_type3 = LogisticRegression(max_iter=1000, random_state=42)
        self.lr_type4 = LogisticRegression(max_iter=1000, random_state=42)


    def fit(self, x_train, y2_train, y3_train, y4_train):
        """
        Fits the logistic regression models for each of the stages (y2, y3, y4) on the training data.

        Args:
            x_train: The feature matrix for training.
            y2_train: The target labels for stage 2 classification.
            y3_train: The target labels for stage 3 classification.
            y4_train: The target labels for stage 4 classification.
        """
        # Train the logistic regression models for each stage
        self.lr_type2.fit(x_train, y2_train)
        self.lr_type3.fit(x_train, y3_train)
        self.lr_type4.fit(x_train, y4_train)


    def predict(self, x_test, y2_test, y3_test, y4_test):
        """
        Makes predictions for each stage of the classification and returns the results.

        Args:
            x_test: The feature matrix for testing.
            y2_test: The true labels for stage 2 in the test data.
            y3_test: The true labels for stage 3 in the test data.
            y4_test: The true labels for stage 4 in the test data.

        Returns:
            Tuple containing:
                - y2_pred: Predictions from stage 2.
                - y3_pred: Predictions from stage 3.
                - y4_pred: Predictions from stage 4.
                - correct_mask: Mask of correct predictions for stage 2.
                - correct_mask2: Mask of correct predictions for stage 3.
                - y3_test_correct: Correct test data for stage 3.
                - y4_test_correct: Correct test data for stage 4.
        """
        # Stage 2 predictions
        y2_pred = self.lr_type2.predict(x_test)
        correct_mask = y2_pred == y2_test
        x_test_correct, y3_test_correct = x_test[correct_mask], y3_test[correct_mask]

        # Stage 3 predictions on correctly predicted data from stage 2
        y3_pred = self.lr_type3.predict(x_test_correct)
        correct_mask2 = y3_pred == y3_test_correct
        x_test_correct2, y4_test_correct = (x_test_correct[correct_mask2], y4_test[correct_mask][correct_mask2])

        # Stage 4 predictions on correctly predicted data from stages 2 and 3
        y4_pred = self.lr_type4.predict(x_test_correct2)

        # Return predictions
        return (y2_pred, y3_pred, y4_pred, correct_mask, correct_mask2, y3_test_correct, y4_test_correct)


    def evaluate(self, y2_test, y2_pred, y3_test_correct, y3_pred, y4_test_correct, y4_pred, correct_mask, correct_mask2):
        """
        Evaluates the chained classification model by calculating accuracy for each stage
        and printing a classification report for the overall prediction.

        Args:
            y2_test: True labels for stage 2 in the test data.
            y2_pred: Predicted labels for stage 2.
            y3_test_correct: Correct test labels for stage 3.
            y3_pred: Predicted labels for stage 3.
            y4_test_correct: Correct test labels for stage 4.
            y4_pred: Predicted labels for stage 4.
            correct_mask: Mask indicating which stage 2 predictions were correct.
            correct_mask2: Mask indicating which stage 3 predictions were correct.

        Returns:
            final_accuracy: The chained accuracy considering all stages.
        """

        def calculate_chained_accuracy(y2_true, y2_pred, y3_true, y3_pred, y4_true, y4_pred):
            """
            Calculates the chained accuracy by taking into account the sequential nature of the predictions across stages (y2, y3, y4).

            Args:
                y2_true: True labels for stage 2.
                y2_pred: Predicted labels for stage 2.
                y3_true: True labels for stage 3.
                y3_pred: Predicted labels for stage 3.
                y4_true: True labels for stage 4.
                y4_pred: Predicted labels for stage 4.

            Returns:
                float: The average accuracy across the three stages.
            """
            accuracies = []
            print("\nClassification report for combined predictions (Type2 + Type3 + Type4):")
            
            # Generate and print classification report for all predictions
            self.generate_classification_report(y2_true + y3_true + y4_true, y2_pred + y3_pred + y4_pred)
            print("Note: This accuracy is not the same as chained accuracy.\n")

            # Calculate accuracy for each instance in the test data based on correct predictions at each stage
            for i in range(len(y2_true)):
                if y2_true[i] == y2_pred[i]:
                    if y3_true[i] == y3_pred[i]:
                        if y4_true[i] == y4_pred[i]:
                            accuracies.append(1.0)  # All stages correct
                        else:
                            accuracies.append(0.67)  # Only stages 2 and 3 correct
                    else:
                        accuracies.append(0.33)  # Only stage 2 correct
                else:
                    accuracies.append(0.0)  # No stage correct

            # Return the mean accuracy across all instances
            return np.mean(accuracies)

        # Calculate and return final chained accuracy
        final_accuracy = calculate_chained_accuracy(
            y2_test[correct_mask][correct_mask2].values,
            y2_pred[correct_mask][correct_mask2],
            y3_test_correct[correct_mask2].values,
            y3_pred[correct_mask2],
            y4_test_correct.values,
            y4_pred)
        return final_accuracy


    def generate_classification_report(self, y_true, y_pred):
        """
        Generates and prints the classification report (precision, recall, f1-score) for the overall predictions.

        Args:
            y_true: True labels for the test data.
            y_pred: Predicted labels for the test data.
        """
        report = classification_report(y_true, y_pred, zero_division=0)
        print(report)
