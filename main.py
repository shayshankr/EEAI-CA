from data_preprocessing import load_and_preprocess_data
from traning_model.logistic_regression_model import LogisticRegressionChained
from traning_model.random_forest_model import RandomForestChained


class MLController:
    """
    The MLController class is responsible for managing the machine learning process.
    It includes model selection, data loading, model training, prediction and accuracy calculation.

    Attributes:
        data_model (str): The dataset to use for training and testing (default is "app_gallery").
        model_type (str): The type of model to train and evaluate (default is "random_forest").
        model (object): The machine learning model instance (RandomForestChained or LogisticRegressionChained).
    """

    def __init__(self, data_model="app_gallery", model_type="random_forest"):
        """
        Initializes the MLController with the specified dataset and model type.

        Args:
            data_model (str): The name of the dataset to use. Options are "app_gallery" and "purchasing".
            model_type (str): The type of model to use. Options are "random_forest" and "logistic_regression".
        """
        self.model_type = model_type
        self.model = None
        self.data_model = data_model


    def get_requested_model(self):
        """
        Returns the model class based on the selected model type.

        Returns:
            class: A machine learning model class (either RandomForestChained or LogisticRegressionChained).
        """
        models = {"random_forest": RandomForestChained, "logistic_regression": LogisticRegressionChained}
        return models.get(self.model_type)


    def load_model_data(self):
        """
        Returns the file path to the dataset based on the selected data model.

        Returns:
            str: File path to the dataset ("data/AppGallery.csv" or "data/Purchasing.csv").
        """
        data_dump = {"app_gallery": "data/AppGallery.csv", "purchasing": "data/Purchasing.csv"}
        return data_dump.get(self.data_model)


    def get_model_name(self):
        """
        Returns the human-readable model name by formatting the model type string.

        Returns:
            str: The formatted model name ("Random Forest" or "Logistic Regression").
        """
        return self.model_type.replace("_", " ").title()


    def get_data_dump_name(self):
        """
        Returns the human-readable dataset name by formatting the data model string.

        Returns:
            str: The formatted dataset name ("App Gallery" or "Purchasing").
        """
        return self.data_model.replace("_", " ").title()


    def train_and_calculate_accuracy(self):
        """
        Loads the dataset, trains the selected model, makes predictions and calculates the final accuracy of the model using the test set.

        The steps followed in this function include:
        1. Loading and preprocessing the dataset.
        2. Training the model on the training data.
        3. Making predictions on the test data.
        4. Calculating the final accuracy based on chained predictions (Type 2, Type 3 and Type 4).
        """
        # Load and initialize the selected model
        loaded_model = self.get_requested_model()()

        # Load and preprocess the data
        (x_train, x_test, y2_train, y2_test, y3_train, y3_test, y4_train, y4_test), _ = load_and_preprocess_data(self.load_model_data())

        # Train the model with the training data
        loaded_model.fit(x_train, y2_train, y3_train, y4_train)

        # Make predictions on the test data
        (y2_pred, y3_pred, y4_pred, correct_mask, correct_mask2, y3_test_correct, y4_test_correct,) = loaded_model.predict(x_test, y2_test, y3_test, y4_test)

        # Print the evaluation metrics of the model
        print(f"\nEvaluation metrics for {self.get_model_name()} for {self.get_data_dump_name()}")

        # Evaluate the model's accuracy using the predictions
        final_accuracy = loaded_model.evaluate(y2_test, y2_pred, y3_test_correct, y3_pred, y4_test_correct, y4_pred, correct_mask, correct_mask2)

        # Print the final accuracy of the model
        print(f"Final Chained Accuracy for {self.get_model_name()} for {self.get_data_dump_name()} is: {final_accuracy:.2%}")


# Main execution block
if __name__ == "__main__":

    # Train and evaluate the model for the 'app_gallery' dataset using Random Forest
    trainer_controller = MLController(data_model="app_gallery", model_type="random_forest")
    trainer_controller.train_and_calculate_accuracy()

    # Train and evaluate the model for the 'app_gallery' dataset using Logistic Regression
    trainer_controller = MLController(data_model="app_gallery", model_type="logistic_regression")
    trainer_controller.train_and_calculate_accuracy()

    # Train and evaluate the model for the 'purchasing' dataset using Random Forest
    trainer_controller = MLController(data_model="purchasing", model_type="random_forest")
    trainer_controller.train_and_calculate_accuracy()

    # Train and evaluate the model for the 'purchasing' dataset using Logistic Regression
    trainer_controller = MLController(data_model="purchasing", model_type="logistic_regression")
    trainer_controller.train_and_calculate_accuracy()
