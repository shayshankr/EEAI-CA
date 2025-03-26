from abc import ABC, abstractmethod


class ModelInterface(ABC):
    """
    This interface ensures that all models implementing it have a consistent
    set of methods that can be used for training, prediction, evaluation and
    generating classification reports.

    It uses Python's Abstract Base Class (ABC) module to define an abstract
    class with abstract methods that must be implemented by any subclass.
    """

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Abstract method for training the model on the given data.

        Args:
            *args: Positional arguments to be passed to the model's fit method.
            **kwargs: Keyword arguments to be passed to the model's fit method.

        This method should implement the logic to train the model on the provided data.
        """
        pass


    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        Abstract method for predicting labels on new data.

        Args:
            *args: Positional arguments to be passed to the model's predict method.
            **kwargs: Keyword arguments to be passed to the model's predict method.

        This method should implement the logic to generate predictions for the provided input data.
        """
        pass


    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """
        Abstract method for evaluating the model's performance.

        Args:
            *args: Positional arguments for evaluation data (e.g., true labels and predicted labels).
            **kwargs: Keyword arguments for evaluation settings or parameters.

        This method should return metrics that represent the performance of the model (e.g., accuracy, F1 score).
        """
        pass


    @abstractmethod
    def generate_classification_report(self, *args, **kwargs):
        """
        Abstract method for generating a classification report (e.g., precision, recall, F1-score).

        Args:
            *args: Positional arguments, including true labels and predicted labels.
            **kwargs: Keyword arguments, such as options for formatting or additional settings.

        This method should generate and print a classification report summarizing the performance of the model.
        """
        pass
