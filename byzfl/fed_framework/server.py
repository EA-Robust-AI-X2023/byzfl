from loguru import logger
import torch

from byzfl.fed_framework import ModelBaseInterface, RobustAggregator
from byzfl.utils.conversion import flatten_dict
import numpy as np

class Server(ModelBaseInterface):
    
    def __init__(self, params):
        # Check for correct types and values in params
        if not isinstance(params, dict):
            raise TypeError(f"'params' must be of type dict, but got {type(params).__name__}")
        if not isinstance(params["test_loader"], torch.utils.data.DataLoader):
            raise TypeError(f"'test_loader' must be a DataLoader, but got {type(params['test_loader']).__name__}")

        # Initialize the Server instance
        super().__init__({
            "device": params["device"],
            "model_name": params["model_name"],
            "optimizer_name": params["optimizer_name"],
            "optimizer_params": params.get("optimizer_params", {}),
            "learning_rate": params["learning_rate"],
            "weight_decay": params["weight_decay"],
            "milestones": params["milestones"],
            "learning_rate_decay": params["learning_rate_decay"],
            "isServer": True,
        })
        self.robust_aggregator = RobustAggregator(params["aggregator_info"], params["pre_agg_list"])
        self.test_loader = params["test_loader"]
        self.validation_loader = params.get("validation_loader")
        if self.validation_loader is not None:
            if not isinstance(params["validation_loader"], torch.utils.data.DataLoader):
                raise TypeError(f"'validation_loader' must be a DataLoader, but got {type(params['validation_loader']).__name__}")

        self.model.eval()

    def aggregate(self, vectors):
        """
        Description
        -----------
        Aggregates input vectors using the configured robust aggregator.

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A collection of input vectors to be aggregated.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Aggregated output vector, with the same type as the input vectors.
        """
        return self.robust_aggregator.aggregate_vectors(vectors)

    def update_model_with_gradients(self, gradients):
        """
        Description
        -----------
        Updates the global model by aggregating the provided gradients and performing
        an optimization step to adjust model parameters.

        Parameters
        ----------
        gradients : list
            A list of gradients to aggregate and apply to the global model.
        """
        aggregate_gradient = self.aggregate(gradients)
        self.set_gradients(aggregate_gradient)
        self._step()
    
    def update_model_with_weights(self, weights):
        """
        Description
        -----------
        Updates the global model by aggregating the provided weights

        Parameters
        ----------
        weights : list
            A list of weights to aggregate and apply to the global model.
        """
        aggregate_weights = self.aggregate(weights)
        self.set_parameters(aggregate_weights)

    def update_model_with_dict_gradients(self, gradients):
        """
        Description
        -----------
        Updates the global model by aggregating the provided gradients and performing
        an optimization step to adjust model parameters.

        Parameters
        ----------
        gradients : list
            A list of gradients given as dicts to aggregate and apply to the global model.
        
        Notes
        -----
        This function assumes that the aggregator takes a list of dicts as an argument
        and returns an aggregated gradient as a dict.
        """
        aggregate_gradient = self.aggregate(gradients)
        self.set_gradients(flatten_dict(aggregate_gradient))
        self._step()

    def _step(self):
        """
        Description
        -----------
        Executes a single optimization step for the global model. The optimizer
        updates model parameters based on aggregated gradients, and the scheduler
        adjusts the learning rate as required.
        """
        self.optimizer.step()
        self.scheduler.step()

    def _compute_accuracy(self, data_loader):
        """
        Description
        -----------
        Computes the accuracy of the global model on a given dataset.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            A DataLoader object for the dataset to evaluate the model on.

        Returns
        -------
        float
            The accuracy of the model on the provided dataset, as a value
            between 0 and 1.
        """
        total = 0
        correct = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        return correct / total
    
    def compute_test_accuracy_per_class(self, nb_classes):
        #should return an np.array of shape (nb_classes,) with the accuracy for each class
        total = np.zeros(nb_classes)
        correct = np.zeros(nb_classes) 
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(nb_classes):
                total[i] += (targets == i).sum().item()
                correct[i] += ((predicted == targets) & (targets == i)).sum().item()
        return correct / total
        
    def compute_test_loss_per_class(self, num_classes):
        #should return an np.array of shape (nb_classes,) with the loss for each class
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        total_loss = np.zeros(num_classes)
        total_samples = np.zeros(num_classes)
        
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            losses = criterion(outputs, targets)
            for i in range(num_classes):
                class_mask = (targets == i)
                total_loss[i] += losses[class_mask].sum().item()
                total_samples[i] += class_mask.sum().item()
        
        average_loss_per_class = total_loss / total_samples
        return average_loss_per_class

    def compute_validation_accuracy(self):
        """
        Description
        -----------
        Computes the accuracy of the global model on the validation dataset.
        This method evaluates the model's performance during training.

        Returns
        -------
        float
            Validation accuracy as a value between 0 and 1.

        Raises
        ------
        RuntimeError
            If the validation DataLoader is not set.
        """
        if self.validation_loader is None:
            logger.critical("Validation Data Loader is not set.")
            return
        return self._compute_accuracy(self.validation_loader)

    def compute_test_accuracy(self):
        """
        Description
        -----------
        Computes the accuracy of the global model on the test dataset.
        This method is used to evaluate the model's final performance.

        Returns
        -------
        float
            Test accuracy as a value between 0 and 1.
        """
        return self._compute_accuracy(self.test_loader)