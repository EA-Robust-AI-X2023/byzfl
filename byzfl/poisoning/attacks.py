
from copy import deepcopy


class StaticLabelFlipping(object): 
    """
    Description
    -----------

    Flip target labels with a fixed permutation.

    Initialization parameters
    --------------------------

    permutation: dict
        A dictionary that maps a label to its flipped version.

    Calling the instance
    --------------------

    Input parameters
    ----------------

    model : torch.nn.Module
        The local model.
    inputs : numpy.ndarray or torch.Tensor
        A collection of input features.
    targets : numpy.ndarray or torch.Tensor
        A collection of targets.

    Returns
    -------
    tuple
        The inputs and targets with flipped labels.
    """
    def __init__(self, permutation):
        if not isinstance(permutation, dict):
            raise TypeError("permutation must be a dict.")
        self.permutation = permutation
    
    def __call__(self, model, inputs, targets):
        for i, target in enumerate(targets.numpy()):
            targets[i] = self.permutation[target]
        
        return inputs, targets