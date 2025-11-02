
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
        #convert all values in permutation to integers. Not optimal but best to avoid bugs
        permutation_int={}
        for i,j in permutation.items():
            permutation_int[int(i)]=int(j)
        self.permutation=permutation_int
    
    def __call__(self, model, inputs, targets):
        for i, target in enumerate(targets.numpy()):
            targets[i] = self.permutation[target]
        
        return inputs, targets
    
    
class DynamicLabelFlipping(object):
    """
    Description
    -----------

    Flip target labels towards the least likely label (according to the model and current parameters)


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
    
    def __call__(self, model, inputs, targets):
        flipped_targets = model(inputs).argmin(dim=1).cpu().numpy()
    
        return inputs, flipped_targets
    

  
