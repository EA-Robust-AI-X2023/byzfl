
from copy import deepcopy
import torch as torch
import numpy as np


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
    def __init__(self, permutation = {0:9, 1:8, 2:7, 3:6, 4:5, 5:4, 6:3, 7:2, 8:1, 9:0}):
        if not isinstance(permutation, dict):
            raise TypeError("permutation must be a dict.")
        #convert all values in permutation to integers. Not optimal but best to avoid bugs
        permutation_int={}
        for i,j in permutation.items():
            permutation_int[int(i)]=int(j)
        self.permutation=permutation_int
    
    def __call__(self, model, inputs, targets):
        # Support both torch.Tensor and numpy.ndarray targets
        if torch.is_tensor(targets):
            # convert to cpu numpy, apply permutation, then convert back to same device/dtype
            device = targets.device
            dtype = targets.dtype
            targets_np = targets.detach().cpu().numpy()
            poisoned_np = np.array([self.permutation[int(t)] for t in targets_np], dtype=np.int64)
            poisoned_targets = torch.from_numpy(poisoned_np).to(device=device, dtype=dtype)
            return inputs, poisoned_targets
        else:
            # assume numpy array-like
            poisoned = np.array([self.permutation[int(t)] for t in targets], dtype=np.int64)
            return inputs, poisoned
    
    
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
    
    def __init__(self, print_flips=False):
        self.print_flips = print_flips
    
    def __call__(self, model, inputs, targets):
        flipped_targets = model(inputs).argmin(dim=1)

        if self.print_flips:
            print(f"Flipping targets:")
            for i, (original, flipped) in enumerate(zip(targets, flipped_targets)):
                if original != flipped:
                    print(f"{original} -> {flipped} /")
            print("\n")

        return inputs, flipped_targets
    
    
    class NoAttack(object):
        """
    Description
    -----------

    No attack is applied, and the original inputs and targets are returned.
    """
    def __call__(self, model, inputs, targets):
        return inputs, targets
  
