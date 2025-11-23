from byzfl.utils.misc import max_distance_to_gradient
import numpy as np
import torch


def compute_scatterings(honest_clients, poisoned_clients, honest_gradients_with_momentum, poisoned_gradients_with_momentum, scatter_momentums):
    #we are interested in the scatterings of honest and byzantine gradients whithout the momentum term
    if scatter_momentums: #default is false
        honest_gradients_for_scattering = honest_gradients_with_momentum
        poisoned_gradients_for_scattering = poisoned_gradients_with_momentum
        gradient = torch.stack(honest_gradients_with_momentum).mean(dim = 0)
    else:
        honest_gradients_for_scattering = [client.get_flat_gradients() for client in honest_clients]
        poisoned_gradients_for_scattering = [client.get_flat_gradients() for client in poisoned_clients]
        gradient = torch.stack(honest_gradients_for_scattering).mean(dim = 0)

    # Evaluate honest gradients scatterings
    max_dist_gradient_honest= max_distance_to_gradient(honest_gradients_for_scattering, gradient)
    if isinstance(max_dist_gradient_honest, torch.Tensor):
        max_dist_gradient_honest = max_dist_gradient_honest.detach().cpu().item()

    # Evaluate poisoned gradients scatterings
    max_dist_gradient_poisoned= max_distance_to_gradient(poisoned_gradients_for_scattering, gradient)
    if isinstance(max_dist_gradient_poisoned, torch.Tensor):
        max_dist_gradient_poisoned = max_dist_gradient_poisoned.detach().cpu().item()
    
    return max_dist_gradient_honest, max_dist_gradient_poisoned