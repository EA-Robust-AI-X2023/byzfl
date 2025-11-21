import time

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from byzfl import Client, Server, DataDistributor, PoisoningClient
from byzfl.utils.misc import max_distance_to_gradient, set_random_seed
from byzfl.utils.conversion import unflatten_dict
from byzfl.benchmark.managers import ParamsManager, FileManager
from byzfl.benchmark.evaluate_results import plot_worker_class_distribution, compute_exclusivity

transforms_hflip = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
transforms_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
transforms_cifar_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transforms_cifar_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#Supported datasets
dict_datasets = {
    "mnist":        ("MNIST", transforms_mnist, transforms_mnist),
    "fashionmnist": ("FashionMNIST", transforms_hflip, transforms_hflip),
    "emnist":       ("EMNIST", transforms_mnist, transforms_mnist),
    "cifar10":      ("CIFAR10", transforms_cifar_train, transforms_cifar_test),
    "cifar100":     ("CIFAR100", transforms_cifar_train, transforms_cifar_test),
    "imagenet":     ("ImageNet", transforms_hflip, transforms_hflip)
}


def start_training(params):
    params_manager = ParamsManager(params)

    # <----------------- File Manager  ----------------->
    file_manager = FileManager({
        "result_path": params_manager.get_results_directory(),
        "dataset_name": params_manager.get_dataset_name(),
        "model_name": params_manager.get_model_name(),
        "nb_workers": params_manager.get_nb_workers(),
        "nb_byz": params_manager.get_f(),
        "declared_nb_byz": params_manager.get_tolerated_f(),
        "data_distribution_name": params_manager.get_name_data_distribution(),
        "distribution_parameter": (
            None if params_manager.get_name_data_distribution() 
            in ["iid", "extreme_niid", "extreme_niid_modified"] 
            else params_manager.get_parameter_data_distribution()
        ),
        "aggregation_name": params_manager.get_aggregator_name(),
        "pre_aggregation_names": [
            dict['name'] 
            for dict in params_manager.get_preaggregators()
        ],
        "attack_name": params_manager.get_attack_name(),
        "learning_rate": params_manager.get_learning_rate(),
        "momentum": params_manager.get_honest_clients_momentum(),
        "weight_decay": params_manager.get_honest_clients_weight_decay(),
    })

    file_manager.save_config_dict(params_manager.get_data())

    # <----------------- Federated Framework ----------------->

    # Configurations
    nb_honest_clients = params_manager.get_nb_honest_clients()
    nb_byz_clients = params_manager.get_f()
    nb_training_steps = params_manager.get_nb_steps()
    batch_size = params_manager.get_honest_clients_batch_size()

    dd_seed = params_manager.get_data_distribution_seed()
    training_seed = params_manager.get_training_seed()
    set_random_seed(dd_seed)

    # Data Preparation
    key_dataset_name = params_manager.get_dataset_name()
    dataset_name = dict_datasets[key_dataset_name][0]
    dataset = getattr(datasets, dataset_name)(
            root = params_manager.get_data_folder(), 
            train = True, 
            download = True,
            transform = None
    )
    dataset.targets = Tensor(dataset.targets).long()

    train_size = int(params_manager.get_size_train_set() * len(dataset))
    val_size = len(dataset) - train_size

    # Split Train set into Train and Validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Apply transformations to each dataset
    train_dataset.dataset.transform = dict_datasets[key_dataset_name][1]
    val_dataset.dataset.transform = dict_datasets[key_dataset_name][2]

    # Prepare Validation and Test data
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=params_manager.get_batch_size_evaluation(), 
            shuffle=False
        )
    else:
        val_loader = None
    
    test_dataset = getattr(datasets, dataset_name)(
                root = params_manager.get_data_folder(),
                train=False, 
                download=True,
                transform=dict_datasets[key_dataset_name][2]
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=params_manager.get_batch_size_evaluation(), 
        shuffle=False
    )

    # Distribute data among clients using non-IID Dirichlet distribution
    data_distributor = DataDistributor({
        "data_distribution_name": params_manager.get_name_data_distribution(),
        "distribution_parameter": params_manager.get_parameter_data_distribution(),
        "nb_workers": nb_honest_clients + nb_byz_clients,
        "data_loader": train_dataset,
        "batch_size": batch_size,
        "min_size": params_manager.get_min_size_data_distribution(),
    })
    client_dataloaders = data_distributor.split_data()

    # Initialize Honest Clients
    honest_clients = [
        Client({
            "model_name": params_manager.get_model_name(),
            "device": params_manager.get_device(),
            "optimizer_name": params_manager.get_optimizer_name(),
            "learning_rate": params_manager.get_learning_rate(),
            "loss_name": params_manager.get_loss_name(),
            "weight_decay": params_manager.get_honest_clients_weight_decay(),
            "milestones": params_manager.get_milestones(),
            "learning_rate_decay": params_manager.get_learning_rate_decay(),
            "LabelFlipping": "LabelFlipping" == params_manager.get_attack_name(),
            "training_dataloader": client_dataloaders[i],
            "momentum": params_manager.get_honest_clients_momentum(),
            "nb_labels": params_manager.get_nb_labels(),
            "store_per_client_metrics": params_manager.get_store_per_client_metrics(),
        }) for i in range(nb_honest_clients)
    ]

    # Server Setup, Use SGD Optimizer
    server = Server({
        "model_name": params_manager.get_model_name(),
        "device": params_manager.get_device(),
        "validation_loader": val_loader,
        "test_loader": test_loader,
        "optimizer_name": params_manager.get_optimizer_name(),
        "learning_rate": params_manager.get_learning_rate(),
        "weight_decay": params_manager.get_honest_clients_weight_decay(),
        "milestones": params_manager.get_milestones(),
        "learning_rate_decay": params_manager.get_learning_rate_decay(),
        "aggregator_info": params_manager.get_aggregator_info(),
        "pre_agg_list": params_manager.get_preaggregators(),
    })

    # Byzantine Client Setup
    attack_parameters={}
    attack_parameters["parameters"] = params_manager.get_attack_parameters()
    attack_parameters["p"] = 1.0 if "p" not in params_manager.get_attack_info() else params_manager.get_attack_info()["p"] #architecture de p à revoir
    attack_parameters["name"] = params_manager.get_attack_name()

    poisoned_clients = [
        PoisoningClient({"model_name": params_manager.get_model_name(),
            "device": params_manager.get_device(),
            "optimizer_name": params_manager.get_optimizer_name(),
            "learning_rate": params_manager.get_learning_rate(),
            "loss_name": params_manager.get_loss_name(),
            "weight_decay": params_manager.get_honest_clients_weight_decay(),
            "milestones": params_manager.get_milestones(),
            "learning_rate_decay": params_manager.get_learning_rate_decay(),
            "LabelFlipping": "LabelFlipping" == params_manager.get_attack_name(),
            "training_dataloader": client_dataloaders[i + nb_honest_clients],
            "momentum": params_manager.get_honest_clients_momentum(),
            "nb_labels": params_manager.get_nb_labels(),
            "store_per_client_metrics": params_manager.get_store_per_client_metrics()},
            attack_parameters,
        ) for i in range(nb_byz_clients)
    ]
    clients = honest_clients+poisoned_clients
    
    if params_manager.get_save_worker_distributions():
        path_plot_distribution=file_manager.make_distribution_dir()
        partitions = plot_worker_class_distribution(honest_clients+poisoned_clients, path_plot_distribution,params_manager.get_nb_labels(), params_manager.get_name_data_distribution(), dd_seed)
        
        #save the per class per worker partition data
        partition_file_name=f"distributions/worker_distributions_dd_seed_{dd_seed}.txt"
        file_manager.write_matrix_in_file(partitions,partition_file_name)
        
        #save the main exclusivity measure:
        exlcusivity_file_name=f"distributions/exclusivity_dd_seed_{dd_seed}.txt"
        exclusivity_measures = compute_exclusivity(partitions)
        file_manager.write_matrix_in_file(exclusivity_measures,exlcusivity_file_name)

        



    set_random_seed(training_seed)

    evaluation_delta = params_manager.get_evaluation_delta()
    evaluate_on_test = params_manager.get_evaluate_on_test()

    make_feature_measures = params_manager.get_make_feature_measures()
    compute_gradient_variance = params_manager.get_compute_gradient_variance()
    compute_gradient_scatterings = params_manager.get_compute_gradient_scatterings()
    scatter_momentums = params_manager.get_scatter_momentums()

    store_models = params_manager.get_store_models()
    store_per_client_metrics = params_manager.get_store_per_client_metrics()

    val_accuracy_list = np.array([])
    test_accuracy_list = np.array([])
    train_loss_list = np.zeros((nb_training_steps))

    honest_scattering_list = np.array([])
    poisoned_scattering_list = np.array([])
    feature_mean = np.array([])
    feature_variance_dict = {i:np.zeros((nb_training_steps//evaluation_delta +1)) for i in range(nb_honest_clients + nb_byz_clients)}
    gradient_variance = np.array([])

    start_time = time.time()

    # Send Initial Model to All Clients
    new_model = server.get_dict_parameters()
    for client in honest_clients:
        client.set_model_state(new_model)
    
    training_algorithm_name = params_manager.get_training_algorithm_name()

    if training_algorithm_name not in ["DSGD"]:
        raise ValueError(f"Training algorithm {training_algorithm_name} not supported, supported algorithms are 'DSGD' and 'FedAvg'")

    # Training Loop
    for training_step in range(nb_training_steps):
        
        #default, no special measures
        compute_gradient_variance_step= False
        compute_gradient_scatterings_step= False
        make_feature_measures_step= False

        # Evaluate Global Model Every Evaluation Delta Steps
        if training_step % evaluation_delta == 0:

            if val_loader is not None:

                val_acc = server.compute_validation_accuracy()

                val_accuracy_list = np.append(val_accuracy_list, val_acc)

                file_manager.write_array_in_file(
                    val_accuracy_list, 
                    "val_accuracy_tr_seed_" + str(training_seed) 
                    + "_dd_seed_" + str(dd_seed) +".txt"
                )

            if evaluate_on_test:
                test_acc = server.compute_test_accuracy()
                test_accuracy_list = np.append(test_accuracy_list, test_acc)

                file_manager.write_array_in_file(
                    test_accuracy_list, 
                    "test_accuracy_tr_seed_" + str(training_seed) 
                    + "_dd_seed_" + str(dd_seed) +".txt"
                )

            if store_models:
                file_manager.save_state_dict(
                    server.get_dict_parameters(),
                    training_seed,
                    dd_seed,
                    training_step
                )
                
            if compute_gradient_variance:
                compute_gradient_variance_step=True
            if compute_gradient_scatterings:
                compute_gradient_scatterings_step=True
            if make_feature_measures:
                make_feature_measures_step=True
        
        if training_algorithm_name == "DSGD" and params_manager.get_aggregator_name() != "Lfighter":

            train_loss_per_client = np.zeros((nb_honest_clients+ nb_byz_clients))
            mean_feature = np.zeros((nb_honest_clients+ nb_byz_clients))
            gradient_variances = np.zeros((nb_honest_clients + nb_byz_clients))
            feature_variance= np.zeros((nb_honest_clients + nb_byz_clients))


            # Honest Clients Compute Gradients
            for i, client in enumerate(honest_clients):
                (train_loss_per_client[i], 
                mean_feature[i], 
                feature_variance[i], 
                gradient_variances[i]) = client.compute_gradients(make_feature_measures=make_feature_measures, 
                                                                  compute_variance=compute_gradient_variance_step)
            
            train_loss_list[training_step] = train_loss_per_client.mean()
            
            # Aggregate Honest Gradients
            honest_gradients_with_momentum = [client.get_flat_gradients_with_momentum() for client in honest_clients]
            
            # Apply poisoning attack
            for i, poisoned_client in enumerate(poisoned_clients):
                (train_loss_per_client[i + nb_honest_clients], 
                 mean_feature[i + nb_honest_clients], 
                 feature_variance[i + nb_honest_clients], 
                 gradient_variances[i + nb_honest_clients]) = poisoned_client.compute_gradients(make_feature_measures=make_feature_measures, compute_variance=compute_gradient_variance_step)

            poisoned_gradients_with_momentum = [client.get_flat_gradients_with_momentum() for client in poisoned_clients]
            
            # Combine Honest and poisoned Gradients
            gradients_with_momentum = honest_gradients_with_momentum + poisoned_gradients_with_momentum

            # Update Global Model
            server.update_model_with_gradients(gradients_with_momentum)
            
            if compute_gradient_scatterings_step:
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
                honest_scattering_list=np.append(honest_scattering_list, max_dist_gradient_honest)

                # Evaluate poisoned gradients scatterings
                max_dist_gradient_poisoned= max_distance_to_gradient(poisoned_gradients_for_scattering, gradient)
                if isinstance(max_dist_gradient_poisoned, torch.Tensor):
                    max_dist_gradient_poisoned = max_dist_gradient_poisoned.detach().cpu().item()
                poisoned_scattering_list=np.append(poisoned_scattering_list,max_dist_gradient_poisoned)

            if make_feature_measures_step:
                # Save features norm mean
                feature_mean=np.append(feature_mean, mean_feature.max())

            if compute_gradient_variance_step:
                gradient_variance=np.append(gradient_variance, gradient_variances.max())


        elif params_manager.get_aggregator_name() == "Lfighter":

            train_loss_per_client = np.zeros((nb_honest_clients+ nb_byz_clients))
            mean_feature = np.zeros((nb_honest_clients+ nb_byz_clients))
            gradient_variances = np.zeros((nb_honest_clients + nb_byz_clients))
            feature_variance=np.zeros((nb_honest_clients + nb_byz_clients))

            # Honest Clients Compute Gradients
            for i, client in enumerate(honest_clients):
                (train_loss_per_client[i], 
                mean_feature[i], 
                feature_variance[i], 
                gradient_variances[i]) = client.compute_gradients_and_update(make_feature_measures=make_feature_measures,compute_variance=compute_gradient_variance_step)
                            
            train_loss_list[training_step] = train_loss_per_client.mean()
            
            # Aggregate Honest Gradients
            honest_gradients = [client.get_flat_gradients_with_momentum() for client in honest_clients]
            
            # Apply poisoning attack
            for i, poisoned_client in enumerate(poisoned_clients):
                (train_loss_per_client[i + nb_honest_clients], 
                 mean_feature[i + nb_honest_clients],
                feature_variance[i + nb_honest_clients], 
                gradient_variances[i + nb_honest_clients]) = poisoned_client.compute_gradients_and_update(make_feature_measures=make_feature_measures, compute_variance=compute_gradient_variance_step)                
            
            poisoned_gradients = [client.get_flat_gradients_with_momentum() for client in poisoned_clients]

            if compute_gradient_scatterings_step:
                
                #honest and byzantine gradients for scattering computations
                honest_gradients_for_scattering = honest_gradients
                poisoned_gradients_for_scattering = poisoned_gradients
                
                if scatter_momentums: #default is false
                    honest_gradients_for_scattering = [client.get_flat_gradients() for client in honest_clients]
                    poisoned_gradients_for_scattering = [client.get_flat_gradients() for client in poisoned_clients]
                    
                # Compute the average honest gradient
                gradient = torch.stack(honest_gradients).mean(dim = 0)

                # Evaluate honest gradients scatterings
                max_dist_gradient_honest_cpu=max_distance_to_gradient(honest_gradients_for_scattering, gradient).cpu().item()
                
                honest_scattering_list=np.append(honest_scattering_list,max_dist_gradient_honest_cpu)

                # Evaluate byzantine gradients scatterings
                max_dist_gradient_poisoned_cpu=max_distance_to_gradient(poisoned_gradients_for_scattering, gradient).cpu().item()
                poisoned_scattering_list=np.append(poisoned_scattering_list,max_dist_gradient_poisoned_cpu)

            if compute_gradient_variance_step:
                gradient_variance=np.append(gradient_variance, gradient_variances.max())

            if make_feature_measures_step:
                # Save features norm mean
                feature_mean=np.append(feature_mean, mean_feature.max())
                for i in range(nb_honest_clients + nb_byz_clients):
                    feature_variance_dict[i] = feature_variance[i]

            clients = honest_clients + poisoned_clients
            # Get the local gradients without momentum for LFighter
            raw_gradients = [client.get_dict_gradients() for client in clients]

            # Identify the malicious gradients
            lfighter = server.robust_aggregator.aggregator
            scores = lfighter.get_scores(raw_gradients)
            gradients_with_momentum = honest_gradients+poisoned_gradients
            
            # Aggregate the gradients with momentum
            aggregate_gradient = lfighter.average_gradients(gradients_with_momentum, scores)
            server.set_gradients(aggregate_gradient)
            server._step()

        else:
            raise ValueError(f"Training algorithm {training_algorithm_name} not supported")
        
        # Send Updated Model to Clients
        new_model = server.get_dict_parameters()
        for client in clients:
            client.set_model_state(new_model)
    
    end_time = time.time()

    file_manager.write_array_in_file(
        train_loss_list, 
        "train_loss_tr_seed_" + str(training_seed) 
        + "_dd_seed_" + str(dd_seed) +".txt"
    )

    if val_loader is not None:
    
        val_acc = server.compute_validation_accuracy()

        val_accuracy_list = np.append(val_accuracy_list, val_acc)

        file_manager.write_array_in_file(
            val_accuracy_list, 
            "val_accuracy_tr_seed_" + str(training_seed) 
            + "_dd_seed_" + str(dd_seed) +".txt"
        )

    if evaluate_on_test:
        test_acc = server.compute_test_accuracy()
        test_accuracy_list = np.append(test_accuracy_list, test_acc)

        file_manager.write_array_in_file(
            test_accuracy_list, 
            "test_accuracy_tr_seed_" + str(training_seed) 
            + "_dd_seed_" + str(dd_seed) +".txt"
        )

    if store_per_client_metrics:

        for client_id, client in enumerate(honest_clients):
            loss = client.get_loss_list()
            acc = client.get_train_accuracy()
            
            file_manager.save_loss(
                loss,
                training_seed,
                dd_seed,
                client_id
            )
            
            file_manager.save_accuracy(
                acc,
                training_seed,
                dd_seed,
                client_id
            )

            file_manager.save_feature_variance(
                feature_variance=feature_variance_dict[client_id],
                training_seed=training_seed,
                data_dist_seed=dd_seed,
                client_id=client_id
            )
        
        for client_id, client in enumerate(poisoned_clients):
            
            file_manager.save_feature_variance(
                feature_variance=feature_variance_dict[client_id + nb_honest_clients],
                training_seed=training_seed,
                data_dist_seed=dd_seed,
                client_id=client_id + nb_honest_clients
            )
    
    if store_models:
        file_manager.save_state_dict(
            server.get_dict_parameters(),
            training_seed,
            dd_seed,
            training_step
        )

    file_manager.save_honest_scattering(
        honest_scattering_list=honest_scattering_list,
        training_seed=training_seed,
        data_dist_seed=dd_seed
    )

    file_manager.save_poisoned_scattering(
        poisoned_scattering_list=poisoned_scattering_list,
        training_seed=training_seed,
        data_dist_seed=dd_seed
    )

    file_manager.save_mean_feature_norm(
        feature_mean=feature_mean,
        training_seed=training_seed,
        data_dist_seed=dd_seed
    )

    file_manager.save_gradients_variance(
        gradients_variance=gradient_variance,
        training_seed=training_seed,
        data_dist_seed=dd_seed
    )
    
    execution_time = end_time - start_time

    file_manager.write_array_in_file(
        np.array(execution_time),
        "train_time_tr_seed_" + str(training_seed) 
        + "_dd_seed_" + str(dd_seed) +".txt"
    )