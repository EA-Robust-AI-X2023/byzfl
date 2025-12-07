import math
import json
import os
import matplotlib.pyplot as plt
from collections import Counter

from loguru import logger
import numpy as np
from numpy import genfromtxt
from byzfl.benchmark.managers import FileManager
import seaborn as sns


def custom_dict_to_str(dictionary):
    """
    Safely convert a dictionary to a string.
    Returns an empty string if the dictionary is empty.
    """
    return '' if not dictionary else str(dictionary)


def ensure_list(value):
    """
    Ensure the given value is returned as a list.
    If it is not, wrap it in a list.
    """
    if not isinstance(value, list):
        value = [value]
    return value


def find_best_hyperparameters(path_to_results):
    """
    Find the best hyperparameters (learning rate, momentum, weight decay) 
    that maximize the minimum accuracy across different attacks.

    Reads a configuration file (config.json) in `path_to_results` 
    and writes out the best hyperparameters and the corresponding 
    step at which maximum accuracy was reached for each aggregator 
    and each attack.
    """
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        logger.error(f"Failed reading config.json: {e}")
        return
    
    path_hyperparameters = path_to_results + "/best_hyperparameters"

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = data["model"]["learning_rate"]

    # <-------------- Honest Nodes Config ------------->
    momentum_list = data["honest_clients"]["momentum"]
    wd_list = data["honest_clients"]["weight_decay"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    # Number of accuracy checkpoints
    nb_accuracies = 1 + math.ceil(nb_steps / evaluation_delta)

    # Main nested loops to explore configurations
    for nb_honest in nb_honest_clients:
        for nb_byzantine in nb_byz:
            
            if nb_declared[0] is None:
                nb_declared_list = [nb_byzantine]
            else:
                nb_declared_list = nb_declared.copy()
                nb_declared_list = [item for item in nb_declared_list if item >= nb_byzantine]

            for nb_decl in nb_declared_list:
                if set_honest_clients_as_clients:
                    nb_nodes = nb_honest
                else:
                    nb_nodes = nb_honest + nb_byzantine

                for data_dist in data_distributions:
                    distribution_parameter_list = ensure_list(data_dist["distribution_parameter"])
                    for distribution_parameter in distribution_parameter_list:
                        for pre_agg in pre_aggregators:
                            # Build a single name from all pre-aggregators
                            pre_agg_names_list = [p["name"] for p in pre_agg]
                            pre_agg_names = "_".join(pre_agg_names_list)

                            # Prepare arrays to store final best hyperparams & steps
                            real_hyper_parameters = np.zeros((len(aggregators), 3))
                            real_steps = np.zeros((len(aggregators), len(attacks)))

                            for k, agg in enumerate(aggregators):
                                # We'll store max accuracy for each (lr, momentum, wd) across attacks
                                num_combinations = len(lr_list) * len(momentum_list) * len(wd_list)
                                max_acc_config = np.zeros((num_combinations, len(attacks)))
                                hyper_parameters = np.zeros((num_combinations, 3))
                                steps_max_reached = np.zeros((num_combinations, len(attacks)))

                                index_combination = 0
                                for lr in lr_list:
                                    for momentum in momentum_list:
                                        for wd in wd_list:
                                            # tab_acc shape: (len(attacks), nb_dd_seeds, nb_training_seeds, nb_accuracies)
                                            tab_acc = np.zeros(
                                                (
                                                    len(attacks),
                                                    nb_data_distribution_seeds,
                                                    nb_training_seeds,
                                                    nb_accuracies
                                                )
                                            )

                                            # Fill tab_acc with loaded accuracy files
                                            for i, attack in enumerate(attacks):
                                                for run_dd in range(nb_data_distribution_seeds):
                                                    for run in range(nb_training_seeds):
                                                        file_name = (
                                                            f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                            f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                            f"{distribution_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                            f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                            f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                                        )
                                                        acc_path = os.path.join(
                                                            path_to_results,
                                                            file_name,
                                                            f"val_accuracy_tr_seed_{run + training_seed}"
                                                            f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                                        )
                                                        tab_acc[i, run_dd, run] = genfromtxt(acc_path, delimiter=',')

                                            tab_acc = tab_acc.reshape(
                                                len(attacks),
                                                nb_data_distribution_seeds * nb_training_seeds,
                                                nb_accuracies
                                            )
                                            
                                            # Compute average accuracy across seeds, find max
                                            for i in range(len(attacks)):
                                                avg_accuracy = np.mean(tab_acc[i], axis=0)
                                                idx_max = np.argmax(avg_accuracy)
                                                max_acc_config[index_combination, i] = avg_accuracy[idx_max]
                                                steps_max_reached[index_combination, i] = idx_max * evaluation_delta

                                            hyper_parameters[index_combination] = [lr, momentum, wd]
                                            index_combination += 1

                                # Create path if needed
                                if not os.path.exists(path_hyperparameters):
                                    try:
                                        os.makedirs(path_hyperparameters)
                                    except OSError as error:
                                        logger.error(f"Failed creating directory: {error}")

                                # Find the combination that maximizes the minimum accuracy across attacks
                                max_minimum_idx = -1
                                max_minimum_val = -1
                                for i in range(num_combinations):
                                    current_min = np.min(max_acc_config[i])
                                    if current_min > max_minimum_val:
                                        max_minimum_idx = i
                                        max_minimum_val = current_min

                                real_hyper_parameters[k] = hyper_parameters[max_minimum_idx]
                                real_steps[k] = steps_max_reached[max_minimum_idx]

                            # Save results to folder
                            hyper_parameters_folder = os.path.join(path_hyperparameters, "hyperparameters")
                            steps_folder = os.path.join(path_hyperparameters, "better_step")

                            os.makedirs(hyper_parameters_folder, exist_ok=True)
                            os.makedirs(steps_folder, exist_ok=True)

                            for i, agg in enumerate(aggregators):
                                # Save best hyperparameters
                                file_name_hparams = (
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                                    f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                    f"{distribution_parameter}_{pre_agg_names}_{agg['name']}.txt"
                                )
                                np.savetxt(
                                    os.path.join(hyper_parameters_folder, file_name_hparams),
                                    real_hyper_parameters[i]
                                )

                                # Save step at which max accuracy occurs for each attack
                                for j, attack in enumerate(attacks):
                                    file_name_steps = (
                                        f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                                        f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                        f"{distribution_parameter}_{pre_agg_names}_{agg['name']}_"
                                        f"{custom_dict_to_str(attack['name'])}.txt"
                                    )
                                    step_val = np.array([real_steps[i, j]])
                                    np.savetxt(os.path.join(steps_folder, file_name_steps), step_val)

colors = [
    (0.000, 0.447, 0.741),   # blue
    (0.850, 0.325, 0.098),   # red-orange
    (0.466, 0.674, 0.188),   # green
    (0.494, 0.184, 0.556),   # purple
    (0.929, 0.694, 0.125),   # yellow
    (0.301, 0.745, 0.933),   # cyan
    (0.635, 0.078, 0.184),   # dark red
    (0.7,   0.2,   0.5),     # magenta blend
    (0.2,   0.2,   0.2),     # dark gray
    (0.7,   0.7,   0.7)      # light gray
]
tab_sign = [
    '-',
    '--',
    '-.',
    ':',
    (0, (5, 1)),     # dashed, fine
    (0, (3, 1, 1, 1)),  # dash-dot-dot
    (0, (1, 1)),     # densely dotted
    (0, (3, 5, 1, 5)),  # dash-space patterns
    (0, (5, 10)),    # very sparse dash
    'solid'
]
markers = [
    'o',   # circle
    's',   # square
    '^',   # triangle up
    'v',   # triangle down
    '<',   # triangle left
    '>',   # triangle right
    'D',   # diamond
    'P',   # plus-filled
    'X',   # x-filled
    '*'    # star
]

def test_accuracy_curve(path_to_results, path_to_plot, colors=colors, tab_sign=tab_sign, markers=markers):
        
        try:
            with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
                data = json.load(file)
        except Exception as e:
            logger.error(f"Failed reading config.json: {e}")
            return
        
        try:
            os.makedirs(path_to_plot, exist_ok=True)
        except OSError as error:
            logger.error(f"Failed creating directory: {error}")
        
        path_to_hyperparameters = path_to_results + "/best_hyperparameters"
        

        # <-------------- Benchmark Config ------------->
        training_seed = data["benchmark_config"]["training_seed"]
        nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
        nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
        nb_byz = data["benchmark_config"]["f"]
        nb_declared = data["benchmark_config"].get("tolerated_f", None)
        data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
        nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
        data_distributions = data["benchmark_config"]["data_distribution"]
        set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
        nb_steps = data["benchmark_config"]["nb_steps"]


        # <-------------- Evaluation and Results ------------->
        evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

        # <-------------- Model Config ------------->
        model_name = data["model"]["name"]
        dataset_name = data["model"]["dataset_name"]
        lr_list = data["model"]["learning_rate"]

        # <-------------- Honest Nodes Config ------------->
        momentum_list = data["honest_clients"]["momentum"]
        wd_list = data["honest_clients"]["weight_decay"]

        # <-------------- Aggregators Config ------------->
        aggregators = data["aggregator"]
        pre_aggregators = data["pre_aggregators"]

        # <-------------- Attacks Config ------------->
        attacks = data["attack"]

        # Ensure certain configurations are always lists
        nb_honest_clients = ensure_list(nb_honest_clients)
        nb_byz = ensure_list(nb_byz)
        nb_declared = ensure_list(nb_declared)
        data_distributions = ensure_list(data_distributions)
        aggregators = ensure_list(aggregators)

        # Pre-aggregators can be multiple or single dict; unify them
        if not pre_aggregators or isinstance(pre_aggregators[0], dict):
            pre_aggregators = [pre_aggregators]

        attacks = ensure_list(attacks)
        lr_list = ensure_list(lr_list)
        momentum_list = ensure_list(momentum_list)
        wd_list = ensure_list(wd_list)

        nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))

        for nb_honest in nb_honest_clients:
            for nb_byzantine in nb_byz:

                if nb_declared[0] is None:
                    nb_declared_list = [nb_byzantine]
                else:
                    nb_declared_list = nb_declared.copy()
                    nb_declared_list = [item for item in nb_declared_list if item >= nb_byzantine]
                
                for nb_decl in nb_declared_list:

                    if set_honest_clients_as_clients:
                        nb_nodes = nb_honest
                    else:
                        nb_nodes = nb_honest + nb_byzantine
                    
                    for data_dist in data_distributions:
                        dist_parameter_list = data_dist["distribution_parameter"]
                        dist_parameter_list = ensure_list(dist_parameter_list)
                        for dist_parameter in dist_parameter_list:
                            for pre_agg in pre_aggregators:
                                pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                                pre_agg_names = "_".join(pre_agg_list_names)
                                for agg in aggregators:

                                    hyper_file_name = (
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_{pre_agg_names}_{agg['name']}.txt"
                                    )


                                    full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                    if os.path.exists(full_path):
                                        hyperparameters = np.loadtxt(full_path)
                                        lr = hyperparameters[0]
                                        momentum = hyperparameters[1]
                                        wd = hyperparameters[2]
                                    else:
                                        lr = lr_list[0]
                                        momentum = momentum_list[0]
                                        wd = wd_list[0]

                                    tab_acc = np.zeros((
                                        len(attacks), 
                                        nb_data_distribution_seeds,
                                        nb_training_seeds,
                                        nb_accuracies
                                    ))

                                    for i, attack in enumerate(attacks):
                                        for run_dd in range(nb_data_distribution_seeds):
                                            for run in range(nb_training_seeds):
                                                file_name = (
                                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                    f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                    f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                    f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                    f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                                )
                                                acc_path = os.path.join(
                                                    path_to_results,
                                                    file_name,
                                                    f"test_accuracy_tr_seed_{run + training_seed}"
                                                    f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                                )
                                                tab_acc[i, run_dd, run] = genfromtxt(acc_path, delimiter=',')

                                    tab_acc = tab_acc.reshape(
                                        len(attacks),
                                        nb_data_distribution_seeds * nb_training_seeds,
                                        nb_accuracies
                                    )
                                    
                                    err = np.zeros((len(attacks), nb_accuracies))
                                    for i in range(len(err)):
                                        err[i] = (1.96*np.std(tab_acc[i], axis = 0))/math.sqrt(nb_training_seeds*nb_data_distribution_seeds)
                                    
                                    plt.rcParams.update({'font.size': 12})

                                    
                                    for i, attack in enumerate(attacks):
                                        attack = attack["name"]
                                        plt.plot(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0), label = attack, color = colors[i], linestyle = tab_sign[i], marker = markers[i], markevery = 1)
                                        plt.fill_between(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0) - err[i], np.mean(tab_acc[i], axis = 0) + err[i], alpha = 0.25)

                                    plt.xlabel('Round')
                                    plt.ylabel('Accuracy')
                                    plt.xlim(0,(nb_accuracies-1)*evaluation_delta)
                                    plt.ylim(0,1)
                                    plt.grid()
                                    plt.legend()

                                    plot_name = (
                                        f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                        f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_"
                                        f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_lr_{lr}_mom_{momentum}_wd_{wd}"
                                    )
                                    
                                    plt.savefig(path_to_plot+"/"+plot_name+'_plot.pdf')
                                    plt.close()
                                    
                                    

def test_accuracy_curve_modified(path_to_results, path_to_plot, colors=colors, tab_sign=tab_sign, markers=markers, min_accuracy=0.0, plot_std=True):
        """
        THis is the modified version of byzfl's terst_accuracy_curve function, that replicates plots in Peng et Al
        """
        
        try:
            with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
                data = json.load(file)
        except Exception as e:
            logger.error(f"Failed reading config.json: {e}")
            return
        
        try:
            os.makedirs(path_to_plot, exist_ok=True)
        except OSError as error:
            logger.error(f"Failed creating directory: {error}")
        
        path_to_hyperparameters = path_to_results + "/best_hyperparameters"
        

        # <-------------- Benchmark Config ------------->
        training_seed = data["benchmark_config"]["training_seed"]
        nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
        nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
        nb_byz = data["benchmark_config"]["f"]
        nb_declared = data["benchmark_config"].get("tolerated_f", None)
        data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
        nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
        data_distributions = data["benchmark_config"]["data_distribution"]
        set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
        nb_steps = data["benchmark_config"]["nb_steps"]


        # <-------------- Evaluation and Results ------------->
        evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

        # <-------------- Model Config ------------->
        model_name = data["model"]["name"]
        dataset_name = data["model"]["dataset_name"]
        lr_list = data["model"]["learning_rate"]

        # <-------------- Honest Nodes Config ------------->
        momentum_list = data["honest_clients"]["momentum"]
        wd_list = data["honest_clients"]["weight_decay"]

        # <-------------- Aggregators Config ------------->
        aggregators = data["aggregator"]
        pre_aggregators = data["pre_aggregators"]

        # <-------------- Attacks Config ------------->
        attacks = data["attack"]

        # Ensure certain configurations are always lists
        nb_honest_clients = ensure_list(nb_honest_clients)
        nb_byz = ensure_list(nb_byz)
        nb_declared = ensure_list(nb_declared)
        data_distributions = ensure_list(data_distributions)
        aggregators = ensure_list(aggregators)

        # Pre-aggregators can be multiple or single dict; unify them
        if not pre_aggregators or isinstance(pre_aggregators[0], dict):
            pre_aggregators = [pre_aggregators]

        attacks = ensure_list(attacks)
        lr_list = ensure_list(lr_list)
        momentum_list = ensure_list(momentum_list)
        wd_list = ensure_list(wd_list)

        nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))

        for nb_honest in nb_honest_clients:
            for nb_byzantine in nb_byz:

                if nb_declared[0] is None:
                    nb_declared_list = [nb_byzantine]
                else:
                    nb_declared_list = nb_declared.copy()
                    nb_declared_list = [item for item in nb_declared_list if item >= nb_byzantine]
                
                for nb_decl in nb_declared_list:

                    if set_honest_clients_as_clients:
                        nb_nodes = nb_honest
                    else:
                        nb_nodes = nb_honest + nb_byzantine
                    
                    for data_dist in data_distributions:
                        dist_parameter_list = data_dist["distribution_parameter"]
                        dist_parameter_list = ensure_list(dist_parameter_list)
                        for dist_parameter in dist_parameter_list:
                            for pre_agg in pre_aggregators:
                                pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                                pre_agg_names = "_".join(pre_agg_list_names)
                                
                                fig, axes = plt.subplots(1,len(attacks), figsize=(5*len(attacks),5), sharey=True)
                                fig.suptitle(f"Accuracy paths, distribution: {data_dist['name']}_{str(dist_parameter)}", fontsize=14)

                                for i_agg, agg in enumerate(aggregators):

                                    hyper_file_name = (
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_{pre_agg_names}_{agg['name']}.txt"
                                    )


                                    full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                    if os.path.exists(full_path):
                                        hyperparameters = np.loadtxt(full_path)
                                        lr = hyperparameters[0]
                                        momentum = hyperparameters[1]
                                        wd = hyperparameters[2]
                                    else:
                                        lr = lr_list[0]
                                        momentum = momentum_list[0]
                                        wd = wd_list[0]

                                    tab_acc = np.zeros((
                                        len(attacks), 
                                        nb_data_distribution_seeds,
                                        nb_training_seeds,
                                        nb_accuracies
                                    ))

                                    for i, attack in enumerate(attacks):
                                        for run_dd in range(nb_data_distribution_seeds):
                                            for run in range(nb_training_seeds):
                                                file_name = (
                                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                    f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                    f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                    f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                    f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                                )
                                                acc_path = os.path.join(
                                                    path_to_results,
                                                    file_name,
                                                    f"test_accuracy_tr_seed_{run + training_seed}"
                                                    f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                                )
                                                tab_acc[i, run_dd, run] = genfromtxt(acc_path, delimiter=',')

                                    tab_acc = tab_acc.reshape(
                                        len(attacks),
                                        nb_data_distribution_seeds * nb_training_seeds,
                                        nb_accuracies
                                    )
                                    
                                    if plot_std:
                                        err = np.zeros((len(attacks), nb_accuracies))
                                        for i in range(len(err)):
                                            err[i] = (1.96*np.std(tab_acc[i], axis = 0))/math.sqrt(nb_training_seeds*nb_data_distribution_seeds)
                                        
                                    plt.rcParams.update({'font.size': 12})

                                    
                                    for i, attack in enumerate(attacks):
                                        attack = attack["name"]
                                        ax = axes[i] if isinstance(axes, np.ndarray) else axes
                                        ax.plot(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0), label = agg["name"], color = colors[i_agg], linestyle = tab_sign[i_agg], marker = markers[i_agg], markevery = 1000)
                                        
                                        if plot_std:
                                            ax.fill_between(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0) - err[i], np.mean(tab_acc[i], axis = 0) + err[i], alpha = 0.25)
                                        
                                        ax.set_title(f"{attack} attack", fontsize=10)
                                        ax.set_xlim(0,(nb_accuracies-1)*evaluation_delta)
                                        ax.grid()
                                        ax.legend()
                                        ax.set_xlabel('Round')
                                        ax.set_ylabel('Accuracy')
                                        ax.set_ylim(min_accuracy,1)
                                        ax.set_xlim(0,(nb_accuracies-1)*evaluation_delta)

                                plt.tight_layout()

                                plot_name = (
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_"
                                    f"{pre_agg_names}_lr_{lr}_mom_{momentum}_wd_{wd}"
                                )
                                
                                plt.savefig(path_to_plot+"/"+plot_name+'_plot.pdf')
                                plt.close()




def loss_heatmap(path_to_results, path_to_plot):
    """
    Creates a heatmap where the axis are the number of 
    byzantine nodes and the distribution parameter.
    Each number is the mean of the best training losses reached 
    by the model across seeds, using a specific aggregation.
    """
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        logger.error(f"Failed reading config.json: {e}")
        return
    
    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        logger.error(f"Failed creating directory: {error}")
    
    path_to_hyperparameters = path_to_results + "/best_hyperparameters" 

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = data["model"]["learning_rate"]

    # <-------------- Honest Nodes Config ------------->
    momentum_list = data["honest_clients"]["momentum"]
    wd_list = data["honest_clients"]["weight_decay"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    if nb_declared[0] is None:
        declared_equal_real = True
        nb_declared = [nb_byz[-1]]
    else:
        declared_equal_real = False

    for pre_agg in pre_aggregators:

        pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
        pre_agg_names = "_".join(pre_agg_list_names)

        for agg in aggregators:

            for data_dist in data_distributions:

                distribution_parameter_list = data_dist["distribution_parameter"]
                distribution_parameter_list = ensure_list(distribution_parameter_list)

                for nb_honest in nb_honest_clients:

                    for nb_decl in nb_declared:
                        actual_nb_byz = [item for item in nb_byz if item <= nb_decl]
                        heat_map_table = np.zeros((len(distribution_parameter_list), len(actual_nb_byz)))

                        for y, nb_byzantine in enumerate(actual_nb_byz):

                            if declared_equal_real:
                                nb_decl = nb_byzantine

                            if set_honest_clients_as_clients:
                                nb_nodes = nb_honest
                                nb_honest = nb_nodes - nb_byzantine
                            else:
                                nb_nodes = nb_honest + nb_byzantine

                            for x, dist_param in enumerate(distribution_parameter_list):

                                hyper_file_name = (
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_{pre_agg_names}_{agg['name']}.txt"
                                )

                                
                                full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                if os.path.exists(full_path):
                                    hyperparameters = np.loadtxt(full_path)
                                    lr = hyperparameters[0]
                                    momentum = hyperparameters[1]
                                    wd = hyperparameters[2]
                                else:
                                    lr = lr_list[0]
                                    momentum = momentum_list[0]
                                    wd = wd_list[0]

                                
                                lowest_loss = 0
                                for attack in attacks:

                                    config_file_name = (
                                        f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                        f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_"
                                        f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_"
                                        f"{custom_dict_to_str(attack['name'])}_lr_{lr}_mom_{momentum}_wd_{wd}"
                                    )

                                    try:
                                        with open(path_to_results+ "/" + config_file_name +'/config.json', 'r') as file:
                                            data = json.load(file)
                                    except Exception as e:
                                        logger.exception(e)

                                    nb_steps = data["benchmark_config"]["nb_steps"]

                                    losses = np.zeros(
                                        (
                                            nb_data_distribution_seeds,
                                            nb_training_seeds,
                                            nb_steps
                                        )
                                    )

                                    for run_dd in range(nb_data_distribution_seeds):
                                        for run in range(nb_training_seeds):
                                            losses[run_dd][run] = genfromtxt(
                                                f"{path_to_results}/{config_file_name}/"
                                                f"train_loss_tr_seed_{run + training_seed}_"
                                                f"dd_seed_{run_dd + data_distribution_seed}.txt",
                                                delimiter=','
                                            )

                                    losses = losses.reshape(
                                        nb_data_distribution_seeds * nb_training_seeds,
                                        nb_steps
                                    )

                                    losses = np.mean(losses, axis=0)

                                    temp_lowest_loss = np.min(losses)

                                    if temp_lowest_loss > lowest_loss:
                                        lowest_loss = temp_lowest_loss
                                    
                                heat_map_table[len(heat_map_table)-1-x][y] = lowest_loss

                        if declared_equal_real:
                            end_file_name = "tolerated_f_equal_real.pdf"
                        else:
                            end_file_name = f"tolerated_f_{nb_decl}.pdf"

                        file_name = (
                            f"train_loss_{dataset_name}_"
                            f"{model_name}_"
                            f"{custom_dict_to_str(data_dist['name'])}_"
                            f"{pre_agg_names}_"
                            f"{agg['name']}_"
                            f"nb_honest_clients_{nb_honest}_"
                            + end_file_name
                        )

                    
                        column_names = [str(dist_param) for dist_param in distribution_parameter_list]
                        row_names = [str(nb_byzantine) for nb_byzantine in actual_nb_byz]
                        column_names.reverse()

                        try:
                            os.makedirs(path_to_plot, exist_ok=True)
                        except OSError as error:
                            logger.error(f"Failed creating directory: {error}")

                        sns.heatmap(heat_map_table, xticklabels=row_names, yticklabels=column_names, cmap=sns.cm.rocket_r, annot=True)
                        plt.xlabel("Number of Byzantine clients")
                        plt.ylabel("Data heterogeneity level")
                        plt.tight_layout()
                        plt.savefig(path_to_plot +"/"+ file_name)
                        plt.close()


def test_heatmap(path_to_results, path_to_plot):
    """
    Creates a heatmap where the axis are the number of 
    byzantine nodes and the distribution parameter.
    Each number is the mean of the best accuracy reached 
    by the model across seeds, using a specific aggregation.
    """
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        logger.error(f"Failed reading config.json: {e}")
        return
    
    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        logger.error(f"Failed creating directory: {error}")
    
    path_to_hyperparameters = path_to_results + "/best_hyperparameters"

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = data["model"]["learning_rate"]

    # <-------------- Honest Nodes Config ------------->
    momentum_list = data["honest_clients"]["momentum"]
    wd_list = data["honest_clients"]["weight_decay"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    if nb_declared[0] is None:
        declared_equal_real = True
        nb_declared = [nb_byz[-1]]
    else:
        declared_equal_real = False

    for pre_agg in pre_aggregators:

        pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
        pre_agg_names = "_".join(pre_agg_list_names)

        for agg in aggregators:

            for data_dist in data_distributions:

                distribution_parameter_list = data_dist["distribution_parameter"]
                distribution_parameter_list = ensure_list(distribution_parameter_list)

                for nb_honest in nb_honest_clients:

                    for nb_decl in nb_declared:
                        actual_nb_byz = [item for item in nb_byz if item <= nb_decl]
                        heat_map_table = np.zeros((len(distribution_parameter_list), len(actual_nb_byz)))

                        for y, nb_byzantine in enumerate(actual_nb_byz):

                            if declared_equal_real:
                                nb_decl = nb_byzantine

                            if set_honest_clients_as_clients:
                                nb_nodes = nb_honest
                                nb_honest = nb_nodes - nb_byzantine
                            else:
                                nb_nodes = nb_honest + nb_byzantine

                            for x, dist_param in enumerate(distribution_parameter_list):

                                hyper_file_name = (
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_{pre_agg_names}_{agg['name']}.txt"
                                )

                                full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                if os.path.exists(full_path):
                                    hyperparameters = np.loadtxt(full_path)
                                    lr = hyperparameters[0]
                                    momentum = hyperparameters[1]
                                    wd = hyperparameters[2]
                                else:
                                    lr = lr_list[0]
                                    momentum = momentum_list[0]
                                    wd = wd_list[0]

                                
                                worst_accuracy = np.inf
                                for attack in attacks:

                                    config_file_name = (
                                        f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                        f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_"
                                        f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_"
                                        f"{custom_dict_to_str(attack['name'])}_lr_{lr}_mom_{momentum}_wd_{wd}"
                                    )

                                    try:
                                        with open(path_to_results+ "/" + config_file_name +'/config.json', 'r') as file:
                                            data = json.load(file)
                                    except Exception as e:
                                        logger.exception(e)

                                    nb_steps = data["benchmark_config"]["nb_steps"]
                                    nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))

                                    tab_acc = np.zeros(
                                        (
                                            nb_data_distribution_seeds,
                                            nb_training_seeds,
                                            nb_accuracies
                                        )
                                    )

                                    for run_dd in range(nb_data_distribution_seeds):
                                        for run in range(nb_training_seeds):
                                            tab_acc[run_dd][run] = genfromtxt(
                                                f"{path_to_results}/{config_file_name}/"
                                                f"test_accuracy_tr_seed_{run + training_seed}_"
                                                f"dd_seed_{run_dd + data_distribution_seed}.txt",
                                                delimiter=','
                                            )

                                    tab_acc = tab_acc.reshape(
                                        nb_data_distribution_seeds * nb_training_seeds,
                                        nb_accuracies
                                    )

                                    tab_acc = tab_acc.mean(axis=0)

                                    accuracy = np.max(tab_acc)

                                    if accuracy < worst_accuracy:
                                        worst_accuracy = accuracy
                                    
                                heat_map_table[len(heat_map_table)-1-x][y] = worst_accuracy
                    
                        if declared_equal_real:
                            end_file_name = "tolerated_f_equal_real.pdf"
                        else:
                            end_file_name = f"tolerated_f_{nb_decl}.pdf"

                        file_name = (
                            f"test_{dataset_name}_"
                            f"{model_name}_"
                            f"{custom_dict_to_str(data_dist['name'])}_"
                            f"{pre_agg_names}_"
                            f"{agg['name']}_"
                            f"nb_honest_clients_{nb_honest}_"
                            + end_file_name
                        )

                    
                        column_names = [str(dist_param) for dist_param in distribution_parameter_list]
                        row_names = [str(nb_byzantine) for nb_byzantine in actual_nb_byz]
                        column_names.reverse()

                        try:
                            os.makedirs(path_to_plot, exist_ok=True)
                        except OSError as error:
                            logger.error(f"Failed creating directory: {error}")

                        sns.heatmap(heat_map_table, xticklabels=row_names, yticklabels=column_names, annot=True)
                        plt.xlabel("Number of Byzantine clients")
                        plt.ylabel("Data heterogeneity level")
                        plt.tight_layout()
                        plt.savefig(path_to_plot +"/"+ file_name)
                        plt.close()


def aggregated_test_heatmap(path_to_results, path_to_plot):
    """
    Heatmap with the aggregated info of all aggregators, 
    for every region in the heatmap, it shows the aggregation 
    with the best accuracy.
    """
    try:
        with open(path_to_results+'/config.json', 'r') as file:
            data = json.load(file)
    except Exception as e:
        logger.exception(e)

    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        logger.error(f"Failed creating directory: {error}")
    
    path_to_hyperparameters = path_to_results + "/best_hyperparameters"
    
    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = data["model"]["learning_rate"]

    # <-------------- Honest Nodes Config ------------->
    momentum_list = data["honest_clients"]["momentum"]
    wd_list = data["honest_clients"]["weight_decay"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    if nb_declared[0] is None:
        declared_equal_real = True
        nb_declared = [nb_byz[-1]]
    else:
        declared_equal_real = False
    
    for pre_agg in pre_aggregators:

        for nb_honest in nb_honest_clients:

            for nb_decl in nb_declared:
                actual_nb_byz = [item for item in nb_byz if item <= nb_decl]
                pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                pre_agg_names = "_".join(pre_agg_list_names)

                for data_dist in data_distributions:

                    data_dist["distribution_parameter"] = ensure_list(data_dist["distribution_parameter"])
                    distribution_parameter_list = data_dist["distribution_parameter"]
                    heat_map_cube = np.zeros((len(aggregators), len(distribution_parameter_list), len(actual_nb_byz)))

                    for z, agg in enumerate(aggregators):

                        heat_map_table = np.zeros((len(distribution_parameter_list), len(actual_nb_byz)))

                        for y, nb_byzantine in enumerate(actual_nb_byz):

                            if declared_equal_real:
                                nb_decl = nb_byzantine

                            if set_honest_clients_as_clients:
                                nb_nodes = nb_honest
                                nb_honest = nb_nodes - nb_byzantine
                            else:
                                nb_nodes = nb_honest + nb_byzantine

                            for x, dist_param in enumerate(distribution_parameter_list):

                                hyper_file_name = (
                                    f"{dataset_name}_"
                                    f"{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_"
                                    f"{pre_agg_names}_{agg['name']}.txt"
                                )

                                
                                full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                if os.path.exists(full_path):
                                    hyperparameters = np.loadtxt(full_path)
                                    lr = hyperparameters[0]
                                    momentum = hyperparameters[1]
                                    wd = hyperparameters[2]
                                else:
                                    lr = lr_list[0]
                                    momentum = momentum_list[0]
                                    wd = wd_list[0]

                                
                                worst_accuracy = np.inf
                                for attack in attacks:
                                    config_file_name = (
                                        f"{dataset_name}_"
                                        f"{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                        f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_"
                                        f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_"
                                        f"{custom_dict_to_str(attack['name'])}_"
                                        f"lr_{lr}_"
                                        f"mom_{momentum}_"
                                        f"wd_{wd}"
                                    )

                                    try:
                                        with open(path_to_results+ "/" + config_file_name +'/config.json', 'r') as file:
                                            data = json.load(file)
                                    except Exception as e:
                                        logger.exception(e)

                                    nb_steps = data["benchmark_config"]["nb_steps"]
                                    nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))

                                    tab_acc = np.zeros(
                                        (
                                            nb_data_distribution_seeds,
                                            nb_training_seeds,
                                            nb_accuracies
                                        )
                                    )

                                    for run_dd in range(nb_data_distribution_seeds):
                                        for run in range(nb_training_seeds):
                                            tab_acc[run_dd][run] = genfromtxt(
                                                f"{path_to_results}/{config_file_name}/"
                                                f"test_accuracy_tr_seed_{run + training_seed}_"
                                                f"dd_seed_{run_dd + data_distribution_seed}.txt",
                                                delimiter=','
                                            )

                                    tab_acc = tab_acc.reshape(
                                        nb_data_distribution_seeds * nb_training_seeds,
                                        nb_accuracies
                                    )
                                    
                                    tab_acc = tab_acc.mean(axis=0)
                                    accuracy = np.max(tab_acc)

                                    if accuracy < worst_accuracy:
                                        worst_accuracy = accuracy
                                    
                                heat_map_table[len(heat_map_table)-1-x][y] = worst_accuracy

                        heat_map_cube[z] = heat_map_table
                    

                    if declared_equal_real:
                        end_file_name = "tolerated_f_equal_real.pdf"
                    else:
                        end_file_name = f"tolerated_f_{nb_decl}.pdf"

                    file_name = (
                        f"best_test_{dataset_name}_"
                        f"{model_name}_"
                        f"{custom_dict_to_str(data_dist['name'])}_"
                        f"{pre_agg_names}_"
                        f"nb_honest_clients_{nb_honest}_"
                        + end_file_name
                    )
                    
                    column_names = [str(dist_param) for dist_param in distribution_parameter_list]
                    row_names = [str(nb_byzantine) for nb_byzantine in actual_nb_byz]
                    column_names.reverse()

                    heat_map_table = np.max(heat_map_cube, axis=0)
                    sns.heatmap(heat_map_table, xticklabels=row_names, yticklabels=column_names, annot=True)
                    plt.xlabel("Number of Byzantine clients")
                    plt.ylabel("Data heterogeneity level")
                    plt.tight_layout()
                    plt.savefig(path_to_plot +"/"+ file_name)
                    plt.close()


def plot_gradients_scattering(path_to_results, path_to_plot):
    """
    Plot honest and poisoned gradient scatterings for different configurations.
    Honest and poisoned gradients are plotted on the same plot as in Peng et al.
    For now, scatterings are plotted for each training step. This will have to be modified to account for
    a delta in scattering measurement (cf accuracy plotting)
    """
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        logger.error(f"Failed reading config.json: {e}")
        return
    
    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        logger.error(f"Failed creating directory: {error}")
    
    path_to_hyperparameters = path_to_results + "/best_hyperparameters"
    

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = data["model"]["learning_rate"]

    # <-------------- Honest Nodes Config ------------->
    momentum_list = data["honest_clients"]["momentum"]
    wd_list = data["honest_clients"]["weight_decay"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]
    
    # <-------------- Gradient type ------------->
    scatter_momentums = data["evaluation_and_results"].get("scatter_momentums", False)

    # Ensure certain configurations are always lists
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    nb_scatterings = nb_steps//evaluation_delta

    for nb_honest in nb_honest_clients:
        for nb_byzantine in nb_byz:

            if nb_declared[0] is None:
                nb_declared_list = [nb_byzantine]
            else:
                nb_declared_list = nb_declared.copy()
                nb_declared_list = [item for item in nb_declared_list if item >= nb_byzantine]
            
            for nb_decl in nb_declared_list:

                if set_honest_clients_as_clients:
                    nb_nodes = nb_honest
                else:
                    nb_nodes = nb_honest + nb_byzantine
                
                for data_dist in data_distributions:
                    dist_parameter_list = data_dist["distribution_parameter"]
                    dist_parameter_list = ensure_list(dist_parameter_list)
                    for dist_parameter in dist_parameter_list:
                        for pre_agg in pre_aggregators:
                            pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                            pre_agg_names = "_".join(pre_agg_list_names)
                            for agg in aggregators:
                                tab_scat_ksi = np.zeros((len(attacks),
                                        nb_data_distribution_seeds,
                                        nb_training_seeds,
                                        nb_scatterings
                                    ))
                                    
                                tab_scat_A = np.zeros((len(attacks),
                                    nb_data_distribution_seeds,
                                    nb_training_seeds,
                                    nb_scatterings
                                ))
                                for i_attack, attack in enumerate(attacks): #in contrast with accuracy path, we separate attacks here
                                    hyper_file_name = (
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_{pre_agg_names}_{agg['name']}.txt"
                                    )


                                    full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                    if os.path.exists(full_path):
                                        hyperparameters = np.loadtxt(full_path)
                                        lr = hyperparameters[0]
                                        momentum = hyperparameters[1]
                                        wd = hyperparameters[2]
                                    else:
                                        lr = lr_list[0]
                                        momentum = momentum_list[0]
                                        wd = wd_list[0]


                                    

                                    for run_dd in range(nb_data_distribution_seeds):
                                        for run in range(nb_training_seeds):
                                            file_name = (
                                                f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                            )
                                            path_scat_ksi = os.path.join(
                                                path_to_results,
                                                file_name,
                                                f"honest_scattering_tr_seed_{training_seed}_dd_seed_{data_distribution_seed}",
                                                f"honest_gradients_scattering.txt"
                                            )
                                            path_scat_A= os.path.join(
                                                path_to_results,
                                                file_name,
                                                f"poisoned_scattering_tr_seed_{training_seed}_dd_seed_{data_distribution_seed}",
                                                f"poisoned_gradients_scattering.txt"
                                            )
                                            tab_scat_ksi[i_attack,run_dd, run] = genfromtxt(path_scat_ksi)
                                            tab_scat_A[i_attack,run_dd, run] = genfromtxt(path_scat_A)

                                    #this part might need to be modified for the purpose of gradient scattering...
                                    tab_scat_ksi = tab_scat_ksi.reshape(len(attacks),
                                        nb_data_distribution_seeds * nb_training_seeds,
                                        nb_scatterings
                                    )
                                    
                                    tab_scat_A = tab_scat_A.reshape(len(attacks),
                                        nb_data_distribution_seeds * nb_training_seeds,
                                        nb_scatterings
                                    )
                                    err_A=np.zeros((len(attacks), nb_scatterings))
                                    err_ksi=np.zeros((len(attacks), nb_scatterings))

                                    err_ksi[i_attack] = (1.96*np.std(tab_scat_ksi[i_attack], axis = 0))/math.sqrt(nb_training_seeds*nb_data_distribution_seeds)
                                    err_A[i_attack] = (1.96*np.std(tab_scat_A[i_attack], axis = 0))/math.sqrt(nb_training_seeds*nb_data_distribution_seeds)

                                    plt.rcParams.update({'font.size': 12})
                                    
                                    tab_scat_ksi = tab_scat_ksi.reshape(len(attacks),
                                        nb_data_distribution_seeds, nb_training_seeds,
                                        nb_scatterings
                                    )
                                    
                                    tab_scat_A = tab_scat_A.reshape(len(attacks),
                                        nb_data_distribution_seeds, nb_training_seeds,
                                        nb_scatterings
                                    )
                                
                                tab_scat_ksi = tab_scat_ksi.reshape(len(attacks),
                                        nb_data_distribution_seeds * nb_training_seeds,
                                        nb_scatterings
                                    )
                                    
                                tab_scat_A = tab_scat_A.reshape(len(attacks),
                                        nb_data_distribution_seeds * nb_training_seeds,
                                        nb_scatterings
                                    )
                                
                                #i_ksi is the index of the attack with the highest maximal scattering value for ksi. Hence we take the argmax:
                                i_ksi = np.argmax(tab_scat_ksi.mean(axis=1).max(axis=1))

                                gradient_type = "Momentum" if scatter_momentums else "Raw"
                                plt.plot(np.linspace(0, (nb_scatterings-1)*evaluation_delta, nb_scatterings), np.mean(tab_scat_ksi[i_ksi], axis = 0), label = r"$ksi$", color = colors[0], linestyle = tab_sign[0], marker = None, markevery = 1)
                                # plt.fill_between(np.linspace(0, (nb_scatterings-1)*evaluation_delta, nb_scatterings), np.mean(tab_scat_ksi[i_ksi], axis = 0) - err_ksi, np.mean(tab_scat_ksi[i_ksi], axis = 0) + err_ksi, alpha = 0.25)

                                for i_attack, attack in enumerate(attacks):
                                    plt.plot(np.linspace(0, (nb_scatterings-1)*evaluation_delta, nb_scatterings), np.mean(tab_scat_A[i_attack], axis = 0), label = f"A - {attack["name"]}", color = colors[i_attack+1], linestyle = tab_sign[1], marker = None, markevery = 1)
                                    # plt.fill_between(np.linspace(0, (nb_scatterings-1)*evaluation_delta, nb_scatterings), np.mean(tab_scat_A[i_attack], axis = 0) - err_A[i_attack], np.mean(tab_scat_A[i_attack], axis = 0) + err_A[i_attack], alpha = 0.25)

                                plt.xlim(0,(nb_scatterings-1)*evaluation_delta)
                                plt.xlabel('Round')
                                plt.ylabel("Gradient heterogeneity")
                                plt.title(f"{gradient_type} gradient scatterings, {data_dist["name"]}-{str(dist_parameter)} distribution")
                                plt.grid()
                                plt.legend()

                                plot_name = (
                                    "honest_gradient_scattering"
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_"
                                    f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_lr_{lr}_mom_{momentum}_wd_{wd}"
                                    f"_scatter_momentums_{str(scatter_momentums)}"
                                )
                                
                                plt.savefig(path_to_plot+"/"+plot_name+'_plot.pdf')
                                plt.close()




def plot_maximum_regular_feature_mean(path_to_results, path_to_plot):
    """
    Plots the poisoned gradient scatterings for different configurations.
    """
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        logger.error(f"Failed reading config.json: {e}")
        return
    
    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        logger.error(f"Failed creating directory: {error}")
    
    path_to_hyperparameters = path_to_results + "/best_hyperparameters"
    

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = data["model"]["learning_rate"]

    # <-------------- Honest Nodes Config ------------->
    momentum_list = data["honest_clients"]["momentum"]
    wd_list = data["honest_clients"]["weight_decay"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))

    for nb_honest in nb_honest_clients:
        for nb_byzantine in nb_byz:

            if nb_declared[0] is None:
                nb_declared_list = [nb_byzantine]
            else:
                nb_declared_list = nb_declared.copy()
                nb_declared_list = [item for item in nb_declared_list if item >= nb_byzantine]
            
            for nb_decl in nb_declared_list:

                if set_honest_clients_as_clients:
                    nb_nodes = nb_honest
                else:
                    nb_nodes = nb_honest + nb_byzantine
                
                for data_dist in data_distributions:
                    dist_parameter_list = data_dist["distribution_parameter"]
                    dist_parameter_list = ensure_list(dist_parameter_list)
                    for dist_parameter in dist_parameter_list:
                        for pre_agg in pre_aggregators:
                            pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                            pre_agg_names = "_".join(pre_agg_list_names)
                            for agg in aggregators:

                                hyper_file_name = (
                                f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_{pre_agg_names}_{agg['name']}.txt"
                                )


                                full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                if os.path.exists(full_path):
                                    hyperparameters = np.loadtxt(full_path)
                                    lr = hyperparameters[0]
                                    momentum = hyperparameters[1]
                                    wd = hyperparameters[2]
                                else:
                                    lr = lr_list[0]
                                    momentum = momentum_list[0]
                                    wd = wd_list[0]

                                tab_acc = np.zeros((
                                    len(attacks), 
                                    nb_data_distribution_seeds,
                                    nb_training_seeds,
                                    nb_accuracies
                                ))

                                for i, attack in enumerate(attacks):
                                    for run_dd in range(nb_data_distribution_seeds):
                                        for run in range(nb_training_seeds):
                                            file_name = (
                                                f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                            )
                                            acc_path = os.path.join(
                                                path_to_results,
                                                file_name,
                                                f"feature_norm_tr_seed_{training_seed}_dd_seed_{data_distribution_seed}",
                                                f"honest_gradients_scattering.txt"
                                            )
                                            tab_acc[i, run_dd, run] = genfromtxt(acc_path, delimiter=',')

                                tab_acc = tab_acc.reshape(
                                    len(attacks),
                                    nb_data_distribution_seeds * nb_training_seeds,
                                    nb_accuracies
                                )
                                
                                err = np.zeros((len(attacks), nb_accuracies))
                                for i in range(len(err)):
                                    err[i] = (1.96*np.std(tab_acc[i], axis = 0))/math.sqrt(nb_training_seeds*nb_data_distribution_seeds)
                                
                                plt.rcParams.update({'font.size': 12})

                                
                                for i, attack in enumerate(attacks):
                                    attack = attack["name"]
                                    plt.plot(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0), label = attack, color = colors[i], linestyle = tab_sign[i], marker = markers[i], markevery = 1)
                                    plt.fill_between(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0) - err[i], np.mean(tab_acc[i], axis = 0) + err[i], alpha = 0.25)

                                plt.xlabel('Round')
                                plt.ylabel(r"$max_{\omega \in R}\frac{1}{J}\lVert \sum_{k=0}^J a_{(\omega, k)} \rVert$")
                                plt.grid()
                                plt.legend()

                                plot_name = (
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_"
                                    f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_lr_{lr}_mom_{momentum}_wd_{wd}"
                                )
                                
                                plt.savefig(path_to_plot+"/"+plot_name+'_plot.pdf')
                                plt.close()


def plot_workers_feature_variance(path_to_results, path_to_plot):
    """
    Plots each worker's feature variance for different configurations.
    """
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        logger.error(f"Failed reading config.json: {e}")
        return
    
    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        logger.error(f"Failed creating directory: {error}")
    
    path_to_hyperparameters = path_to_results + "/best_hyperparameters"
    

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = data["model"]["learning_rate"]

    # <-------------- Honest Nodes Config ------------->
    momentum_list = data["honest_clients"]["momentum"]
    wd_list = data["honest_clients"]["weight_decay"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))

    for nb_honest in nb_honest_clients:
        for nb_byzantine in nb_byz:

            if nb_declared[0] is None:
                nb_declared_list = [nb_byzantine]
            else:
                nb_declared_list = nb_declared.copy()
                nb_declared_list = [item for item in nb_declared_list if item >= nb_byzantine]
            
            for nb_decl in nb_declared_list:

                if set_honest_clients_as_clients:
                    nb_nodes = nb_honest
                else:
                    nb_nodes = nb_honest + nb_byzantine
                
                for data_dist in data_distributions:
                    dist_parameter_list = data_dist["distribution_parameter"]
                    dist_parameter_list = ensure_list(dist_parameter_list)
                    for dist_parameter in dist_parameter_list:
                        for pre_agg in pre_aggregators:
                            pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                            pre_agg_names = "_".join(pre_agg_list_names)
                            for agg in aggregators:

                                hyper_file_name = (
                                f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_{pre_agg_names}_{agg['name']}.txt"
                                )


                                full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                if os.path.exists(full_path):
                                    hyperparameters = np.loadtxt(full_path)
                                    lr = hyperparameters[0]
                                    momentum = hyperparameters[1]
                                    wd = hyperparameters[2]
                                else:
                                    lr = lr_list[0]
                                    momentum = momentum_list[0]
                                    wd = wd_list[0]

                                tab_acc = np.zeros((
                                    len(attacks), 
                                    nb_nodes,
                                    nb_data_distribution_seeds,
                                    nb_training_seeds,
                                    nb_accuracies
                                ))

                                for i, attack in enumerate(attacks):
                                    for run_dd in range(nb_data_distribution_seeds):
                                        for run in range(nb_training_seeds):
                                            for client_id in range(nb_nodes):
                                                file_name = (
                                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                    f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                    f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                    f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                    f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                                )
                                                acc_path = os.path.join(
                                                    path_to_results,
                                                    file_name,
                                                    f"feature_variance_tr_seed_{training_seed}_dd_seed_{data_distribution_seed}",
                                                    f"feature_variance_client_{client_id}.txt"
                                                )
                                                tab_acc[i, run_dd, run, client_id] = genfromtxt(acc_path, delimiter=',')

                                tab_acc = tab_acc.reshape(
                                    len(attacks),
                                    nb_data_distribution_seeds * nb_training_seeds,
                                    nb_accuracies,
                                    nb_nodes
                                )
                                
                                # a discuter (liste de listes)
                                err = np.zeros((len(attacks), nb_accuracies))
                                for i in range(len(err)):
                                    err[i] = (1.96*np.std(tab_acc[i], axis = 0))/math.sqrt(nb_training_seeds*nb_data_distribution_seeds)
                                
                                plt.rcParams.update({'font.size': 12})

                                
                                for i, attack in enumerate(attacks):
                                    attack = attack["name"]
                                    plt.plot(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0), label = attack, color = colors[i], linestyle = tab_sign[i], marker = markers[i], markevery = 1)
                                    plt.fill_between(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0) - err[i], np.mean(tab_acc[i], axis = 0) + err[i], alpha = 0.25)

                                plt.xlabel('Round')
                                plt.ylabel(r"$max_{\omega \in R}\frac{1}{J}\lVert \sum_{k=0}^J a_{(\omega, k)} \rVert$")
                                plt.grid()
                                plt.legend()

                                plot_name = (
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_"
                                    f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_lr_{lr}_mom_{momentum}_wd_{wd}"
                                )
                                
                                plt.savefig(path_to_plot+"/"+plot_name+'_plot.pdf')
                                plt.close()


def gradient_estimator_variance(path_to_results, path_to_plot):
    """
    Plots the poisoned gradient scatterings for different configurations.
    """

    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        logger.error(f"Failed reading config.json: {e}")
        return
    
    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        logger.error(f"Failed creating directory: {error}")
    
    path_to_hyperparameters = path_to_results + "/best_hyperparameters"
    

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = data["model"]["learning_rate"]

    # <-------------- Honest Nodes Config ------------->
    momentum_list = data["honest_clients"]["momentum"]
    wd_list = data["honest_clients"]["weight_decay"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))

    for nb_honest in nb_honest_clients:
        for nb_byzantine in nb_byz:

            if nb_declared[0] is None:
                nb_declared_list = [nb_byzantine]
            else:
                nb_declared_list = nb_declared.copy()
                nb_declared_list = [item for item in nb_declared_list if item >= nb_byzantine]
            
            for nb_decl in nb_declared_list:

                if set_honest_clients_as_clients:
                    nb_nodes = nb_honest
                else:
                    nb_nodes = nb_honest + nb_byzantine
                
                for data_dist in data_distributions:
                    dist_parameter_list = data_dist["distribution_parameter"]
                    dist_parameter_list = ensure_list(dist_parameter_list)
                    for dist_parameter in dist_parameter_list:
                        for pre_agg in pre_aggregators:
                            pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                            pre_agg_names = "_".join(pre_agg_list_names)
                            for agg in aggregators:

                                hyper_file_name = (
                                f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_{pre_agg_names}_{agg['name']}.txt"
                                )


                                full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                if os.path.exists(full_path):
                                    hyperparameters = np.loadtxt(full_path)
                                    lr = hyperparameters[0]
                                    momentum = hyperparameters[1]
                                    wd = hyperparameters[2]
                                else:
                                    lr = lr_list[0]
                                    momentum = momentum_list[0]
                                    wd = wd_list[0]

                                tab_acc = np.zeros((
                                    len(attacks), 
                                    nb_data_distribution_seeds,
                                    nb_training_seeds,
                                    nb_accuracies
                                ))

                                for i, attack in enumerate(attacks):
                                    for run_dd in range(nb_data_distribution_seeds):
                                        for run in range(nb_training_seeds):
                                            file_name = (
                                                f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                            )
                                            acc_path = os.path.join(
                                                path_to_results,
                                                file_name,
                                                f"maximal_gradient_variance_tr_seed_{training_seed}_dd_seed_{data_distribution_seed}",
                                                f"maximal_gradient_estimator_variance.txt"
                                            )
                                            tab_acc[i, run_dd, run] = genfromtxt(acc_path, delimiter=',')

                                tab_acc = tab_acc.reshape(
                                    len(attacks),
                                    nb_data_distribution_seeds * nb_training_seeds,
                                    nb_accuracies
                                )
                                
                                err = np.zeros((len(attacks), nb_accuracies))
                                for i in range(len(err)):
                                    err[i] = (1.96*np.std(tab_acc[i], axis = 0))/math.sqrt(nb_training_seeds*nb_data_distribution_seeds)
                                
                                plt.rcParams.update({'font.size': 12})

                                
                                for i, attack in enumerate(attacks):
                                    attack = attack["name"]
                                    plt.plot(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0), label = attack, color = colors[i], linestyle = tab_sign[i], marker = markers[i], markevery = 1)
                                    plt.fill_between(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0) - err[i], np.mean(tab_acc[i], axis = 0) + err[i], alpha = 0.25)

                                plt.xlabel('Round')
                                plt.ylabel(r"$\sigma^2$")
                                plt.grid()
                                plt.legend()

                                plot_name = (
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_"
                                    f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_lr_{lr}_mom_{momentum}_wd_{wd}"
                                )
                                
                                plt.savefig(path_to_plot+"/"+plot_name+'_plot.pdf')
                                plt.close()


def rho_SGD_convergence_rate():
    # laquelle :)
    pass


def plot_worker_class_distribution(clients, path, num_classes, dist_name, dd_seed):
    """
    Description
    -----------
    plot and save the repartition of classes accross the client's data subset
        clients = Client object list
    """
    nb_clients = len(clients)

    colors = plt.cm.tab10.colors  
    
    partitions = []
    for client in clients:
        subset = client.training_dataloader.dataset
        targets = np.array([subset.dataset.targets[idx] for idx in subset.indices])
        counts = Counter(targets)
        total = len(targets)
        prop = [counts.get(c, 0) for c in range(num_classes)]
        partitions.append(prop)
    partitions=np.array(partitions, dtype=int)
    
    # Plot horizontal empilé
    left = np.zeros(nb_clients)
    plt.figure()
    for c in range(num_classes):
        plt.barh(range(nb_clients), partitions[:, c], left=left,
                color=colors[c], label=f'Classe {c}')
        left += partitions[:, c]

    plt.legend()
    plt.yticks(range(nb_clients))
    plt.savefig(os.path.join(path,f"worker_distributions_dd_seed_{dd_seed}.png"), dpi=300)
    plt.close()
    return partitions

def compute_exclusivity(partitions):
    """
    computes the exclusivity measure of the partition

    Args:
        partition (np.array): (nb_worker, nb_classes) array
    """

    exclusivity=[]
    for client_distrib in partitions:
        majority_class= np.argmax(client_distrib) #nb_honest is the index of the byzantine worker
        exclusivity.append(client_distrib[majority_class]/partitions[:,majority_class].sum())
    return exclusivity                                        
    
    
def evaluate_impact_exclusivity(path_to_results, path_to_plot, colors=colors, tab_sign=tab_sign, markers=markers):
    """
    This script plots accuracy performance against exclusivity of the data held by the byzantine worker
    """
    
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return
    
    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}")
    
    path_to_hyperparameters = path_to_results + "/best_hyperparameters"
    

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = False
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = data["model"]["learning_rate"]
    nb_labels= data["model"]["nb_labels"]

    # <-------------- Honest Nodes Config ------------->
    momentum_list = data["honest_clients"]["momentum"]
    wd_list = data["honest_clients"]["weight_decay"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))

    for nb_honest in nb_honest_clients:
        for nb_byzantine in nb_byz:

            if nb_declared[0] is None:
                nb_declared_list = [nb_byzantine]
            else:
                nb_declared_list = nb_declared.copy()
                nb_declared_list = [item for item in nb_declared_list if item >= nb_byzantine]
            
            for nb_decl in nb_declared_list:

                if set_honest_clients_as_clients:
                    nb_nodes = nb_honest
                else:
                    nb_nodes = nb_honest + nb_byzantine
                
                for data_dist in data_distributions:
                    dist_parameter_list = data_dist["distribution_parameter"]
                    dist_parameter_list = ensure_list(dist_parameter_list)
                    for dist_parameter in dist_parameter_list:
                        for pre_agg in pre_aggregators:
                            pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                            pre_agg_names = "_".join(pre_agg_list_names)
                            
                            fig, axes = plt.subplots(1,len(attacks), figsize=(5*len(attacks),5), sharey=True)
                            fig.suptitle(f"{data_dist['name']}, {str(dist_parameter)}", fontsize=12)

                            for i_agg, agg in enumerate(aggregators):

                                hyper_file_name = (
                                f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_{pre_agg_names}_{agg['name']}.txt"
                                )


                                full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                if os.path.exists(full_path):
                                    hyperparameters = np.loadtxt(full_path)
                                    lr = hyperparameters[0]
                                    momentum = hyperparameters[1]
                                    wd = hyperparameters[2]
                                else:
                                    lr = lr_list[0]
                                    momentum = momentum_list[0]
                                    wd = wd_list[0]

                                tab_acc = np.zeros((
                                    len(attacks), 
                                    nb_data_distribution_seeds,
                                    nb_training_seeds,
                                    nb_accuracies
                                ))
                                tab_distrib = np.zeros((
                                    len(attacks), 
                                    nb_data_distribution_seeds,
                                    nb_honest+nb_byzantine,nb_labels
                                ))

                                for i, attack in enumerate(attacks):
                                    file_name = (
                                                f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                            )
                                    for run_dd in range(nb_data_distribution_seeds):
                                        
                                        exclusivity_path = os.path.join(
                                                path_to_results,
                                                file_name,
                                                f"distributions/worker_distributions_dd_seed_{run_dd + data_distribution_seed}.txt"
                                            )
                                        tab_distrib[i, run_dd] = genfromtxt(exclusivity_path, delimiter=',')

                                        for run in range(nb_training_seeds):
                                            
                                            acc_path = os.path.join(
                                                path_to_results,
                                                file_name,
                                                f"test_accuracy_tr_seed_{run + training_seed}"
                                                f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                            )
                                            tab_acc[i, run_dd, run] = genfromtxt(acc_path, delimiter=',')

                                plt.rcParams.update({'font.size': 12})
                                                           
                                for i, attack in enumerate(attacks):
                                    attack = attack["name"]
                                    ax = axes[i] if isinstance(axes, np.ndarray) else axes
                                    
                                    #calculate exclusivity:
                                    exclusivity=[]
                                    max_accuracy=[]
                                    for seed_i, distrib_seed in enumerate(tab_distrib[i]):
                                        majority_class= np.argmax(distrib_seed[nb_honest]) #nb_honest is the index of the byzantine worker
                                        exclusivity.append(distrib_seed[nb_honest][majority_class]/distrib_seed[:,majority_class].sum())
                                        max_accuracy.append(np.max(tab_acc[i][seed_i]))
                                        
                                    if agg['name'] == "Average":
                                        col = (0.635, 0.078, 0.184) #dark red
                                        mark= 'o'
                                        a=1.0
                                    else:
                                        col=(0.000, 0.447, 0.741)  # blue
                                        mark= markers[i_agg]
                                        a=0.6
                                    
                                    ax.scatter(exclusivity,max_accuracy,marker= mark,color=col, label=agg["name"], alpha=a)
                                    ax.set_title(f"{attack} attack", fontsize=10)
                                    ax.grid()
                                    ax.legend()
                                    ax.set_xlabel('Exclusivity of byzantine data')
                                    ax.set_ylabel('Maximal accuracy')

                            plt.tight_layout()

                            plot_name = (
                                f"exclusivity_study_"
                                f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_"
                                f"{pre_agg_names}_lr_{lr}_mom_{momentum}_wd_{wd}"
                            )
                            
                            plt.savefig(path_to_plot+"/"+plot_name+'_plot.pdf')
                            plt.close()

def evaluate_impact_subset_size(path_to_results, path_to_plot, colors=colors, tab_sign=tab_sign, markers=markers):
    """
    This script plots accuracy performance against the size of the subset held by the byzantine worker (smaller subsets translate into weaker attacks!)
    """
    
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return
    
    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}")
    
    path_to_hyperparameters = path_to_results + "/best_hyperparameters"
    

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr_list = data["model"]["learning_rate"]
    nb_labels= data["model"]["nb_labels"]

    # <-------------- Honest Nodes Config ------------->
    momentum_list = data["honest_clients"]["momentum"]
    wd_list = data["honest_clients"]["weight_decay"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))

    for nb_honest in nb_honest_clients:
        for nb_byzantine in nb_byz:

            if nb_declared[0] is None:
                nb_declared_list = [nb_byzantine]
            else:
                nb_declared_list = nb_declared.copy()
                nb_declared_list = [item for item in nb_declared_list if item >= nb_byzantine]
            
            for nb_decl in nb_declared_list:

                if set_honest_clients_as_clients:
                    nb_nodes = nb_honest
                else:
                    nb_nodes = nb_honest + nb_byzantine
                
                for data_dist in data_distributions:
                    dist_parameter_list = data_dist["distribution_parameter"]
                    dist_parameter_list = ensure_list(dist_parameter_list)
                    for dist_parameter in dist_parameter_list:
                        for pre_agg in pre_aggregators:
                            pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                            pre_agg_names = "_".join(pre_agg_list_names)
                            
                            fig, axes = plt.subplots(1,len(attacks), figsize=(5*len(attacks),5), sharey=True)
                            fig.suptitle(f"{data_dist['name']}, {str(dist_parameter)}", fontsize=12)

                            for i_agg, agg in enumerate(aggregators):

                                hyper_file_name = (
                                f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_{pre_agg_names}_{agg['name']}.txt"
                                )


                                full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                if os.path.exists(full_path):
                                    hyperparameters = np.loadtxt(full_path)
                                    lr = hyperparameters[0]
                                    momentum = hyperparameters[1]
                                    wd = hyperparameters[2]
                                else:
                                    lr = lr_list[0]
                                    momentum = momentum_list[0]
                                    wd = wd_list[0]

                                tab_acc = np.zeros((
                                    len(attacks), 
                                    nb_data_distribution_seeds,
                                    nb_training_seeds,
                                    nb_accuracies
                                ))
                                tab_distrib = np.zeros((
                                    len(attacks), 
                                    nb_data_distribution_seeds,
                                    nb_honest+nb_byzantine,nb_labels
                                ))

                                for i, attack in enumerate(attacks):
                                    file_name = (
                                                f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                            )
                                    for run_dd in range(nb_data_distribution_seeds):
                                        
                                        exclusivity_path = os.path.join(
                                                path_to_results,
                                                file_name,
                                                f"distributions/worker_distributions_dd_seed_{run_dd + data_distribution_seed}.txt"
                                            )
                                        tab_distrib[i, run_dd] = genfromtxt(exclusivity_path, delimiter=',')

                                        for run in range(nb_training_seeds):
                                            
                                            acc_path = os.path.join(
                                                path_to_results,
                                                file_name,
                                                f"test_accuracy_tr_seed_{run + training_seed}"
                                                f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                            )
                                            tab_acc[i, run_dd, run] = genfromtxt(acc_path, delimiter=',')

                                plt.rcParams.update({'font.size': 12})
                                                           
                                for i, attack in enumerate(attacks):
                                    attack = attack["name"]
                                    ax = axes[i] if isinstance(axes, np.ndarray) else axes
                                    
                                    #calculate exclusivity:
                                    subset_sizes=[]
                                    max_accuracy=[]
                                    for seed_i, distrib_seed in enumerate(tab_distrib[i]):
                                        subset_sizes.append(int(np.sum(distrib_seed[nb_honest]))) #nb_honest is the index of the byzantine worker
                                        max_accuracy.append(np.max(tab_acc[i][seed_i]))
                                        
                                    if agg['name'] == "Average":
                                        col = (0.635, 0.078, 0.184) #dark red
                                        mark= 'o'
                                        a=1.0
                                    else:
                                        col=(0.000, 0.447, 0.741)  # blue
                                        mark= markers[i_agg]
                                        a=0.6
                                    
                                    ax.scatter(subset_sizes,max_accuracy,marker= mark,color=col, label=agg["name"], alpha=a)
                                    ax.set_title(f"{attack} attack", fontsize=10)
                                    ax.grid()
                                    ax.legend()
                                    ax.set_xlabel('Size of byzantine dataset')
                                    ax.set_ylabel('Maximal accuracy')

                            plt.tight_layout()

                            plot_name = (
                                f"subset_size_study_"
                                f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_"
                                f"{pre_agg_names}_lr_{lr}_mom_{momentum}_wd_{wd}"
                            )
                            
                            plt.savefig(path_to_plot+"/"+plot_name+'_plot.pdf')
                            plt.close()
                            

def evaluate_impact_exclusivity_adaptive_bins(path_to_results, path_to_plot, colors=colors, tab_sign=tab_sign, markers=markers):
    """
    Trace des boxplots de (Best_Robust_Acc - Mean_Acc).
    Les bins sont adaptatives : chaque bin contient environ 6 points de données,
    triés par exclusivité croissante.
    """
    
    # --- CONFIGURATION ---
    mean_agg_name = "Average"
    points_per_bin = 6  # Nombre cible de points par boxplot
    
    # Lecture config
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return
    
    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}")
    
    # <-------------- Config Loading ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = ensure_list(data["benchmark_config"]["nb_honest_clients"])
    nb_byz = ensure_list(data["benchmark_config"]["f"])
    nb_declared = ensure_list(data["benchmark_config"].get("tolerated_f", None))
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = ensure_list(data["benchmark_config"]["data_distribution"])
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    
    # Model & Honest Config
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    lr = ensure_list(data["model"]["learning_rate"])[0]
    momentum = ensure_list(data["honest_clients"]["momentum"])[0]
    wd = ensure_list(data["honest_clients"]["weight_decay"])[0]

    aggregators = ensure_list(data["aggregator"])
    pre_aggregators = data["pre_aggregators"]
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]
    
    pre_agg_names = "_".join([pa['name'] for pa in pre_aggregators[0]])
    attacks = ensure_list(data["attack"])
    
    # --- BOUCLES PRINCIPALES ---
    for nb_honest in nb_honest_clients:
        for nb_byzantine in nb_byz:
            
            # Gestion nb_declared
            if nb_declared[0] is None:
                curr_nb_declared = [nb_byzantine]
            else:
                curr_nb_declared = [d for d in nb_declared if d >= nb_byzantine]
            
            for nb_decl in curr_nb_declared:
                
                if set_honest_clients_as_clients:
                    nb_nodes = nb_honest
                else:
                    nb_nodes = nb_honest + nb_byzantine

                # Création de la figure
                fig, axes = plt.subplots(1, len(attacks), figsize=(6*len(attacks), 7), sharey=True) # Hauteur augmentée pour les labels X
                if len(attacks) == 1: axes = [axes]
                
                fig.suptitle(f"Gain Robustesse (Bins de {points_per_bin} seeds) - n={nb_nodes}, f={nb_byzantine}", fontsize=14)

                for i_atk, attack in enumerate(attacks):
                    attack_name = attack['name']
                    ax = axes[i_atk]

                    all_points = [] # Stocke (exclusivité, delta)

                    # --- 1. COLLECTE DES DONNÉES ---
                    for data_dist in data_distributions:
                        dist_params = ensure_list(data_dist["distribution_parameter"])
                        
                        for dist_param in dist_params:
                            for run_dd in range(nb_data_distribution_seeds):
                                # Calcul Exclusivité
                                exclusivity_path = os.path.join(
                                    path_to_results,
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_{mean_agg_name}_"
                                    f"{pre_agg_names}_{custom_dict_to_str(attack_name)}_lr_{lr}_mom_{momentum}_wd_{wd}",
                                    f"distributions/worker_distributions_dd_seed_{run_dd + data_distribution_seed}.txt"
                                )
                                
                                try:
                                    distrib = genfromtxt(exclusivity_path, delimiter=',')
                                    majority_class = np.argmax(distrib[nb_honest]) 
                                    excl = distrib[nb_honest][majority_class] / distrib[:, majority_class].sum()
                                except Exception:
                                    excl = np.nan
                                
                                if np.isnan(excl): continue

                                for run_tr in range(nb_training_seeds):
                                    # Fonction interne chargement
                                    def get_acc(agg_n):
                                        f_name = (
                                            f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                            f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_{custom_dict_to_str(agg_n)}_"
                                            f"{pre_agg_names}_{custom_dict_to_str(attack_name)}_lr_{lr}_mom_{momentum}_wd_{wd}"
                                        )
                                        p = os.path.join(path_to_results, f_name, 
                                                         f"test_accuracy_tr_seed_{run_tr + training_seed}_dd_seed_{run_dd + data_distribution_seed}.txt")
                                        if os.path.exists(p):
                                            return np.max(genfromtxt(p, delimiter=','))
                                        return np.nan

                                    # Delta Calcul
                                    acc_mean = get_acc(mean_agg_name)
                                    best_robust = -np.inf
                                    found_robust = False
                                    for agg in aggregators:
                                        if agg['name'] == mean_agg_name: continue
                                        val = get_acc(agg['name'])
                                        if not np.isnan(val):
                                            found_robust = True
                                            if val > best_robust: best_robust = val
                                    
                                    if not np.isnan(acc_mean) and found_robust:
                                        delta = best_robust - acc_mean
                                        all_points.append((excl, delta))

                    # --- 2. LOGIQUE DE BINNING ADAPTATIF ---
                    # On trie tous les points par exclusivité croissante
                    all_points.sort(key=lambda x: x[0])
                    
                    bin_data = []   # Liste des listes de deltas (Y)
                    bin_labels = [] # Liste des labels (X)
                    
                    if len(all_points) > 0:
                        i = 0
                        total_points = len(all_points)
                        
                        while i < total_points:
                            # Définir la fin du chunk actuel
                            end_idx = i + points_per_bin
                            
                            # Logique de fusion pour les restes :
                            # Si le reste après ce chunk est trop petit (ex: < 3 points),
                            # on inclut tout le reste dans le chunk actuel.
                            # Sinon, on garde le chunk de taille standard.
                            if (total_points - end_idx) < (points_per_bin / 2):
                                end_idx = total_points
                            
                            # Extraction du chunk
                            chunk = all_points[i:end_idx]
                            
                            # Extraction des données pour le plot
                            deltas = [p[1] for p in chunk]
                            excls = [p[0] for p in chunk]
                            
                            bin_data.append(deltas)
                            
                            # Création du label : "Min-Max" exclusivité dans ce chunk
                            if len(excls) > 0:
                                min_e = min(excls)
                                max_e = max(excls)
                                # Si c'est quasiment le même point (arrondi), on met juste un chiffre
                                if abs(max_e - min_e) < 0.01:
                                    bin_labels.append(f"{min_e:.2f}")
                                else:
                                    bin_labels.append(f"{min_e:.2f}-{max_e:.2f}")
                            else:
                                bin_labels.append("N/A")

                            # Avancer l'index
                            i = end_idx

                    # --- PLOTTING ---
                    if bin_data:
                        ax.boxplot(bin_data, labels=bin_labels, patch_artist=True)
                    
                    # Esthétique
                    ax.set_title(attack_name)
                    ax.set_xlabel("Intervalle d'Exclusivité")
                    ax.set_ylabel("Gain (Max Robuste - Moyenne)")
                    ax.grid(True, linestyle='--', alpha=0.5)
                    ax.axhline(0, color='black', linewidth=1, linestyle='--')
                    
                    # Rotation des labels X car "0.12-0.18" prend de la place
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
                                                                    
                plt.tight_layout()

                # Construction du nom de fichier
                plot_name = (
                    f"boxplot_adaptive_bins_"
                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                    f"{pre_agg_names}_lr_{lr}"
                )
                
                final_path = os.path.join(path_to_plot, plot_name + '_plot.pdf')
                plt.savefig(final_path)
                plt.close()
                print(f"Plot sauvé : {final_path}")

# Fonctions auxiliaires
def ensure_list(x):
    return x if isinstance(x, list) else [x]

def custom_dict_to_str(d):
    return str(d)