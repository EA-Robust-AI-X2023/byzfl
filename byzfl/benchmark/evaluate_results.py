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
from scipy.stats import linregress


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
                                fig.suptitle(f"Accuracy paths, distribution param : {str(dist_parameter)}", fontsize=14)

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
                                        # save max and min accuracy across steps for each attack, to plot shaded area between them
                                        minmax = np.zeros((len(attacks), nb_data_distribution_seeds*nb_training_seeds, 2))
                                        for i in range(len(err)):
                                            for j in range(nb_data_distribution_seeds*nb_training_seeds):
                                                minmax[i,j,0] = np.min(tab_acc[i,j,:])
                                                minmax[i,j,1] = np.max(tab_acc[i,j,:])
                                                
                                    plt.rcParams.update({'font.size': 12})

                                    
                                    for i, attack in enumerate(attacks):
                                        attack = attack["name"]
                                        ax = axes[i] if isinstance(axes, np.ndarray) else axes
                                        ax.plot(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0), label = agg["name"], color = colors[i_agg], linestyle = tab_sign[i_agg], marker = markers[i_agg], markevery = 1000)
                                        
                                        if plot_std:
                                            ax.fill_between(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0) - err[i], np.mean(tab_acc[i], axis = 0) + err[i], alpha = 0.25)
                                            # ax.fill_between(np.arange(nb_accuracies)*evaluation_delta, err[i,:,0], err[i,:,1], alpha = 0.25, color = colors[i_agg])                                         
                                        ax.set_title(f"{attack} attack", fontsize=10)
                                        ax.set_xlim(0,(nb_accuracies-1)*evaluation_delta)
                                        ax.grid()
                                        ax.legend()
                                        ax.set_xlabel('Round')
                                        ax.set_ylabel('Accuracy')
                                        ax.set_ylim(min_accuracy,1)
                                        ax.set_xlim(0,(nb_accuracies-1)*evaluation_delta)
                                        
                                        #also print in the terminal min and max accuracy across seeds for each attack
                                        if plot_std:
                                            for i in range(len(attacks)):
                                                print(f"Distribution: {custom_dict_to_str(data_dist['name'])}, Aggregator : {agg['name']}, Attack: {attacks[i]['name']}, Min accuracy across seeds: {np.mean(minmax[i,:,0])}, Max accuracy across seeds: {np.mean(minmax[i,:,1])}\n\n")


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
                                    attack_name = attack['name']
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
                                                f"{pre_agg_names}_{custom_dict_to_str(attack_name)}_"
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
                                
                                err_A = np.zeros((len(attacks), nb_scatterings))
                                err_ksi = np.zeros((len(attacks), nb_scatterings))

                                for i_attack in range(len(attacks)):
                                    err_ksi[i_attack] = (1.96*np.std(tab_scat_ksi[i_attack], axis=0))/math.sqrt(nb_training_seeds*nb_data_distribution_seeds)
                                    err_A[i_attack] = (1.96*np.std(tab_scat_A[i_attack], axis=0))/math.sqrt(nb_training_seeds*nb_data_distribution_seeds)

                                plt.rcParams.update({'font.size': 12})
                                
                                #i_ksi is the index of the attack with the highest maximal scattering value for ksi. Hence we take the argmax:
                                i_ksi = np.argmax(tab_scat_ksi.mean(axis=1).max(axis=1))

                                gradient_type = "Momentum" if scatter_momentums else "Raw"
                                plt.plot(np.linspace(0, (nb_scatterings-1)*evaluation_delta, nb_scatterings), np.mean(tab_scat_ksi[i_ksi], axis = 0), label = r"$ksi$", color = colors[0], linestyle = tab_sign[0], marker = None, markevery = 1)
                                # plt.fill_between(np.linspace(0, (nb_scatterings-1)*evaluation_delta, nb_scatterings), np.mean(tab_scat_ksi[i_ksi], axis = 0) - err_ksi, np.mean(tab_scat_ksi[i_ksi], axis = 0) + err_ksi, alpha = 0.25)

                                for i_attack, attack in enumerate(attacks):
                                    attack_name = attack['name']
                                    plt.plot(np.linspace(0, (nb_scatterings-1)*evaluation_delta, nb_scatterings), np.mean(tab_scat_A[i_attack], axis = 0), label = f"A - {attack_name}", color = colors[i_attack+1], linestyle = tab_sign[1], marker = None, markevery = 1)
                                    # plt.fill_between(np.linspace(0, (nb_scatterings-1)*evaluation_delta, nb_scatterings), np.mean(tab_scat_A[i_attack], axis = 0) - err_A[i_attack], np.mean(tab_scat_A[i_attack], axis = 0) + err_A[i_attack], alpha = 0.25)

                                plt.xlim(0,(nb_scatterings-1)*evaluation_delta)
                                plt.xlabel('Round')
                                plt.ylabel("Gradient heterogeneity")
                                plt.title(f"{gradient_type} gradient scatterings, {data_dist['name']}-{str(dist_parameter)} distribution")
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

  
def compute_entropy(distrib: np.array):
    "returns the entropy of the classes held by each client"
    distrib = distrib + 1e-10 # to avoid log(0)
    distrib = distrib / distrib.sum(axis=1, keepdims=True) # normalize to get probabilities
    entropy = -np.sum(distrib * np.log(distrib), axis=1)
    return entropy
                        

def evaluate_impact_exclusivity_adaptive_bins(path_to_results, path_to_plot,exclusivity_computation = compute_entropy, plot_regression=False):
    """
    Trace des boxplots de (Best_Robust_Acc - Mean_Acc).
    Les bins sont adaptatives : chaque bin contient environ 6 points de données,
    triés par mesure d'exclusivité croissante.
    
    exclusivity_computation is a function computing a measure of exclusivity from the label distribution amongst clients
    """
    
    # --- CONFIGURATION ---
    mean_agg_name = "Average"
    points_per_bin = 10  # Nombre cible de points par boxplot
    
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
    
    if exclusivity_computation == compute_entropy:
        xlabel = "entropy"
    else:
        xlabel="exclusivity"

    def _compute_exclusivity_value(distrib, nb_honest, nb_nodes):
        try:
            raw_value = exclusivity_computation(distrib)
        except TypeError:
            raw_value = exclusivity_computation(distrib, nb_honest, nb_nodes)

        if np.isscalar(raw_value):
            return float(raw_value)

        try:
            values = np.array(raw_value, dtype=float).reshape(-1)
        except Exception:
            return np.nan

        byz_client_index = int(nb_honest)
        if byz_client_index < 0 or byz_client_index >= len(values):
            return np.nan
        return float(values[byz_client_index])
    
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
                
                fig.suptitle(f"Advantage of robustness (Bins of {points_per_bin} seeds) - n={nb_nodes}, f={nb_byzantine}", fontsize=14)

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
                                    distrib = np.array(genfromtxt(exclusivity_path, delimiter=','))

                                    excl = _compute_exclusivity_value(distrib, nb_honest, nb_nodes)

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
                                            # return genfromtxt(p, delimiter=',')[-1] # accuracy finale
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
                            
                    #fusionner les bins qui ont le meme label
                    merged_bin_data = []
                    merged_bin_labels = []
                    for lbl, data in zip(bin_labels, bin_data):
                        if lbl in merged_bin_labels:
                            idx = merged_bin_labels.index(lbl)
                            merged_bin_data[idx].extend(data)
                        else:
                            merged_bin_labels.append(lbl)
                            merged_bin_data.append(data)
                    # ajouter nombre de points dans le label

                    merged_bin_labels = [f"{lbl} ({len(data)})" for lbl, data in zip(merged_bin_labels, merged_bin_data)]
                            

                    # --- PLOTTING ---
                    if merged_bin_data:
                        ax.boxplot(merged_bin_data, tick_labels=merged_bin_labels, patch_artist=True)
                    
                    if plot_regression:
                        #plot the linear regression of the delta in accuracy on the exclusivity
                        all_excls = [p[0] for p in all_points]
                        all_deltas = [p[1] for p in all_points]
                        if len(all_excls) >=2:
                            slope, intercept, r_value, p_value, std_err = linregress(all_excls, all_deltas)
                            x_vals = np.array([min(all_excls), max(all_excls)])
                            y_vals = intercept + slope * x_vals
                            #rendre l'axe des abscisses compatible avec les boxplots
                            if merged_bin_labels:
                                x_ticks = []
                                for lbl in merged_bin_labels:
                                    if '-' in lbl:
                                        parts = lbl.split('-')
                                        mid = (float(parts[0]) + float(parts[1])) / 2
                                        x_ticks.append(mid)
                                    else:
                                        x_ticks.append(float(lbl))
                                ax.set_xticks(range(1, len(x_ticks)+1), labels=merged_bin_labels)
                                x_vals_transformed = np.interp(x_vals, x_ticks, range(1, len(x_ticks)+1))
                            else :
                                x_vals_transformed = x_vals
                            ax.plot(x_vals_transformed, y_vals, color='orange', linestyle='--', label=f"Linear regression,p={p_value:.3f}")
                            ax.legend()
                    
                    # Esthétique
                    ax.set_title(attack_name)
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel("Gain (Max Robuste - Mean)")
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
                
                final_path = os.path.join(path_to_plot, plot_name + f'_{xlabel}'+'_plot.pdf')
                plt.savefig(final_path)
                plt.close()
                print(f"Plot sauvé : {final_path}")

# Fonctions auxiliaires
def ensure_list(x):
    return x if isinstance(x, list) else [x]

def custom_dict_to_str(d):
    return str(d)



def evaluate_per_class_accuracies(path_to_results, path_to_plot):
    """
    Trace des boxplots de l'accuracy par classe en fonction d'une mesure d'exclusivité, d'entropie ou du paramètre de la distribution de Dirichlet.
    Les classes "source" et "target" sont définies respectivement comme la classe majoritaire dans le worker byzantin et la classe vers laquelle il flip (9-i).
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
        return

    path_to_hyperparameters = os.path.join(path_to_results, "best_hyperparameters", "hyperparameters")

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = ensure_list(data["benchmark_config"]["nb_honest_clients"])
    nb_byz = ensure_list(data["benchmark_config"]["f"])
    nb_declared = ensure_list(data["benchmark_config"].get("tolerated_f", None))
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = ensure_list(data["benchmark_config"]["data_distribution"])
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    nb_labels = data["model"]["nb_labels"]
    lr_list = ensure_list(data["model"]["learning_rate"])

    # <-------------- Honest Nodes Config ------------->
    momentum_list = ensure_list(data["honest_clients"]["momentum"])
    wd_list = ensure_list(data["honest_clients"]["weight_decay"])

    # <-------------- Aggregators Config ------------->
    aggregators = ensure_list(data["aggregator"])
    pre_aggregators = data["pre_aggregators"]
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    # <-------------- Attacks Config ------------->
    attacks = ensure_list(data["attack"])

    def _safe_read_array(path):
        if not os.path.exists(path):
            return None
        try:
            array = np.array(genfromtxt(path, delimiter=','), dtype=float)
        except Exception:
            return None
        if array.size == 0:
            return None
        return array

    def _extract_best_per_class_accuracy(per_class_values, nb_classes):
        array = np.array(per_class_values, dtype=float)
        if array.ndim == 1:
            if array.shape[0] < nb_classes:
                return None
            return array[:nb_classes]

        if array.ndim != 2:
            return None

        if array.shape[1] != nb_classes:
            if array.shape[0] == nb_classes:
                array = array.T
            else:
                return None

        row_scores = np.nanmean(array, axis=1)
        if np.all(np.isnan(row_scores)):
            return None

        best_step = int(np.nanargmax(row_scores))
        
        #rather than taking best step, take the last step for which all values are not 0
        # last_valid_step = -3
        # for step in range(array.shape[0]):
        #     if np.all(array[step] > 0):
        #         last_valid_step = step
        #     else:
        #         break
        # best_step = last_valid_step if last_valid_step >= 0 else 0
        return array[best_step]

    def _resolve_distribution_parameter_values(dist_name, dist_parameter):
        dist_name_str = str(dist_name).strip().lower()
        if dist_name_str == "extreme_niid_modified":
            return "None", 0.0

        filename_value = dist_parameter
        try:
            plot_value = float(dist_parameter)
        except (TypeError, ValueError):
            plot_value = np.nan
        return filename_value, plot_value

    def _build_bins(x_values, mode, max_bins=10):
        finite_values = [float(value) for value in x_values if np.isfinite(value)]
        if len(finite_values) == 0:
            return []

        if mode == "distribution_parameter":
            unique_values = sorted(set(finite_values))
            return [{"label": f"{value:g}", "min": value, "max": value} for value in unique_values]

        sorted_values = sorted(finite_values)
        bins = []
        points_per_bin = max(1, int(math.ceil(len(sorted_values) / float(max_bins))))
        idx = 0
        total = len(sorted_values)
        while idx < total:
            end_idx = idx + points_per_bin
            if (total - end_idx) < (points_per_bin / 2):
                end_idx = total
            chunk = sorted_values[idx:end_idx]
            min_x = float(np.min(chunk))
            max_x = float(np.max(chunk))
            if abs(max_x - min_x) < 1e-12:
                label = f"{min_x:.3f}"
            else:
                label = f"{min_x:.3f}-{max_x:.3f}"
            bins.append({"label": label, "min": min_x, "max": max_x})
            idx = end_idx
        return bins

    def _assign_bin(value, bins, mode):
        if not np.isfinite(value) or len(bins) == 0:
            return None
        value = float(value)

        if mode == "distribution_parameter":
            for bin_index, one_bin in enumerate(bins):
                if np.isclose(value, one_bin["min"], atol=1e-12, rtol=0.0):
                    return bin_index
            return None

        for bin_index, one_bin in enumerate(bins):
            if bin_index == len(bins) - 1:
                if one_bin["min"] <= value <= one_bin["max"]:
                    return bin_index
            else:
                if one_bin["min"] <= value < one_bin["max"]:
                    return bin_index
        return None

    def _sanitize_filename_component(value):
        safe_value = str(value).replace(" ", "_")
        safe_value = safe_value.replace(os.sep, "-")
        if os.altsep:
            safe_value = safe_value.replace(os.altsep, "-")
        return safe_value

    def _plot_cross_correlation_per_aggregator(points, agg_name, title, output_path):
        class_keys = ["source", "target", "other"]
        class_labels = ["source", "target", "other"]

        matrix_values = []
        for point in points:
            values = [point.get(class_key, np.nan) for class_key in class_keys]
            if all(np.isfinite(value) for value in values):
                matrix_values.append(values)

        if len(matrix_values) < 2:
            logger.warning(f"Not enough points to compute cross correlation for {agg_name} at {output_path}")
            return

        values_array = np.array(matrix_values, dtype=float)
        corr_matrix = np.corrcoef(values_array, rowvar=False)
        if corr_matrix.shape != (len(class_keys), len(class_keys)):
            logger.warning(f"Invalid correlation matrix shape for {agg_name} at {output_path}")
            return

        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        image = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0)

        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(class_labels)))
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)

        for row_id in range(corr_matrix.shape[0]):
            for col_id in range(corr_matrix.shape[1]):
                value = corr_matrix[row_id, col_id]
                text_value = "nan" if not np.isfinite(value) else f"{value:.2f}"
                ax.text(col_id, row_id, text_value, ha='center', va='center', color='black')

        if title:
            ax.set_title(f"{title} | {agg_name}")
        else:
            ax.set_title(f"Cross correlation | {agg_name}")

        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Pearson correlation")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _plot_per_attack_class_aggregators_binned(aggregator_points, bins, mode, x_key, class_key, class_label, xlabel, title, output_path):
        from matplotlib.lines import Line2D

        ordered_aggregators = [agg_name for agg_name in aggregator_points.keys()]
        if len(ordered_aggregators) == 0 or len(bins) == 0:
            logger.warning(f"No data to plot for {output_path}")
            return

        grouped_values = {
            agg_name: [[] for _ in bins]
            for agg_name in ordered_aggregators
        }

        for agg_name in ordered_aggregators:
            points = aggregator_points[agg_name]
            for point in points:
                bin_index = _assign_bin(point[x_key], bins, mode)
                if bin_index is None:
                    continue
                class_value = point[class_key]
                if np.isfinite(class_value):
                    grouped_values[agg_name][bin_index].append(float(class_value))

        non_empty_aggregators = []
        for agg_name in ordered_aggregators:
            has_values = any(len(bin_values) > 0 for bin_values in grouped_values[agg_name])
            if has_values:
                non_empty_aggregators.append(agg_name)

        if len(non_empty_aggregators) == 0:
            logger.warning(f"No data to plot for {output_path}")
            return

        ordered_aggregators = non_empty_aggregators

        fig_width = max(9, 1.5 * len(bins))
        fig, ax = plt.subplots(figsize=(fig_width, 5.5))

        bin_positions = np.arange(len(bins), dtype=float)
        nb_aggregators = len(ordered_aggregators)
        group_width = 0.8
        one_box_width = group_width / max(1, nb_aggregators)

        legend_handles = []
        for agg_index, agg_name in enumerate(ordered_aggregators):
            offset = (agg_index - (nb_aggregators - 1) / 2.0) * one_box_width
            positions = bin_positions + offset
            values_for_plot = [
                bin_values if len(bin_values) > 0 else [np.nan]
                for bin_values in grouped_values[agg_name]
            ]

            boxplot = ax.boxplot(
                values_for_plot,
                positions=positions,
                widths=one_box_width * 0.9,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False
            )

            color = colors[agg_index % len(colors)]
            for box in boxplot['boxes']:
                box.set(facecolor=color, alpha=0.75)
            for median in boxplot['medians']:
                median.set(color='black', linewidth=1.2)

            legend_handles.append(
                Line2D([0], [0], color=color, lw=6, label=agg_name)
            )

        ax.set_xticks(bin_positions)
        ax.set_xticklabels([one_bin["label"] for one_bin in bins], rotation=35, ha='right')
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Per-class test accuracy")
        #set inferior limite of y axis to the minimal accuracy observed in the data, and superior limit to 1.0
        y_inf_lim = min(
            float(np.nanmin(values_for_plot))
            for agg_name in ordered_aggregators
            for values_for_plot in [grouped_values[agg_name][bin_index] for bin_index in range(len(bins))]
        )
        ax.set_ylim(max(0.0, y_inf_lim - 0.05), 1.0)
        if title:
            ax.set_title(f"{title} | class={class_label}")
        else:
            ax.set_title(f"class={class_label}")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(handles=legend_handles, loc='best')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    measure_settings = [
        {
            "mode": "exclusivity",
            "x_key": "x_exclusivity",
            "xlabel": "Exclusivity (byzantine worker)"
        },
        {
            "mode": "entropy",
            "x_key": "x_entropy",
            "xlabel": "Entropy (byzantine worker)"
        },
        {
            "mode": "distribution_parameter",
            "x_key": "x_distribution_parameter",
            "xlabel": "Dirichlet parameter"
        }
    ]

    for nb_honest in nb_honest_clients:
        for nb_byzantine in nb_byz:
            if nb_declared[0] is None:
                nb_declared_list = [nb_byzantine]
            else:
                nb_declared_list = [item for item in nb_declared if item >= nb_byzantine]

            for nb_decl in nb_declared_list:
                if set_honest_clients_as_clients:
                    nb_nodes = nb_honest
                else:
                    nb_nodes = nb_honest + nb_byzantine

                for pre_agg in pre_aggregators:
                    pre_agg_names = "_".join([one_pre_agg['name'] for one_pre_agg in pre_agg])

                    for attack in attacks:
                        attack_name = custom_dict_to_str(attack['name'])

                        aggregator_points = {
                            custom_dict_to_str(agg['name']): []
                            for agg in aggregators
                        }

                        for agg in aggregators:
                            agg_name = custom_dict_to_str(agg['name'])

                            for data_dist in data_distributions:
                                dist_name = custom_dict_to_str(data_dist['name'])
                                dist_parameter_list = ensure_list(data_dist.get("distribution_parameter", None))

                                for dist_parameter in dist_parameter_list:
                                    dist_parameter_for_filename, dirichlet_value = _resolve_distribution_parameter_values(
                                        dist_name,
                                        dist_parameter
                                    )

                                    hyper_file_name = (
                                        f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                        f"{dist_name}_{dist_parameter_for_filename}_{pre_agg_names}_{agg_name}.txt"
                                    )
                                    hyper_path = os.path.join(path_to_hyperparameters, hyper_file_name)

                                    if os.path.exists(hyper_path):
                                        hyperparameters = np.loadtxt(hyper_path)
                                        lr = float(hyperparameters[0])
                                        momentum = float(hyperparameters[1])
                                        wd = float(hyperparameters[2])
                                    else:
                                        lr = lr_list[0]
                                        momentum = momentum_list[0]
                                        wd = wd_list[0]

                                    experiment_name = (
                                        f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                        f"{dist_name}_{dist_parameter_for_filename}_{agg_name}_{pre_agg_names}_{attack_name}_"
                                        f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                    )
                                    experiment_path = os.path.join(path_to_results, experiment_name)

                                    for run_dd in range(nb_data_distribution_seeds):
                                        dd_seed = run_dd + data_distribution_seed
                                        distribution_path = os.path.join(
                                            experiment_path,
                                            f"distributions/worker_distributions_dd_seed_{dd_seed}.txt"
                                        )

                                        partitions = _safe_read_array(distribution_path)
                                        if partitions is None:
                                            continue

                                        if partitions.ndim == 1:
                                            partitions = np.expand_dims(partitions, axis=0)

                                        byz_client_index = nb_honest
                                        if byz_client_index >= partitions.shape[0]:
                                            continue

                                        source_class = int(np.argmax(partitions[byz_client_index]))
                                        target_class = int(nb_labels - 1 - source_class)
                                        target_class = max(0, min(nb_labels - 1, target_class))
                                        other_classes = [
                                            class_id for class_id in range(nb_labels)
                                            if class_id not in [source_class, target_class]
                                        ]

                                        all_exclusivities = compute_exclusivity(partitions)
                                        all_entropies = compute_entropy(partitions)
                                        if byz_client_index >= len(all_exclusivities) or byz_client_index >= len(all_entropies):
                                            continue

                                        exclusivity_value = float(all_exclusivities[byz_client_index])
                                        entropy_value = float(all_entropies[byz_client_index])

                                        for run_tr in range(nb_training_seeds):
                                            tr_seed = run_tr + training_seed
                                            per_class_path = os.path.join(
                                                experiment_path,
                                                f"test_accuracy_per_class_tr_seed_{tr_seed}_dd_seed_{dd_seed}.txt"
                                            )

                                            per_class_values = _safe_read_array(per_class_path)
                                            if per_class_values is None:
                                                continue

                                            best_per_class = _extract_best_per_class_accuracy(per_class_values, nb_labels)
                                            if best_per_class is None:
                                                continue

                                            source_acc = float(best_per_class[source_class])
                                            target_acc = float(best_per_class[target_class])
                                            if other_classes:
                                                other_acc = float(np.nanmean(best_per_class[other_classes]))
                                            else:
                                                other_acc = np.nan

                                            aggregator_points[agg_name].append({
                                                "x_exclusivity": exclusivity_value,
                                                "x_entropy": entropy_value,
                                                "x_distribution_parameter": dirichlet_value,
                                                "source": source_acc,
                                                "target": target_acc,
                                                "other": other_acc
                                            })

                        non_empty_aggregators = [
                            agg_name for agg_name, points in aggregator_points.items() if len(points) > 0
                        ]

                        if len(non_empty_aggregators) == 0:
                            logger.warning(
                                f"No per-class accuracy data found for attack={attack_name}, "
                                f"n={nb_nodes}, f={nb_byzantine}, d={nb_decl}, pre_agg={pre_agg_names}"
                            )
                            continue

                        base_plot_name = (
                            f"per_class_accuracy_{dataset_name}_{model_name}_n_{nb_nodes}_"
                            f"f_{nb_byzantine}_d_{nb_decl}_{pre_agg_names}_{attack_name}"
                        )

                        title = ("")
                        #plot cross correlation of source, target and other accuracies for each aggregator
                        for agg_name in non_empty_aggregators:
                            safe_agg_name = _sanitize_filename_component(agg_name)
                            _plot_cross_correlation_per_aggregator(
                                aggregator_points[agg_name],
                                agg_name,
                                title,
                                os.path.join(path_to_plot, base_plot_name + f"_cross_corr_{safe_agg_name}_plot.pdf")
                            )

                        for measure_cfg in measure_settings:
                            measure_mode = measure_cfg["mode"]
                            x_key = measure_cfg["x_key"]
                            xlabel = measure_cfg["xlabel"]

                            all_x_values = []
                            for points in aggregator_points.values():
                                all_x_values.extend([point[x_key] for point in points if np.isfinite(point[x_key])])

                            if len(all_x_values) == 0:
                                logger.warning(
                                    f"No valid {measure_mode} values for attack={attack_name}, n={nb_nodes}, f={nb_byzantine}, d={nb_decl}, pre_agg={pre_agg_names}"
                                )
                                continue

                            bins = _build_bins(all_x_values, measure_mode, max_bins=10)
                            if len(bins) == 0:
                                logger.warning(
                                    f"No valid bins for mode={measure_mode}, attack={attack_name}, n={nb_nodes}, f={nb_byzantine}, d={nb_decl}, pre_agg={pre_agg_names}"
                                )
                                continue

                            _plot_per_attack_class_aggregators_binned(
                                aggregator_points,
                                bins,
                                measure_mode,
                                x_key,
                                class_key="source",
                                class_label="source",
                                xlabel=xlabel,
                                title=title,
                                output_path=os.path.join(path_to_plot, base_plot_name + f"_{measure_mode}_source_plot.pdf")
                            )

                            _plot_per_attack_class_aggregators_binned(
                                aggregator_points,
                                bins,
                                measure_mode,
                                x_key,
                                class_key="target",
                                class_label="target",
                                xlabel=xlabel,
                                title=title,
                                output_path=os.path.join(path_to_plot, base_plot_name + f"_{measure_mode}_target_plot.pdf")
                            )

                            _plot_per_attack_class_aggregators_binned(
                                aggregator_points,
                                bins,
                                measure_mode,
                                x_key,
                                class_key="other",
                                class_label="others",
                                xlabel=xlabel,
                                title=title,
                                output_path=os.path.join(path_to_plot, base_plot_name + f"_{measure_mode}_others_plot.pdf")
                            )
                            
                            
                            
#We add a new method to verify that our per class accuracy plots are working correctly.
#This method should compute overall server test accuracy from the per class accuracies.
#This requires taking into account that all classes are not represented equally in the test set, so we need to weight the per class accuracies by the proportion of each class in the test set.
def _compute_overall_accuracy_from_per_class(per_class_accuracies, test_set_proportions):
    if len(per_class_accuracies) != len(test_set_proportions):
        raise ValueError("Length of per_class_accuracies and test_set_proportions must be the same")
    
    overall_accuracy = 0.0
    for acc, prop in zip(per_class_accuracies, test_set_proportions):
        overall_accuracy += acc * prop
    
    return overall_accuracy
