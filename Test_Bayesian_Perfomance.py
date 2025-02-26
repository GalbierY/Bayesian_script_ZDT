import Bayesian
import pytest
import Functions
import matplotlib.pyplot as plt
import numpy as np

# pytest Test_Bayesian_Perfomance.py -s

@pytest.mark.parametrize('n_iterations', [
    20,
    25,
    30,
])
def test_bayesian(n_iterations):
    
    x1_min = 0
    x1_max = 1

    x2_min = 0
    x2_max = 1


    Bayesian_code = Bayesian.Bayesian(n_iterations, [x1_min, x2_min], [x1_max, x2_max], 10)

    pareto_points_bayes = Bayesian_code.pareto_front


    print(f"{n_iterations} iterations:")
    print('\nPoints on Bayesian: ', pareto_points_bayes)

    ################################################################################################################################################
    # PLOTS
    ################################################################################################################################################

    f1_pareto = np.linspace(0,1,100)
    f2_pareto = 1 - np.sqrt(f1_pareto)

    plt.clf()
    plt.figure(figsize=(12, 6))
    plt.scatter(Bayesian_code.final1_y_vals, Bayesian_code.final2_y_vals, c='blue', label='Solutions Bayesian')
    plt.scatter(Bayesian_code.pareto_front[:, 0], Bayesian_code.pareto_front[:, 1], color='red', label='Pareto Front Bayesian')
    pareto_labels_bayes = [f"#{i+1}" for i in range(len(Bayesian_code.pareto_front_indices))]
    for i, label in enumerate(pareto_labels_bayes):
        plt.text(Bayesian_code.pareto_front[i, 0], Bayesian_code.pareto_front[i, 1], label, fontsize=9, ha='right', color='black')
    plt.plot(f1_pareto, f2_pareto, 'r-', linewidth = 1.5, label = 'True Function')
    plt.title('Pareto Front')
    plt.xlabel('y1')
    plt.ylabel('y2')
    plt.legend()
    plt.grid(False)
    plt.savefig(f'Plots/All_Points_{n_iterations}.png')

    sorted_pareto_bayes = Functions.sort.sort_points_by_x(Bayesian_code.pareto_front)

    plt.clf()
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_pareto_bayes[:,0], sorted_pareto_bayes[:,1],  c='blue', label='Pareto Front Bayesian')
    plt.scatter(Bayesian_code.pareto_front[:, 0], Bayesian_code.pareto_front[:, 1], color='blue', label='Pareto Front Bayesian')
    pareto_labels_bayes = [f"#{i+1}" for i in range(len(Bayesian_code.pareto_front_indices))]
    for i, label in enumerate(pareto_labels_bayes):
        plt.text(Bayesian_code.pareto_front[i, 0], Bayesian_code.pareto_front[i, 1], label, fontsize=9, ha='right', color='black')
    plt.plot(f1_pareto, f2_pareto, 'r-', linewidth = 1.5, label = 'True Function')
    plt.title('Pareto Front')
    plt.xlabel('y1)')
    plt.ylabel('y2')
    plt.legend()
    plt.grid(False)
    plt.savefig(f'Plots/Pareto_Fronts_{n_iterations}.png')

