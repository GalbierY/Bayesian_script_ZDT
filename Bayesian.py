import pygad as pg
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.preprocessing import StandardScaler
from pyDOE import lhs
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import Functions
from nds import ndomsort

class Bayesian:
    def initial_sets(self):
        self.n_samples = self.number_samples

        self.lhs_samples = lhs(2, samples = self.n_samples)

        
        self.lhs_samples = self.lhs_samples * (np.array(self.x_maximum) - np.array(self.x_minimum)) + np.array(self.x_minimum)
        
        self.lhs_samples = np.array(self.lhs_samples)
        print(self.lhs_samples)
        
        self.y_vals = Functions.ZDT1.zdt1(self.lhs_samples)

        weights = np.ones(self.y_vals.shape[1]) / self.y_vals.shape[1]

        self.Y = np.dot(self.y_vals, weights)

        kernel = 1.0*Matern(length_scale=1.0, nu=1.5)

        self.gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, optimizer='fmin_l_bfgs_b', normalize_y=True)

        print('INITIALSETS\nlhs samples: ', self.lhs_samples)
        print('y_vals: ', self.y_vals)
        print('Y: ', self.Y)

        return self.gp_model, self.lhs_samples,self.y_vals, self.Y, weights

    
    def fitness(self, ga_instance, solution, solution_id):
        fitness = Functions.aq_func.aquisition_function_EI(solution, self.gp_model, np.min(self.Y), 2)
        
        return fitness.item()

    def run_ga(self): 
        print('running GA...')

        num_generations = 10
        num_genes = 2
        sol_per_pop = 10
        num_parents_mating = round(sol_per_pop/2)
        
        gene_space = []
        for x_min, x_max in zip(self.x_minimum, self.x_maximum):
            bound = {'low': x_min, 'high': x_max}
            gene_space.append(bound)

        ga_instance = pg.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=self.fitness,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        gene_space=gene_space,
                        gene_type=[float,float],
                        parent_selection_type="tournament",
                        save_solutions=True,
                        keep_parents =1,
                        save_best_solutions=True,
                        mutation_type="adaptive",
                        mutation_percent_genes=[0.1, 0.3],
                        stop_criteria='saturate_100',
                        allow_duplicate_genes=True)
        
        ga_instance.run()

        return ga_instance


    def __init__(self, n_iteration, minimus, maximuns, number_samples):
        self.number_samples = int(number_samples)
        self.x_maximum = [float(m) for m in maximuns]
        self.x_minimum = [float(n) for n in minimus]
        self.n_iterations = int(n_iteration)

        self.gp_model, self.lhs_samples,self.y_vals, self.Y, weights = self.initial_sets()

        all_y1, all_y2 = [], []

        for i in range(self.n_iterations):
            self.gp_model.fit(self.lhs_samples, self.Y)

            self.GA_instance = self.run_ga()
            self.x_best, self.fitness_best, self.index_best = self.GA_instance.best_solution()
            self.x_best = np.array(self.x_best).reshape(1, -1)

            self.y_next = Functions.ZDT1.zdt1(self.x_best)
            new_Y = np.dot(self.y_next, weights)
            
            self.lhs_samples = np.vstack((self.lhs_samples, self.x_best))
            self.y_vals = np.vstack((self.y_vals, self.y_next))
            self.Y = np.append(self.Y, new_Y)

            all_y1.append(self.y_next[0][0])
            all_y2.append(self.y_next[0][1])

            Functions.plots.plot_zdt1(all_y1, all_y2, f'Objective_space_{i+1}')
        
        print('lhs samples: ', self.lhs_samples)
        print('y_vals: ', self.y_vals)

        self.y_vals_for_pf = np.array([[x[0], x[1]] for x in self.y_vals])

        fronts = ndomsort.non_domin_sort(self.y_vals_for_pf, only_front_indices=True)

        self.pareto_front_indices = [i for i, f in enumerate(fronts) if f == 0]

        self.pareto_front = self.y_vals_for_pf[self.pareto_front_indices]

        print('\nPareto front: ', self.pareto_front)

        self.final1_y_vals = np.array([result[0] for result in self.y_vals_for_pf])
        self.final2_y_vals = np.array([result[1] for result in self.y_vals_for_pf])

        self.final1_lhs_samples = np.array(result[0] for result in self.lhs_samples)
        self.final2_lhs_samples = np.array(result[1] for result in self.lhs_samples)