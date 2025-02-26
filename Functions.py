import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class aq_func:
    def aquisition_function_EI(x, gp_model, y_best, n):
        x = x.reshape(-1, n)
        y_pred, y_std = gp_model.predict(x, return_std = True)
        y_std = np.maximum(y_std,1e-10)
        z = (y_best - y_pred) / y_std
        ei = (y_best - y_pred) * norm.cdf(z) + y_std * norm.pdf(z)
        return ei
    
class normalization:
    def standardize(y_vals):
        y_mean = np.mean(y_vals, axis=0)
        y_std = np.std(y_vals, axis=0)

        return (y_vals - y_mean) / y_std
    
    def reverse_standardize(y_normalized, y_mean, y_std):
        return y_normalized * y_std + y_mean
    
class sort:
    def sort_points_by_x(points):
        return points[np.argsort(points[:,0])]
    
class ZDT2:
    def __init__(x):
        x = np.asarray(x)
        f1 = x[0]  
        g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)  
        f2 = g * (1 - (f1 / g) ** 2)  

        return f1, f2

class ZDT1:
    def zdt1(x):
        x = np.atleast_2d(x)
        f1 = np.asarray(x[:, 0], dtype=np.float64)  
        g = np.asarray(1 + 9 * np.sum(x[:, 1:], axis=1)/ (x.shape[1] -1), dtype=np.float64)  
        h = 1 - np.sqrt(np.asarray(f1 / g))
        f2 = g * h
        return np.column_stack((f1, f2))
    

class plots:
    def plot_zdt1(y1, y2, save_fig):
        f1_pareto = np.linspace(0,1,100)
        f2_pareto = 1 - np.sqrt(f1_pareto)

        plt.figure(figsize=(12, 6))
        plt.scatter(y1, y2, c='blue', label = 'Solution Bayesian')
        for i, label in enumerate(y1):
            plt.text(y1[i], y2[i], f"#{i}", fontsize = 9, ha ='right', color = 'black')
        plt.plot(f1_pareto, f2_pareto, 'r-', linewidth = 1.5, label = 'True Function')
        plt.title('Objective Space')
        plt.xlabel('y1')
        plt.ylabel('y2')
        plt.grid(False)
        plt.savefig(f"Plots/{save_fig}.png")
