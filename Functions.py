import numpy as np
from scipy.stats import norm

class aq_func:
    def aquisition_function_EI(x, gp_model, y_best):
        y_pred, y_std = gp_model.predict(x.reshape(1, -1), return_std = True)
        epsilon = 1e-9
        z = (y_pred - y_best) / (y_std + epsilon)
        ei = (y_pred - y_best) * norm.cdf(z) + y_std * norm.pdf(z)
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