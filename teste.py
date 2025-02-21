import Bayesian
ego_velocity_min = 0
ego_velocity_max = 1

target_velocity_min = 0
target_velocity_max = 1

Bayesian_code = Bayesian.Bayesian(10, [ego_velocity_min, target_velocity_min], [ego_velocity_max, target_velocity_max], 10)

pareto_points_bayes = Bayesian_code.pareto_front
