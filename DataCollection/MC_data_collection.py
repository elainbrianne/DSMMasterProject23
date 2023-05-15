# Code to run MC multiple times, saves in arrays and in separate CSVs. 
# Change number of simulations with num_simulations. 

from tabulate import tabulate
from MC_Engines.MC_Heston import Heston_Engine
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools import Types
from Tools import RNG
from prettytable import PrettyTable
from time import time
import numpy as np
import csv

# Initializations
epsilon = 1.1
k = 0.5
rho = -0.9
v0 = 0.05
theta = 0.05

f0 = 100
T = 2.0

seed = 123456789

delta = 1.0 / 32.0
no_time_steps = int(T / delta)
no_paths = 100000
strike = 120.0

# Random Generator
rnd_generator = RNG.RndGenerator(seed)

# Vector of parameters
parameters = [k, theta, epsilon, rho]

notional = 1.0

# European option
european_option = EuropeanOption(strike, notional, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T)

# Vector of option price parameters
parameters_option_price = [0.0, theta, rho, k, epsilon, v0, 0.0]

# Simulation parameters
num_simulations = 3

# Arrays to store results
option_prices = np.zeros(num_simulations)
standard_errors = np.zeros(num_simulations)
deltas = np.zeros(num_simulations)
gammas = np.zeros(num_simulations)
asset_prices = np.zeros(num_simulations)

# Run the simulation
start_time = time()
for i in range(num_simulations):
    # Compute price using MC
    map_heston_output = Heston_Engine.get_path_multi_step(0.0, T, parameters, f0, v0, no_paths,
                                                          no_time_steps,
                                                          Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                          rnd_generator)
    result = european_option.get_price(map_heston_output[Types.HESTON_OUTPUT.PATHS])
    
    # Compute delta and gamma by bumping in Heston model (finite differences)
    delta_shift = 0.0001
    f0_shift = f0 * (1.0 + delta_shift)
    f0_shift_left = f0 * (1.0 - delta_shift)
    
    rnd_generator.set_seed(seed)
    
    # Calculating with first underlying asset price
    map_heston_output_shift = Heston_Engine.get_path_multi_step(0.0, T, parameters, f0_shift, v0,
                                                                no_paths, no_time_steps,
                                                                Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                                rnd_generator)
    
    result_shift = european_option.get_price(map_heston_output_shift[Types.HESTON_OUTPUT.PATHS])
    heston_delta_fd = (result_shift[0] - result[0]) / (delta_shift * f0)
    
    # Second asset price price
    rnd_generator.set_seed(seed)
    map_heston_output_shift_left = Heston_Engine.get_path_multi_step(0.0, T, parameters, f0_shift_left, v0,
                                                                     no_paths, no_time_steps,
                                                                     Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                                     rnd_generator)
    
    result_shift_left = european_option.get_price(map_heston_output_shift_left[Types.HESTON_OUTPUT.PATHS])

    # Computing option price using finite differences 
    heston_gamma_fd = (result_shift[0] - 2 * result[0] + result_shift_left[0]) / (delta_shift * f0)**2
    
    # Store results in arrays
    option_prices[i] = result[0]
    standard_errors[i] = result[1]
    deltas[i] = heston_delta_fd
    gammas[i] = heston_gamma_fd
    asset_prices[i] = np.mean([sub_array[-1] for sub_array in map_heston_output[Types.HESTON_OUTPUT.PATHS]]) # average of last value for 200,000 simulations of asset prices

# Print the collected results
print("Options Prices:\n", option_prices)
print("Standard Errors:\n", standard_errors)
print("Deltas:\n", deltas)
print("Gammas:\n", gammas)
print("Asset Prices in First Time Period:\n", asset_prices)

end_time = time()
execution_time = end_time - start_time
print("Execution Time:", execution_time, "seconds")

# Save results to CSV
np.savetxt("drive/MyDrive/option_prices.csv", option_prices, delimiter=",")
np.savetxt("drive/MyDrive/standard_errors.csv", standard_errors, delimiter=",")
np.savetxt("drive/MyDrive/deltas.csv", deltas, delimiter=",")
np.savetxt("drive/MyDrive/gammas.csv", gammas, delimiter=",")
np.savetxt("drive/MyDrive/asset_prices.csv", asset_prices, delimiter=",")

print("Results saved to CSV files.")




