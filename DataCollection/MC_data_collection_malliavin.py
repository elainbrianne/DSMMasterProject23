# MP MC Code
from tabulate import tabulate
from MC_Engines.MC_Heston import Heston_Engine
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools import Types
from Tools import RNG
from prettytable import PrettyTable
from time import time
import numpy as np

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
#num_simulations = 3

#Spot price parameters
lower=80
upper=120
num = 100
spots = np.linspace(lower, upper, num).reshape((-1, 1))

# Arrays to store results
option_prices = np.zeros(num)
standard_errors = np.zeros(num)
deltas = np.zeros(num)
gammas = np.zeros(num)
asset_prices = np.zeros(num)

start_time = time()
# Run the simulation
for i in range(num):
    spot_price = spots[i][0]  # Extract the spot price from the array
    
    # Compute price using MC
    map_heston_output = Heston_Engine.get_path_multi_step(0.0, T, parameters, spot_price, v0, no_paths,
                                                          no_time_steps,
                                                          Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                          rnd_generator)
    result = european_option.get_price(map_heston_output[Types.HESTON_OUTPUT.PATHS])
    
   # Compute price, delta and gamma by MC and Malliavin in Heston model
    malliavin_delta = european_option.get_malliavin_delta(map_heston_output[Types.HESTON_OUTPUT.PATHS],
                                                      map_heston_output[Types.HESTON_OUTPUT.DELTA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL])
    malliavin_gamma = european_option.get_malliavin_gamma(map_heston_output[Types.HESTON_OUTPUT.PATHS],
                                                      map_heston_output[Types.HESTON_OUTPUT.GAMMA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL])
    
    # Store results in arrays
    option_prices[i] = result[0]
    standard_errors[i] = result[1]
    deltas[i] = malliavin_delta[0]
    gammas[i] = malliavin_gamma[0]
    asset_prices[i] = np.mean([sub_array[-1] for sub_array in map_heston_output[Types.HESTON_OUTPUT.PATHS]]) # average of last value for 200,000 simulations of asset prices

# Print the collected results
print("MC (Prices) and Malliavin method (Greeks).")
print("Options Prices:\n", option_prices)
print("Standard Errors:\n", standard_errors)
print("Deltas:\n", deltas)
print("Gammas:\n", gammas)
print("Asset Prices in First Time Period:\n", asset_prices)


end_time = time()
execution_time = end_time - start_time
print("Execution Time:", execution_time, "seconds")

# Save results to CSV
np.savetxt("drive/MyDrive/malliavin_option_prices.csv", option_prices, delimiter=",")
np.savetxt("drive/MyDrive/malliavin_standard_errors.csv", standard_errors, delimiter=",")
np.savetxt("drive/MyDrive/malliavin_deltas.csv", deltas, delimiter=",")
np.savetxt("drive/MyDrive/malliavin_gammas.csv", gammas, delimiter=",")
np.savetxt("drive/MyDrive/malliavin_asset_prices.csv", asset_prices, delimiter=",")

print("Results saved to CSV files.")

