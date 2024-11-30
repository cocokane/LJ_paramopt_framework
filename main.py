import numpy as np
import pandas as pd
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from deap import creator, base, tools, algorithms
import warnings
# Ignore UserWarning raised by sklearn when predicting without feature names
warnings.filterwarnings("ignore", category=UserWarning)

######################### 
# Set this as required
numberofind = 10 # Number of parameters that shall be finally printed 
#########################
# Linux 
target_file = 'reference/EMS_final_0.csv'  #Training data file
oo_target = pd.read_csv('reference/OO_target.out', sep=" ", header=None, names=["dist", "g(r)"])   #AIMD O-O RDF file
actual_rdf_oo = oo_target['g(r)'].to_numpy()
# print(actual_rdf_oo)
ss_target = pd.read_csv('reference/SS_target.out', sep=" ", header=None, names=["dist", "g(r)"])  #AIMD S-S RDF file
actual_rdf_ss = ss_target['g(r)'].to_numpy()
# print(actual_rdf_ss)
actual_density = 1168.81  # Experimental/AIMD Target density for comparison

def calculate_fitness(black_box_output):
    """
    Calculates the Rx value and density from the black box output and computes the fitness value.
    
    Parameters:
    - black_box_output: A numpy array with 835 elements for OO and SS histogram bins (417 each)
      followed by the density value.
    
    Returns:
    - A tuple containing the density, Rx value, and the fitness value.
    """
    # Validate input length
    if len(black_box_output) != 835:  # 417 OO + 417 SS bins + 1 density
        raise ValueError("The black box output must have 835 elements.")
    
    # Extract OO and SS bins
    pred_density = black_box_output[0]
    ss_bins = black_box_output[1:418]
    oo_bins = black_box_output[418:836]  # Adjust indices for 417 bins

    
    # Calculate OO and SS sum of squares
    oo_sum_of_squares = np.sum(np.square(oo_bins - actual_rdf_oo))
    ss_sum_of_squares = np.sum(np.square(ss_bins - actual_rdf_ss))
    
    # Calculate OO and SS average of squares
    oo_avg_of_squares = oo_sum_of_squares / np.sum(np.square(actual_rdf_oo))
    ss_avg_of_squares = ss_sum_of_squares / np.sum(np.square(actual_rdf_ss))
    
    # Calculate Rx as the square root of the average of OO and SS avg_of_squares
    Rx = np.sqrt((oo_avg_of_squares + ss_avg_of_squares) / 2)
    
    # Compute the deviation from the target density
    denfit = abs(pred_density - actual_density)
    
    # Fitness calculation weights
    weight_den = 0.5
    weight_rx = 500  # Adjust the weights as per your requirement
    
    # Calculate fitness
    fitness = weight_rx * Rx + weight_den * denfit
    
    return  fitness #, pred_density, Rx


# Load the dataset
file_path = target_file
data = pd.read_csv(file_path)

# Correct way to drop columns
data = data.drop(['Error %', 'Mutation EMS'], axis=1) 
data_subset = data.iloc[:, 16:]    #columns defined according to atom types 
data_subset
fitness_values = []
for index, row in data.iterrows():
    # Combine OO, SS, and density values into a single array to pass to the function
    output = np.concatenate([row[16:].values])
    fitness = calculate_fitness(output)
    fitness_values.append(fitness)

# Create the new DataFrame with sigma, epsilon parameters, and the calculated fitness values
data_fitness = pd.DataFrame(data.iloc[:, :16])  # First 16 columns are sigma and epsilon parameters (sigma(1), eps(1), sigma(2), eps(2)..)
data_fitness['Fitness'] = fitness_values  # Append the fitness values as a new column
X = data_fitness.iloc[:, :16].values
# Assuming the fitness function value is stored in the last column
y = data_fitness['Fitness'].values  # Update 'F' if your column name is different
y = y.reshape(-1, 1)
print(X.shape)
print(y.shape)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np

# Define the kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))

# Initialize the GPR model
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Train the model using the entire dataset
gpr.fit(X, y)  # Using X and y0 directly, without splitting into training and testing sets
print("GPR Trained!")

from deap import base, creator, tools, algorithms
from itertools import cycle
import random
# BOUNDS
SIGMA_EPSILON_MIN, SIGMA_EPSILON_MAX = 0.1, 2.5
initial_population = X.tolist()
def create_variation(individual):
    # Applies mutation to an individual by scaling each parameter randomly within 75% to 125% of its original value
    # and clamping it within specified SIGMA bounds.
    return [max(min(param * np.random.uniform(0.75, 1.25), SIGMA_EPSILON_MAX), SIGMA_EPSILON_MIN) for param in individual]
# Create an iterator that will cycle through the initial_population cyclically
cyclic_individuals = cycle(initial_population)
# Generate the required number of mutated individuals using the cyclic iterator
# This will start from the first individual, proceed to the last, and then start again from the first,
# continuing until it has generated 500 mutated individuals.
mutated_population = [create_variation(next(cyclic_individuals)) for _ in range(500)]


######### CROSSOVER ##########
data = np.array(mutated_population)
crossprob = 0.4 # Crossover probability
# Define the number of crossovers
num_crossovers = data.shape[0] // 2
print(f"Total number of crossovers for population of size {data.shape[0]}:{num_crossovers}")
# Perform crossover
for _ in range(num_crossovers):
    # Randomly select two distinct individuals
    i, j = np.random.choice(data.shape[0], size=2, replace=False)
    # Create a mask to determine genes to swap
    mask = np.random.rand(16) < crossprob  # Swap with a probability of 0.4

    # Swap genes according to the mask (Uncomment to observe mask in action)
    # print(i,j)
    # print(mask)
    # print(f"Before Crossover: {crossover_pop[i]} and {crossover_pop[j]}")
    data[i, mask], data[j, mask] = data[j, mask], data[i, mask]
    # print(f"After Crossover: {crossover_pop[i]} and {crossover_pop[j]}")

    # Compute fitness for each individual
def eval(individual):
  X_pred = np.array(individual)
  predicted_values = gpr.predict(X_pred.reshape(1, -1))
  return predicted_values[0]

fitnesses = np.array([eval(individual) for individual in data])

# Combine individuals with their fitnesses
individuals_with_fitness = list(zip(data, fitnesses))

# Sort by fitness
individuals_with_fitness.sort(key=lambda x: x[1])

# Print the top n sorted individuals with their fitness
print(f"Top {numberofind} sorted individuals and their fitness:")
for ind, fitness in individuals_with_fitness[:numberofind]:  # Access the first 10 after sorting
    print("Individual:", ind, "Fitness:", fitness)
# Extract just the top n individuals without their fitness values
top_n_individuals = np.array([ind for ind, fit in individuals_with_fitness[:numberofind]])

def process_and_save(new_x):
    # Assuming new_x is a numpy array of shape [8, 16]
    
    atom_types = ["S", "O", "CS", "HS", "CB", "HB", "CC", "HC"]  #Adjust accordingly
    
    # Specify the file name
    output_file = 'output.txt'
    
    # Open the file to write
    with open(output_file, 'w') as file:
        # Iterate over each mutation (each row in new_x)
        for mutation_index, values in enumerate(new_x):
            # Prepare the data for DataFrame for the current mutation
            data = []
            for i, atom_type in enumerate(atom_types):
                sigma = values[2 * i]  # Extracting sigma values
                epsilon = values[2 * i + 1]  # Extracting epsilon values
                data.append([atom_type, sigma, epsilon])
            
            # Create DataFrame for the current mutation
            df = pd.DataFrame(data, columns=["atomtype", "sigma(nm)", "eps (kJ/mol)"])
            
            # Print and save mutation number
            mutation_info = f"mutation: {mutation_index}"
            # print(mutation_info)
            file.write(mutation_info + '\n')
            
            # Print and save column titles
            column_titles = "{:<10} {:<10} {:<10}".format("atomtype", "sigma(nm)", "eps (kJ/mol)")
            # print(column_titles)
            file.write(column_titles + '\n')
            
            # Print and save rows
            for index, row in df.iterrows():
                row_data = "{:<10} {:<10.6f} {:<10.6f}".format(row['atomtype'], row['sigma(nm)'], row['eps (kJ/mol)'])
                # print(row_data)
                file.write(row_data + '\n')
            
            # Add an empty line for separation between mutations
            # print()
            file.write('\n')

# Assuming best_individuals is available and correctly formatted
process_and_save(top_n_individuals)
print("output.txt generated")

print("Run complete. Please use output.txt to run MD simulations.")
