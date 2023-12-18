import numpy as np

# Genetic Algorithm parameters
population_size = 50
num_parameters = 8
parameter_ranges = [(0, 10), (0, 5), (0, 8), (0, 15), (0, 20), (0, 12), (0, 10), (0, 7)]
mutation_rate = 0.1
crossover_rate = 0.8
generations = 100

# Fitness function (example: sum of squares)
def fitness(individual):
    return -sum(np.square(individual))  # Minimization problem

# Initialize population
def initialize_population(size, num_parameters, parameter_ranges):
    population = []
    for _ in range(size):
        individual = [np.random.uniform(low, high) for (low, high) in parameter_ranges]
        population.append(individual)
    return population

# Crossover
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation
def mutate(individual, mutation_rate, parameter_ranges):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.uniform(parameter_ranges[i][0], parameter_ranges[i][1])
    return individual

# Selection
def select_parents(population, fitness_values):
    probabilities = np.exp(fitness_values) / sum(np.exp(fitness_values))
    indices = np.arange(len(population))
    selected_indices = np.random.choice(indices, size=len(population), p=probabilities)
    return [population[i] for i in selected_indices]

# Genetic Algorithm
def genetic_algorithm(population_size, num_parameters, parameter_ranges, mutation_rate, crossover_rate, generations):
    population = initialize_population(population_size, num_parameters, parameter_ranges)

    for generation in range(generations):
        fitness_values = [fitness(individual) for individual in population]

        # Select parents
        parents = select_parents(population, fitness_values)

        # Create offspring through crossover and mutation
        offspring = []
        for i in range(0, len(parents), 2):
            if np.random.rand() < crossover_rate:
                child1, child2 = crossover(parents[i], parents[i + 1])
            else:
                child1, child2 = parents[i], parents[i + 1]

            child1 = mutate(child1, mutation_rate, parameter_ranges)
            child2 = mutate(child2, mutation_rate, parameter_ranges)

            offspring.extend([child1, child2])

        # Replace old population with offspring
        population = offspring

        # Print best fitness in each generation
        best_fitness = max(fitness_values)
        print(f"Generation {generation + 1}, Best Fitness: {best_fitness:.4f}")

    # Return the best individual in the final generation
    best_index = np.argmax(fitness_values)
    best_individual = population[best_index]
    return best_individual

if __name__ == "__main__":
    best_solution = genetic_algorithm(population_size, num_parameters, parameter_ranges, mutation_rate, crossover_rate, generations)
    print("Best Solution:", best_solution)
