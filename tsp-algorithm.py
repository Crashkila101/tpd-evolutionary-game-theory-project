import numpy as np
import tsplib95
import random
import matplotlib.pyplot as plt

# Load TSP file
problem = tsplib95.load('tsp-files/berlin52.tsp')

# Extract city coordinates
cities = list(problem.node_coords.values())
print(f"Number of cities: {len(cities)}")
print(f"City coordinates: {cities[:5]}")  # Print first 5 cities

# Compute the Euclidean distance between all city pairs
def calculate_distance_matrix(cities):
    n = len(cities)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = cities[i]
                x2, y2 = cities[j]
                distance_matrix[i][j] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance_matrix

distance_matrix = calculate_distance_matrix(cities)
print(f"Distance matrix shape: {distance_matrix.shape}")

# Parameters
POPULATION_SIZE = 2000
GENERATIONS = 500
MUTATION_RATE = 0.08
CROSSOVER_RATE = 0.9
ELITISM_RATE = 0.1

# Initialize population
def create_individual(cities):
    return random.sample(range(len(cities)), len(cities))

def create_population(cities, size):
    return [create_individual(cities) for _ in range(size)]

# Fitness function
def calculate_fitness(individual, distance_matrix):
    total_distance = 0
    for i in range(len(individual) - 1):
        total_distance += distance_matrix[individual[i]][individual[i + 1]]
    total_distance += distance_matrix[individual[-1]][individual[0]]  # Return to start
    return 1 / total_distance  # Higher fitness for shorter distances

# Selection (Tournament Selection)
def select_parent(population, fitnesses, tournament_size=5):
    tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
    tournament.sort(key=lambda x: x[1], reverse=True)
    return tournament[0][0]

# Crossover (Order Crossover)
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    remaining = [item for item in parent2 if item not in child]
    child = [item if item is not None else remaining.pop(0) for item in child]
    return child

# Mutation (Swap Mutation)
def mutate(individual):
    if random.random() < MUTATION_RATE:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

# Genetic Algorithm
def genetic_algorithm(cities, distance_matrix, population_size, generations, crossover_rate, elitism_size):
    population = create_population(cities, population_size)
    best_individual = None
    best_fitness = 0
    best_distance = float('inf')

    for generation in range(generations):
        fitnesses = [calculate_fitness(ind, distance_matrix) for ind in population]

        # Track the best individual
        current_best_fitness = max(fitnesses)
        current_best_distance = 1 / current_best_fitness
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_individual = population[fitnesses.index(current_best_fitness)]

        # Select elites (top elitism_rate% of individuals)
        sorted_population = [ind for _, ind in sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)]
        elites = sorted_population[:elitism_size]

        new_population = elites.copy()

        # Generate new population (preserve elites and create new children)
        for _ in range(population_size):
            parent1 = select_parent(population, fitnesses)
            # Crossover probability is controlled by crossover rate, otherwise just pass parent1 genes down
            if random.random() < crossover_rate:
                parent2 = select_parent(population, fitnesses)
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy()

            child = mutate(child)
            new_population.append(child)

        population = new_population

        # Track the best individual
        current_best_fitness = max(fitnesses)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[fitnesses.index(current_best_fitness)]

        print(f"Generation {generation + 1}: Best Fitness = {1 / best_fitness}")

    return best_individual, 1 / best_fitness

# Run the genetic algorithm
best_tour, best_distance = genetic_algorithm(
    cities, 
    distance_matrix,
    population_size = POPULATION_SIZE, 
    generations = GENERATIONS, 
    crossover_rate = CROSSOVER_RATE,
    elitism_size = int(ELITISM_RATE*POPULATION_SIZE)
    )

print(f"Best tour: {best_tour}")
print(f"Best distance: {best_distance}")