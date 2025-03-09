import numpy as np
import tsplib95
import random
import matplotlib.pyplot as plt
import itertools
import time

# Load TSP Data
def load_tsp_data(file_path):
    problem = tsplib95.load(file_path)
    cities = list(problem.node_coords.values())
    return cities

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
def select_parent(population, fitnesses, tournament_size=3):
    tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
    tournament.sort(key=lambda x: x[1], reverse=True)
    return tournament[0][0]

# Crossover (Order Crossover)
def ox_crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1]*size
    
    # Copy segment from parent1
    child[a:b] = parent1[a:b]
    
    # Fill remaining positions from parent2
    ptr = b
    for gene in parent2[b:] + parent2[:b]:
        if gene not in child[a:b]:
            if ptr >= size:
                ptr = 0
            while child[ptr] != -1:
                ptr += 1
                if ptr >= size:
                    ptr = 0
            child[ptr] = gene
            ptr += 1
    return child

def pmx_crossover(parent1, parent2):
    size = len(parent1)
    
    # Choose two random crossover points
    cx1, cx2 = sorted(random.sample(range(size), 2))
    
    # Create a child with a copied segment from Parent 1
    child = [-1] * size
    child[cx1:cx2+1] = parent1[cx1:cx2+1]

    # Mapping relationships from Parent 1 to Parent 2
    mapping = {parent1[i]: parent2[i] for i in range(cx1, cx2+1)}

    def fill_remaining(child, parent, mapping):
        for i in range(size):
            if child[i] == -1:  # Need to fill
                value = parent[i]
                while value in mapping:  # Resolve mapping conflicts
                    value = mapping[value]
                child[i] = value

    # Fill remaining genes from Parent 2
    fill_remaining(child, parent2, mapping)

    return child  # Return only one child

    
# Swap Mutation
def swap_mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

# Inversion Mutation
def inversion_mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = sorted(random.sample(range(len(individual)), 2))
        individual[a:b+1] = reversed(individual[a:b+1])
    return individual


# Genetic Algorithm
def genetic_algorithm(cities,
                      distance_matrix, 
                      population_size, 
                      generations, 
                      crossover_rate, 
                      mutation_rate, 
                      elitism_size,
                      mutation_fn,
                      crossover_fn):

    # Set parameters
    population = create_population(cities, population_size)
    best_individual = None
    best_fitness = 0
    best_distance = float('inf')
    fitness_history = []
    crossover = ox_crossover if crossover_fn == "ox" else pmx_crossover
    if mutation_fn == "swap":
        mutate = swap_mutate 
    elif mutation_fn == "inversion":
        mutate = inversion_mutate 
    
    print(f"Calculating fitness over {generations} generations with {crossover_fn} crossover and {mutation_fn} mutation:")

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

            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

        # Track the best individual
        current_best_fitness = max(fitnesses)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[fitnesses.index(current_best_fitness)]
        
        # Track fitness history
        fitness_history.append(1 / best_fitness)

        if generation % 50 == 0:
                print(f"Generation {generation}: Best Distance = {1/best_fitness:.2f}")

    return best_individual, 1 / best_fitness, fitness_history

# Plot tour taken
def plot_tour(cities, tour):
    x = [cities[i][0] for i in tour]
    y = [cities[i][1] for i in tour]
    x.append(x[0])  # Return to start
    y.append(y[0])

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.scatter(x, y, c='red')
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(tour[i % len(tour)]), fontsize=12)
    plt.title(f"Best Tour (Distance = {best_distance:.2f})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid()
    plt.savefig("plot.png")
    
# Plot fitness progression over generations
def plot_fitness(fitness_history):
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, 'b')
    plt.title('Best Distance Progression')
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.savefig('fig.png')


# Run the genetic algorithm
if __name__ == "__main__":
    # Load TSP data
    cities = load_tsp_data("tsp-files/kroA100.tsp")
    distance_matrix = calculate_distance_matrix(cities)


    start_time = time.time()

    best_tour, best_distance, fitness_history = genetic_algorithm(
        cities, distance_matrix,
        population_size=1000,
        generations=300,
        crossover_rate=0.9,
        mutation_rate=0.02,
        elitism_size=10,
        mutation_fn="swap",
        crossover_fn="ox"
    )

    time_taken = time.time() - start_time

    plot_tour(cities, best_tour)
    plot_fitness(fitness_history)
    print(f"Best tour: {best_tour}")
    print(f"Best distance: {best_distance:.2f}")
    print(f"Time taken: {time_taken:.2f} seconds")


