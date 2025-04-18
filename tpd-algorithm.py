import random
import matplotlib.pyplot as plt
from itertools import product
from args import args

# Constants
COOPERATE, DEFECT = 1, 0
PAYOFF_MATRIX = {
    (COOPERATE, COOPERATE): (3, 3),
    (COOPERATE, DEFECT): (0, 5),
    (DEFECT, COOPERATE): (5, 0),
    (DEFECT, DEFECT): (1, 1)
}

def decode_strategy(genome):
    initial = genome[0]
    memory = {}
    index = 1
    for h in product([0, 1], repeat=4): 
        memory[tuple(h)] = genome[index]
        index += 1
    return initial, memory

# def prob_move(probability):
#     return COOPERATE if random.random() < probability else DEFECT

def prob_move(probability, noise):
    move = COOPERATE if random.random() < probability else DEFECT
    # Small probability of move being flipped
    if random.random() < noise:
        move = 1-move
    return move
    


# Play IPD game for N rounds
def play_ipd(strategy_a, strategy_b, rounds, noise):
    total_a, total_b = 0, 0
    a_initial, a_mem = decode_strategy(strategy_a)
    b_initial, b_mem = decode_strategy(strategy_b)

    # Start with no initial moves
    a_history, b_history = [prob_move(a_initial, noise)], [prob_move(b_initial, noise)]


    for _ in range(rounds):
        # Determine moves
        if len(a_history) < 2:
            move_a = prob_move(a_initial, noise)
            move_b = prob_move(b_initial, noise)
        else:
            move_a = prob_move(a_mem[(a_history[-2], a_history[-1], b_history[-2], b_history[-1])], noise)
            move_b = prob_move(b_mem[(b_history[-2], b_history[-1], a_history[-2], a_history[-1])], noise)

        payoff_a, payoff_b = PAYOFF_MATRIX[(move_a, move_b)]
        total_a += payoff_a
        total_b += payoff_b

        a_history.append(move_a)
        b_history.append(move_b)

    return total_a


# Fixed strategies
def always_cooperate(_): return 1
def always_defect(_): return 0
def tit_for_tat(last): return 1 if last[1] == 1 else 0



# Wrap fixed strategies into 5-bit genome style
def fixed_strategies(name):
    if name == 'ALLC':
        return [1] + [1]*16
    elif name == 'ALLD':
        return [0] + [0]*16
    elif name == 'TFT':
        # Start with cooperation. Mimic opponent’s last move in recent history.
        genome = [1]
        for history in product([0, 1], repeat=4):
            response = history[3]  # opponent's last move
            genome.append(response)
        return genome
    
# Genetic Algorithm Functions
def random_genome():
    return [random.uniform(0, 1) for _ in range(17)]

def initialize_population(size):
    return [random_genome() for _ in range(size)]

def tournament_selection(pop, fitnesses, k=3):
    selected = random.sample(list(zip(pop, fitnesses)), k)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]

def crossover(parent1, parent2):
    return [random.choice([g1, g2]) for g1, g2 in zip(parent1, parent2)]


def mutate(genome, mut_rate):
    return [min(1.0, max(0.0, gene + random.gauss(0, 0.1))) if random.random() < mut_rate else gene for gene in genome]

# Evaluate fitness against fixed opponents
def evaluate_fitness(ind, strategy, rounds, noise):
    score = 0
    opponents = ['ALLC', 'ALLD', 'TFT']
    if strategy == 'ALL':
        for opp in opponents:
            score += play_ipd(ind, fixed_strategies(opp), rounds, noise)
    else:
        score = play_ipd(ind, fixed_strategies(strategy), rounds, noise)
    return score

# Evaluate fitness against population
def evaluate_fitness_coevolution(population, sample_size, rounds, noise):
    fitnesses = []
    for i, ind in enumerate(population):
        opponents = random.sample([j for j in range(len(population)) if j != i], sample_size)
        total_score = 0
        for j in opponents:
            total_score += play_ipd(ind, population[j], rounds, noise)
        fitnesses.append(total_score / sample_size)
    return fitnesses

# Genetic Algorithm
def evolve(generations,
           pop_size,
           cross_rate,
           mut_rate,
           rounds,
           noise,
           strategy,
           sample_size):
    
    population = initialize_population(pop_size)
    best_fitness = 0
    best_individual = None
    fitness_history = []
    for gen in range(generations):
        if strategy == 'EVOLVE':
            fitnesses = evaluate_fitness_coevolution(population, sample_size, rounds, noise)
        else: 
            fitnesses = [evaluate_fitness(ind, strategy, rounds, noise) for ind in population]
        new_population = []

        for _ in range(pop_size):
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            if random.random() < cross_rate:
                child = crossover(p1, p2)
            else:
                child = p1.copy()

            child = mutate(child, mut_rate)
            new_population.append(child)

        population = new_population
        best = max(fitnesses)
        if best > best_fitness:
            best_fitness = best
            best_individual = population[fitnesses.index(best)]
        fitness_history.append(best_fitness)

        print(f'Generation {gen+1}: Best Fitness = {best_fitness:.2f}')
    return best_individual, best_fitness, fitness_history



# Run
if __name__ == "__main__":
    best_strategy, best_fitness, fitness_history = evolve(
        generations=args.generations,
        pop_size=args.pop_size,
        cross_rate=args.cross_rate,
        mut_rate=args.mut_rate,
        rounds=args.rounds,
        noise=args.noise,
        strategy=args.strategy,
        sample_size=args.sample_size
    )
    
    # Plot results
    plt.plot(fitness_history)
    plt.title('Best Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid()
    plt.savefig('plot.png')

    best_strategy = [round(elem, 2) for elem in best_strategy]
    initial, mem = decode_strategy(best_strategy)
    mem = {elem: round(mem[elem], 2) for elem in mem}
    # Show best evolved strategy
    print("Best fitness:", best_fitness)
    print("Best Strategy Genome: ", best_strategy)
    print("Decoded as: ", (initial, mem))