import argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--file_path", help="Specify the path at which the TSP file resides (default: tsp-files/kroA100.tsp)", type=str, default="tsp-files/kroA100.tsp")
parser.add_argument("--pop_size", help="Specify the population size for the algorithm (default: 750)", type=int, default=750)
parser.add_argument("--generations", help="Specify how many generations the algorithm should run (default: 300)", type=int, default=300)
parser.add_argument("--cross_rate", help="Specifies the rate at which genes from both parents combine into a child (default: 0.9)", type=float, default=0.9)
parser.add_argument("--mut_rate", help="Specify the rate at which genes mutate (default: 0.02)", type=float, default = 0.02)
parser.add_argument("--elitism_size", help="Specify the number of 'elites' that pass down to the next generation (default: 10)", type=int, default=10)
parser.add_argument("--mut_fn", help="Specify whether to use swap or inversion mutation (default: swap)", type=str, default="swap")
parser.add_argument("--cross_fn", help="Specify whether to use order crossover (ox) or pmx crossover (default: ox)", type=str, default="ox")

args = parser.parse_args()