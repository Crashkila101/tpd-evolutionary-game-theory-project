import argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--generations", help="Specify how many generations the algorithm should run (default: 300)", type=int, default=100)
parser.add_argument("--pop_size", help="Specify the population size for the algorithm (default: 750)", type=int, default=50)
parser.add_argument("--cross_rate", help="Specifies the rate at which genes from both parents combine into a child (default: 0.9)", type=float, default=0.9)
parser.add_argument("--mut_rate", help="Specify the rate at which genes mutate (default: 0.02)", type=float, default = 0.05)
parser.add_argument("--rounds", help="Specify the number of rounds the iterated prisoner dilemma is played for (default: 150)", type=int, default=150)
parser.add_argument("--noise", help="Specify the probability any genome is flipped (default: 0.05)", type=float, default=0.05)
parser.add_argument("--strategy", help="Specify the strategy (or strategies) which the algorithm competes against (default: ALL)", type=str, default='ALL')
parser.add_argument("--sample_size", help="(If running against population) Specify the subset of the population which an individual will play in the iterated prisoner's dilemma (default: 20)", type=int, default=20)

args = parser.parse_args()