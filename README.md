# tpd-evolutionary-search-project
This is a project which aims to evolve strategies for the the Iterated Prisoner's Dilemma (IPD)
This program requires **Python 3** to be installed on your system in order to work; you can get it from here: https://www.python.org/downloads/

## Environment setup for Linux

1. Clone the repository:

```bash
git clone https://github.com/Crashkila101/tpd-evolutionary-game-theory-project.git
```

2. Create a new python environment:
```bash
cd tpd-evolutionary-game-theory-project
python -m venv venv
```

3. Activate the environment:
```bash
source venv/bin/activate
```

4. Install the required packages from requirements.txt:
```bash
pip install -r requirements.txt
#(This may take a while so be patient)
```

## Environment setup for Windows

1. Clone the repository:

```bash
git clone https://github.com/Crashkila101/tpd-evolutionary-game-theory-project.git
```

2. Create a new python environment:
```bash
cd tpd-evolutionary-game-theory-project
python -m venv venv
```

3. Activate the environment:
```bash
venv/Scripts/activate.bat
```

4. Install the required packages from requirements.txt:
```bash
pip install -r requirements.txt
#(This may take a while so be patient)
```

## Running the program
Run the program using python
```bash
python tpd-evolutionary-game-theory-project
```     
If you want to tweak your algorithm, you can change the values of parameters in the command line, or alternatively, you can change the default values in args.py to suit your needs.

## Arguments
**--pop_size <population>** (optional) Specify the population size for the algorithm (default: 750)

**--generations <generations>** (optional) Specify how many generations the algorithm should run (default: 300)

**--cross_rate <rate>** (optional) Specifies the rate at which genes from both parents combine into a child (default: 0.9)

**--mut_rate <rate>** (optional) Specify the rate at which genes mutate (default: 0.05)

**--rounds <rounds>** (optional) Specify the number of rounds two agents play for (default: 150)

**--noise <noise>** (optional) Specify the probability that any genome is flipped (default: 0.05)

**--strategy <strategy>** (optional) Specify which strategy the algorithm competes against. 'TFT' is tit-for-tat, 'ALLC' always cooperates, 'ALLD' always defects, and 'EVOLVE' will make the population compete against itself. 'ALL' will compete against every fixed strategy.

**--sample_size <size>** (optional) If running against the population, specify the subset of the population which an individual will play in the iterated prisoner's dilemma (default: 20)

Example:
```bash
python tpd-algorithm.py --pop_size 1000 --generations 500 --cross_rate 1.0 --mut_rate 0.05 --rounds 200 --noise 0 --strategy 'ALL'
```     
