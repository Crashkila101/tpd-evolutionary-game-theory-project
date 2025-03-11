# tsp-evolutionary-search-project
This is a project which aims to solve TSP datasets with the **.tsp** file extension.
This program requires **Python 3** to be installed on your system in order to work; you can get it from here: https://www.python.org/downloads/

## Environment setup for Linux

1. Clone the repository:

```bash
git clone https://github.com/Crashkila101/tsp-evolutionary-search-project.git
```

2. Create a new python environment
```bash
python -m venv venv
```

3. Activate the environment
```bash
source venv/bin/activate
```

4. Install the required packages from requirements.txt
```bash
pip install -r requirements.txt
#(This may take a while so be patient)
```

## Environment setup for Windows

1. Clone the repository:

```bash
git clone https://github.com/Crashkila101/tsp-evolutionary-search-project.git
```

2. Create a new python environment
```bash
python -m venv venv
```

3. Activate the environment
```bash
venv/Scripts/activate.bat
```

4. Install the required packages from requirements.txt
```bash
pip install -r requirements.txt
#(This may take a while so be patient)
```

## Running the program
Run the program by specifying the path to your tsp file, for example:
```bash
python tsp-algorithm.py --file_path "tsp-files/berlin52.tsp"
```     
If you want to tweak your algorithm, you can change the values of hyperparameters in the command line, or alternatively, you can change the default values in args.py to suit your needs.

## Arguments
**--file_path <path-to-file>** (optional) Specify the path at which the TSP file resides (default: tsp-files/kroA100.tsp)
**--pop_size <population>** (optional) Specify the population size for the algorithm (default: 750)
**--generations <generations>** (optional) Specify how many generations the algorithm should run (default: 300)
**--cross_rate <rate>** (optional) Specifies the rate at which genes from both parents combine into a child (default: 0.9)
**--mut_rate <rate>** (optional) Specify the rate at which genes mutate (default: 0.02)
**--elitism_size <size>** (optional) Specify the number of 'elites' that pass down to the next generation (default: 10)
**--mut_fn <function>** (optional) Specify whether to use swap ["swap"] or inversion mutation ["invert"] (default: swap)
**--cross_fn <function>** (optional) Specify whether to use order crossover ["ox"] or pmx crossover ["pmx"] (default: ox)

Example:
```bash
python tsp-algorithm.py --file_path "tsp-files/berlin52.tsp" --pop_size 1000 --generations 500 --cross_rate 1.0 --mut_rate 0.05 --elitism_size 5 --mut_fn "invert" --cross_fn "pmx"
```     