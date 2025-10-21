# Quantum Optimization problems

## Solving Optimization problems using quantum computing

This repo, contains some optimization problems I solved using some quantum computing techniques, like QUBO and QAOA.

## HOW TO USE

## For python environments

first of all, install `uv`, and run:

```bash
uv sync
```

then, you can run any script, just run:

```bash
uv run script
```

### For cuda environments

You need to have cuda toolkit installed and then run

```bash
make
```

inside the project folder.

## Solved Problems

* [knapsack](./knapsack.py)
* [knapsack cuda qubo](./knapsack-cuda/)
