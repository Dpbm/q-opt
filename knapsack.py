from typing import Dict
from pprint import pp
import pandas as pd
import dimod

def get_qubo(items:Dict[str, float], max_weight:float) -> dimod.BinaryQuadraticModel:
    """
    Gets a dictionary of items and return the qubo model that fits the maximum
    of items inside a bag with at most max_weight.
    """

    model = dimod.BinaryQuadraticModel(vartype="BINARY")

    """
    P(max(min_weight <= sum{i}(x_i * Q_ii) <= max_weight))^2
    """
    P = 3
    min_weight = min(items.values())
    model.add_linear_inequality_constraint(list(items.items()), P, lb=min_weight,ub=max_weight, label="weight")

    return model

if __name__ == "__main__":
    items = {
        "laptop":2,
        "notebook":0.6,
        "book":0.5,
        "umbrella":0.3,
        "apple": 0.1
    }

    print("--Knapsack problem using qubo--")
    print("items: ")
    pp(items,indent=2,width=10)

    qubo = get_qubo(items,3)
    print("\nqubo model:")
    print(qubo)

    sampler = dimod.ExactSolver()
    sampleset = sampler.sample(qubo)
    df = sampleset.to_pandas_dataframe()

    target_columns = list(items.keys())
    weights = pd.Series(list(items.values()), index=list(items.keys()))

    solutions = df[df['energy'] == df['energy'].min()][target_columns]
    mul_weights = solutions.mul(weights, axis=1)
    

    solutions['total_weight'] = mul_weights.sum(axis=1)

    print("\nSolutions:")
    print(solutions)
