import sys
from typing import Callable

from algo import variable_neighborhood_search, generalized_neighborhood_search
from config import config

if __name__ == '__main__':
    # hyperparameters
    k_max: int = config['vns']['neighborhoods']['count']
    dimensions: int = config['dimensions']
    convergence_threshold: float = config['convergence_threshold']
    vns_max_itr: int = config['vns']['max_itr']
    ls_max_itr: int = config['local_search']['max_itr']
    m_max: int = config['gns']['layers']

    if dimensions < 2 and config['plot_results']:
        print("Error: Must have >=2 dimensions to plot results")
        sys.exit(1)

    search_algorithms: dict[str, Callable] = {
        'vns': lambda: variable_neighborhood_search(k_max, dimensions, convergence_threshold,
                                                    vns_max_itr, ls_max_itr),
        'gns': lambda: generalized_neighborhood_search(k_max, m_max, dimensions,
                                                       convergence_threshold, vns_max_itr,
                                                       ls_max_itr)
    }

    selected_algo: str = config['search_algorithm']
    print(selected_algo.upper())
    best_x, best_cost, _, _ = search_algorithms[selected_algo]()
    print(f"cost: {best_cost}, vector: {best_x}")
