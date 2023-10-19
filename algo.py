import random
import itertools
import time
from typing import Callable, Optional

import numpy as np

import config
from plot_utils import plot_results


def schwefel(x: list[float]) -> float:
    d = len(x)
    f = 418.9829 * d
    for xi in x:
        f = f - (xi * np.sin(np.sqrt(np.abs(xi))))
    return f


def bound_solution_in_x_range(x: list[float], x_range: list[tuple[float, float]]) -> list[float]:
    for j in range(len(x)):
        if x[j] < x_range[j][0]:
            x[j] = x_range[j][0]
        elif x[j] > x_range[j][1]:
            x[j] = x_range[j][1]
    return x


def local_search(cost_function: Callable, max_itr: int, convergence_threshold: float,
                 x_range: list[tuple[float, float]],
                 x_initial: Optional[np.array] = None) -> tuple[
    np.array, float, list[np.array], list[float]]:
    # Set the x_initial
    if x_initial is None:
        x_initial = [random.uniform(x_range[i][0], x_range[i][1]) for i in range(len(x_range))]

    x_current = x_initial
    cost_current = cost_function(x_current)

    x_history = [x_current]
    cost_history = [cost_current]

    convergence = False
    itr = 0
    while not convergence:
        # Generate neighboring solutions
        x_neighbor = [random.gauss(x, 0.1) for x in x_current]
        x_neighbor = bound_solution_in_x_range(x=x_neighbor, x_range=x_range)
        cost_neighbor = cost_function(x_neighbor)

        # Accept the neighbor if it has lower cost
        if cost_neighbor < cost_current:
            x_current = x_neighbor
            cost_current = cost_neighbor
            if cost_current < convergence_threshold:
                convergence = True

        x_history.append(x_current)
        cost_history.append(cost_current)

        if itr >= max_itr:
            convergence = True

        itr += 1

    # Get the best solution
    best_cost_index = np.argmin(cost_history)
    best_x = x_history[best_cost_index]
    best_cost = cost_history[best_cost_index]

    if config.PLOT:
        if config.CURRENT_K in config.K_TO_PLOT:
            config.K_TO_PLOT.remove(config.CURRENT_K)  # only plot the first visit to this neighborhood
            plot_results(f'local search (k={config.CURRENT_K})', best_x, best_cost, x_history, cost_history, schwefel,
                         x_range)
        if config.ALGO_DEMO and not (config.ALGO_DEMO_STEP == 2 and config.CURRENT_K == 1):
            if config.ALGO_DEMO_PREV_K > config.CURRENT_K:
                config.ALGO_DEMO_RESET += 1
                config.ALGO_DEMO_PREV_K = config.CURRENT_K - 1
                if config.ALGO_DEMO_RESET == 2:
                    config.ALGO_DEMO = False
            grid_div = tuple(config.FACTORS)
            diff = 1000
            x_ticks = [(i * diff / grid_div[0]) - 500 for i in range(grid_div[0] + 1)]
            y_ticks = [(i * diff / grid_div[1]) - 500 for i in range(grid_div[1] + 1)]
            plot_results(f'VNS step {config.ALGO_DEMO_STEP}', best_x, best_cost, x_history, cost_history, schwefel,
                         [(-500, 500), (-500, 500)], (x_ticks, y_ticks))
            config.ALGO_DEMO_PREV_K += 1
            config.ALGO_DEMO_STEP += 1

    return best_x, best_cost, x_history, cost_history


def largest_factors(num_neighborhoods: int, dims: int) -> list[int]:
    """
    Obtain the largest factors that can multiply to form the given num_slices
    :param num_neighborhoods: number to form from factors
    :param dims: number of desired factors
    :return: list of factors
    """
    factors = []
    factor = round(num_neighborhoods ** (1.0 / dims))
    while num_neighborhoods % factor != 0:
        factor = factor - 1
    factors.append(factor)
    if dims > 1:
        factors = factors + largest_factors(num_neighborhoods / factor, dims - 1)
    return factors


def get_neighborhoods(x_range: tuple[float, float], num_neighborhoods: int, dims: int) -> list[
    list[tuple[float, float]]]:
    """
    Create a given number of neighborhoods over the given landscape in the given dimensions
    :param x_range: hypercube range describing the landscape
    :param num_neighborhoods: number of neighborhoods to create in landscape
    :param dims: number of dimensions of the landscape and neighborhoods
    :return: List of neighborhoods. Each neighborhood is a list of ranges for each dimension, in order
    """
    low = x_range[0]
    high = x_range[1]
    diff = high - low
    split_x_ranges = []
    factors = largest_factors(num_neighborhoods, dims)
    # randomize to remove bias of lower dimensions being sliced less, which results in less sensitivity across that dimension
    np.random.shuffle(factors)
    if config.PLOT:
        config.FACTORS = factors[:2]
    for d in range(dims):
        split_x_ranges.append([((i * diff / factors[d]) + low, ((i + 1) * diff / factors[d]) + low) for i in
                               range(factors[d])])
    hypercube_ranges = list(itertools.product(*split_x_ranges))  # cartesian product
    return [list(x) for x in hypercube_ranges]


def get_layers(x_range: tuple[float, float], num_layers: int) -> list[tuple[float, float]]:
    """
    Create a number of hypercube layers from the given hypercube landscape
    :param x_range: hypercube range describing the landscape
    :param num_layers: number of layers to create in landscape
    :return: List of hypercube ranges describing the layers.
    """
    low = x_range[0]
    high = x_range[1]
    diff = high - low
    return [((low + diff / 2) - i * diff / (2 * num_layers), (low + diff / 2) + i * diff / (2 * num_layers)) for i in
            range(1, num_layers + 1)]


def variable_neighborhood_search(k_max: int, dims: int, convergence_threshold: float, vns_max_itr: int,
                                 ls_max_itr: int, x_range: tuple[float, float] = (-500, 500)) -> tuple[
    np.array, float, list[np.array], list[float]]:
    neighborhoods = get_neighborhoods(x_range, k_max, dims)

    # initial guess over entire landscape
    best_x, best_cost, x_history, cost_history = local_search(schwefel, ls_max_itr, convergence_threshold,
                                                              [x_range for _ in range(dims)])

    all_best_x = [best_x]
    all_best_cost = [best_cost]

    vns_iter = 0
    convergence = False
    while not convergence:
        k = 0
        while k < k_max:
            if config.PLOT:
                config.CURRENT_K = k + 1
            N = neighborhoods[k]
            iter_x, iter_cost, iter_x_history, iter_cost_history = local_search(schwefel, ls_max_itr,
                                                                                convergence_threshold, N)
            x_history += iter_x_history
            cost_history += iter_cost_history
            all_best_x.append(iter_x)
            all_best_cost.append(iter_cost)

            if iter_cost < best_cost:
                best_cost = iter_cost
                best_x = iter_x
                k = 0
            else:
                k += 1
        vns_iter += 1
        if best_cost < convergence_threshold or vns_iter >= vns_max_itr:
            convergence = True

    if config.PLOT:
        grid_div = tuple(config.FACTORS)
        diff = x_range[1] - x_range[0]
        x_ticks = [(i * diff / grid_div[0]) + x_range[0] for i in range(grid_div[0] + 1)]
        y_ticks = [(i * diff / grid_div[1]) + x_range[0] for i in range(grid_div[1] + 1)]
        plot_results('VNS', best_x, best_cost, x_history, cost_history, schwefel, [x_range, x_range],
                     (x_ticks, y_ticks), include_initial=False)
        plot_results('VNS - Best Only', best_x, best_cost, all_best_x, all_best_cost, schwefel,
                     [x_range, x_range], (x_ticks, y_ticks), include_initial=False)
    return best_x, best_cost, x_history, cost_history


def generalized_neighborhood_search(k_max: int, m_max: int, dims: int, convergence_threshold: float, vns_max_itr: int,
                                    ls_max_itr: int) -> tuple[np.array, float, list[np.array], list[float]]:
    layers = get_layers((-500, 500), m_max)

    best_x, best_cost, x_history, cost_history = local_search(schwefel, ls_max_itr, convergence_threshold,
                                                              get_neighborhoods(layers[0], 1, dims)[0])
    all_best_x = [best_x]
    all_best_cost = [best_cost]
    for layer in layers:
        if config.PLOT:
            # Spamming plots causes a 429 error in matplotlib for some reason
            time.sleep(2)
        iter_x, iter_cost, iter_x_history, iter_cost_history = variable_neighborhood_search(k_max, dims,
                                                                                            convergence_threshold,
                                                                                            vns_max_itr, ls_max_itr,
                                                                                            layer)
        x_history += iter_x_history
        cost_history += iter_cost_history
        all_best_x.append(iter_x)
        all_best_cost.append(iter_cost)

        if iter_cost < best_cost:
            best_cost = iter_cost
            best_x = iter_x
            if best_cost < convergence_threshold:
                break

    if config.PLOT:
        x_range = (-500, 500)
        grid_div = tuple(config.FACTORS)
        diff = x_range[1] - x_range[0]
        x_ticks = [(i * diff / grid_div[0]) + x_range[0] for i in range(grid_div[0] + 1)]
        y_ticks = [(i * diff / grid_div[1]) + x_range[0] for i in range(grid_div[1] + 1)]
        plot_results('GNS', best_x, best_cost, x_history, cost_history, schwefel, [x_range, x_range],
                     (x_ticks, y_ticks), include_initial=False)
        plot_results('GNS - Best Only', best_x, best_cost, all_best_x, all_best_cost, schwefel,
                     [x_range, x_range], (x_ticks, y_ticks), include_initial=False)
    return best_x, best_cost, x_history, cost_history
