import yaml
import numpy as np

with open('./config/config.yml', 'r') as f:
    config = yaml.safe_load(f)

# global vars used for plotting
PLOT = config['plot_results']
num_to_plot = config['vns']['neighborhoods']['num_to_plot']
plot_pt_dist = np.floor(config['vns']['neighborhoods']['count'] / num_to_plot) if num_to_plot > 0 else 0
K_TO_PLOT = [i * plot_pt_dist for i in range(1, num_to_plot + 1)]
CURRENT_K = 0
FACTORS = []  # Must save factors to show neighborhoods in graph

# Used for showing VNS steps
ALGO_DEMO = config['show_vns_steps']
ALGO_DEMO_RESET = 0
ALGO_DEMO_PREV_K = 0
ALGO_DEMO_STEP = 0
