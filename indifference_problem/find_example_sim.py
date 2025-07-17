from tqdm import tqdm
import matplotlib.pyplot as plt
plt.ion()
from simulate_election_with_turnout import *

def search_for_examples(alpha_start=0+1e-5, alpha_end=10.0, beta_start=0+1e-5, beta_end=10.0, resolution=50):
    #grid search
    alpha_vals = np.linspace(alpha_start, alpha_end, resolution)
    beta_vals = np.linspace(beta_start, beta_end, resolution)
    distortion_grid = np.zeros((len(alpha_vals), len(beta_vals)))
    reversal_grid = np.zeros_like(distortion_grid)
    successes = []

    for i, alpha_ in enumerate(tqdm(alpha_vals)):
        for j, beta_ in enumerate(beta_vals):
            result = simulate_election(alpha_, beta_, num_voters=25000)
            distortion_grid[i, j] = result['distortion']
            if result['true_winner'] == 'a' and result['voted_winner'] == 'b':
                reversal_grid[i, j] = 1
            successes.append((alpha_, beta_))

    #plot distortion heatmap
    plt.figure(figsize=(5, 5))
    plt.imshow(distortion_grid, extent=(beta_vals[0], beta_vals[-1], alpha_vals[0], alpha_vals[-1]),
            origin='lower', aspect='auto', cmap='coolwarm')
    plt.colorbar(label='Distortion')
    plt.xlabel('Beta')
    plt.ylabel('Alpha')
    plt.suptitle('Distortion across (alpha, beta)')
    plt.title("Found examples: {successes}")
    plt.tight_layout()

    #plot reversal region
    plt.figure(figsize=(5, 5))
    plt.imshow(reversal_grid, extent=(beta_vals[0], beta_vals[-1], alpha_vals[0], alpha_vals[-1]),
            origin='lower', aspect='auto', cmap='Greys')
    plt.xlabel('Beta')
    plt.ylabel('Alpha')
    plt.suptitle('Winner Reversal Region (true: a, voted: b)')
    #plt.title(f"Found examples: {successes}")
    plt.tight_layout()
    plt.show(block=False)
    input("Enter to exit and close all plots")

    return successes

search_for_examples(resolution=100)
