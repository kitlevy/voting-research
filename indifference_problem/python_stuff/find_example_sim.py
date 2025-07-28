from tqdm import tqdm
import matplotlib.pyplot as plt
plt.ion()
from simulate_election_with_turnout import *

def search_for_examples(alpha_start=0+1e-5, alpha_end=10.0, beta_start=0+1e-5, beta_end=10.0, resolution=50, voters_per_sim=25000):
    #grid search
    epsilon = 1e-4
    alpha_vals = np.linspace(alpha_start, alpha_end, resolution)
    beta_vals = np.linspace(beta_start, beta_end, resolution)
    distortion_grid = np.ones((len(alpha_vals), len(beta_vals)))
    reversal_grid = np.ones_like(distortion_grid)
    neg_dist_grid = np.ones_like(distortion_grid)
    successes = []

    for i, alpha_ in enumerate(tqdm(alpha_vals)):
        for j, beta_ in enumerate(beta_vals):
            if abs(alpha_ - beta_) > epsilon:
                result = simulate_election(alpha_, beta_, num_voters=voters_per_sim)
                distortion = result['distortion']
                if distortion != 1:
                    distortion_grid[i, j] = distortion
                    if distortion < 1:
                        neg_dist_grid[i, j] = distortion
                if result['true_winner'] == 'a' and result['voted_winner'] == 'b':
                    reversal_grid[i, j] = distortion
                    successes.append((float(alpha_), float(beta_)))
    print(successes)

    min_color = 0.997
    max_color = 1.003

    #plot distortion heatmap
    plt.figure(figsize=(5, 5))
    plt.imshow(distortion_grid, extent=(beta_vals[0], beta_vals[-1], alpha_vals[0], alpha_vals[-1]),
            origin='lower', aspect='auto', cmap='bwr', vmin=min_color, vmax=max_color)
    plt.colorbar(label='Distortion')
    plt.xlabel('Beta')
    plt.ylabel('Alpha')
    plt.suptitle('Distortion across (alpha, beta)')
    plt.tight_layout()

    
    #plot reversal region
    plt.figure(figsize=(5, 5))
    plt.imshow(reversal_grid, extent=(beta_vals[0], beta_vals[-1], alpha_vals[0], alpha_vals[-1]),
            origin='lower', aspect='auto', cmap='bwr', vmin=min_color, vmax=max_color)
    plt.xlabel('Beta')
    plt.ylabel('Alpha')
    plt.colorbar(label='Distortion')
    plt.suptitle('Winner Reversal Region (true: a, voted: b)')
    plt.tight_layout()

    #plot distortion < 1 region
    plt.figure(figsize=(5, 5))
    plt.imshow(neg_dist_grid, extent=(beta_vals[0], beta_vals[-1], alpha_vals[0], alpha_vals[-1]),
            origin='lower', aspect='auto', cmap='seismic', vmin=min_color, vmax=max_color)
    plt.xlabel('Beta')
    plt.ylabel('Alpha')
    plt.colorbar(label='Distortion')
    plt.suptitle('Region with Distortion < 1')
    plt.tight_layout()

    plt.show(block=False)
    input("Enter key to close all plots")

    return successes

if __name__ == "__main__":
    voters = 500_000
    voters = 1_000_000
    search_for_examples(voters_per_sim=voters)
