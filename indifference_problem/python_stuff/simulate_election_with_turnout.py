import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

def simulate_election(alpha_, beta_, num_voters=100000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    u_all = beta.rvs(alpha_, beta_, size=num_voters)
    #probability of showing up is abs(2u - 1)
    turnout_probs = np.abs(2 * (u_all - 0.5))
    turns_out = np.random.rand(num_voters) < turnout_probs
    u_voted = u_all[turns_out]

    #population-level preferred candidate
    pop_support_a = np.mean(u_all > 0.5)
    true_winner = 'a' if pop_support_a > 0.5 else 'b'

    #vote-level winner
    vote_support_a = np.mean(u_voted > 0.5)
    voted_winner = 'a' if vote_support_a > 0.5 else 'b'

    #sw = total utility for selected winner
    sw_true = np.sum(u_all if true_winner == 'a' else 1 - u_all)
    sw_vote = np.sum(u_all if voted_winner == 'a' else 1 - u_all)
    distortion = sw_true / sw_vote

    return {
        'alpha': alpha_,
        'beta': beta_,
        'true_winner': true_winner,
        'voted_winner': voted_winner,
        'distortion': distortion,
        'turnout_rate': len(u_voted) / len(u_all),
        'u_all': u_all,
        'u_voted': u_voted
    }

def plot_simulation(result, bins=50):
    #u_all vs. u_voted
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(result['u_all'], bins=bins, alpha=0.7, label="Entire Population", density=True)
    ax[0].hist(result['u_voted'], bins=bins, alpha=0.7, label="People who Voted", density=True)
    ax[0].axvline(0.5, color='k', linestyle='--', label="Preference Boundary")
    ax[0].set_title("Distribution of Utilities")
    ax[0].legend()

    fig.suptitle(f"Alpha: {result['alpha']}, Beta: {result['beta']}, True winner: {result['true_winner']}, Voted winner: {result['voted_winner']}, Distortion: {result['distortion']:.4f}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    epsilon = 0
    examples = []
    for i in range(10000):
        alph, bet = 10 * np.random.random_sample(), 10 * np.random.random_sample()
        result = simulate_election(alph, bet)
        if result['true_winner'] == 'a' and result['voted_winner'] == 'b':
            print((alph, bet))
            examples.append((alph, bet))
            plot_simulation(result)
    print(examples)
