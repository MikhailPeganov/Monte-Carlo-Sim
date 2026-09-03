"""
Monte Carlo Option Pricing Model
---------------------------------
Prices European call and put options via Monte Carlo simulation of
Geometric Brownian Motion, and validates the result against the
closed-form Black-Scholes price.

Author: Mikhail Peganov
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
PRICE = 100      # current underlying price
STRIKE = 105     # strike price
YEARS = 1.0      # time to maturity (years)
RATE = 0.05      # risk-free rate
VOL = 0.20       # annualised volatility
SIMS = 100_000   # number of simulated paths
DAYS = 252       # trading days per year

np.random.seed(42)


def bs_price(price, strike, years, rate, vol):
    """Closed-form Black-Scholes price for a European call and put."""
    a = (np.log(price / strike) + (rate + 0.5 * vol**2) * years) / (vol * np.sqrt(years))
    b = a - vol * np.sqrt(years)
    call = price * norm.cdf(a) - strike * np.exp(-rate * years) * norm.cdf(b)
    put = strike * np.exp(-rate * years) * norm.cdf(-b) - price * norm.cdf(-a)
    return call, put


def simulate_paths(price, years, rate, vol, sims, days):
    """Simulate `sims` GBM price paths with `days` steps using vectorised NumPy."""
    dt = years / days
    z = np.random.standard_normal((sims, days))
    steps = (rate - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
    log_paths = np.log(price) + np.cumsum(steps, axis=1)
    paths = np.exp(log_paths)
    paths = np.hstack([np.full((sims, 1), price), paths])
    return paths


def main():
    # Closed-form benchmark
    bs_call, bs_put = bs_price(PRICE, STRIKE, YEARS, RATE, VOL)

    # Monte Carlo simulation
    paths = simulate_paths(PRICE, YEARS, RATE, VOL, SIMS, DAYS)
    end_prices = paths[:, -1]
    call_payoffs = np.maximum(end_prices - STRIKE, 0)
    put_payoffs = np.maximum(STRIKE - end_prices, 0)

    mc_call = np.exp(-RATE * YEARS) * np.mean(call_payoffs)
    mc_put = np.exp(-RATE * YEARS) * np.mean(put_payoffs)
    call_err = np.exp(-RATE * YEARS) * np.std(call_payoffs) / np.sqrt(SIMS)
    put_err = np.exp(-RATE * YEARS) * np.std(put_payoffs) / np.sqrt(SIMS)

    print("=" * 55)
    print("MONTE CARLO OPTION PRICING RESULTS")
    print("=" * 55)
    print(f"Parameters: price={PRICE}, strike={STRIKE}, years={YEARS}, rate={RATE}, vol={VOL}")
    print(f"Simulations: {SIMS:,}  |  Steps/path: {DAYS}")
    print("-" * 55)
    print(f"{'':15}{'Monte Carlo':>15}{'Black-Scholes':>18}{'Diff':>10}")
    print(f"{'Call price':15}{mc_call:>15.4f}{bs_call:>18.4f}{mc_call - bs_call:>10.4f}")
    print(f"{'Put price':15}{mc_put:>15.4f}{bs_put:>18.4f}{mc_put - bs_put:>10.4f}")
    print("-" * 55)
    print(f"Call 95% CI: [{mc_call - 1.96 * call_err:.4f}, {mc_call + 1.96 * call_err:.4f}]")
    print(f"Put  95% CI: [{mc_put - 1.96 * put_err:.4f}, {mc_put + 1.96 * put_err:.4f}]")
    print("=" * 55)

    # ----------------------------
    # Plots
    # ----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    n_show = 100
    for i in range(n_show):
        axes[0, 0].plot(paths[i], linewidth=0.6, alpha=0.5)
    axes[0, 0].axhline(STRIKE, color="black", linestyle="--", label=f"Strike={STRIKE}")
    axes[0, 0].set_title(f"Sample of {n_show} Simulated Price Paths")
    axes[0, 0].set_xlabel("Day")
    axes[0, 0].set_ylabel("Price")
    axes[0, 0].legend()

    axes[0, 1].hist(end_prices, bins=80, color="steelblue", alpha=0.8)
    axes[0, 1].axvline(STRIKE, color="black", linestyle="--", label=f"Strike={STRIKE}")
    axes[0, 1].set_title("Distribution of End Prices")
    axes[0, 1].set_xlabel("Price at maturity")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()

    checkpoints = np.linspace(100, SIMS, 60).astype(int)
    running = [np.exp(-RATE * YEARS) * np.mean(call_payoffs[:n]) for n in checkpoints]
    axes[1, 0].plot(checkpoints, running, color="darkorange", label="Running estimate")
    axes[1, 0].axhline(bs_call, color="black", linestyle="--", label="Black-Scholes price")
    axes[1, 0].set_title("Convergence of Call Price Estimate")
    axes[1, 0].set_xlabel("Simulations used")
    axes[1, 0].set_ylabel("Call price")
    axes[1, 0].legend()

    x = np.linspace(0.5 * STRIKE, 1.5 * STRIKE, 200)
    axes[1, 1].plot(x, np.maximum(x - STRIKE, 0), label="Call payoff", color="green")
    axes[1, 1].plot(x, np.maximum(STRIKE - x, 0), label="Put payoff", color="red")
    axes[1, 1].axvline(STRIKE, color="black", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("Payoff Diagrams at Maturity")
    axes[1, 1].set_xlabel("Price at maturity")
    axes[1, 1].set_ylabel("Payoff")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("mc_simple.png", dpi=130)
    print("\nChart saved to mc_simple.png")


if __name__ == "__main__":
    main()
