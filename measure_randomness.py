import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

def calculate_approx_entropy(U, m, r):
    """
    Calculate Approximate Entropy (ApEn) for a time series.

    Args:
        U (np.array): Time series data.
        m (int): Embedding dimension, length of patterns to compare.
        r (float): Tolerance, determines the threshold for accepting matches between sequences.

    Returns:
        float: The Approximate Entropy (ApEn) value.
    """
    N = len(U)  # Length of the time series

    def _phi(m):
        X = np.array(
            [U[i : i + m] for i in range(N - m + 1)]
        )  # Create overlapping subsequences of length m
        C = np.sum(np.abs(X[:, np.newaxis] - X).max(axis=2) <= r, axis=0) / (
            N - m + 1
        )  # Count matches within tolerance r
        return np.sum(np.log(C)) / (
            N - m + 1
        )  # Compute the correlation sum and average log likelihood

    return _phi(m) - _phi(
        m + 1
    )  # ApEn is the difference in log likelihood for m and m+1


def calculate_sample_entropy(U, m, r):
    """
    Calculate Sample Entropy (SampEn) for a time series.

    Args:
        U (np.array): Time series data.
        m (int): Embedding dimension, length of patterns to compare.
        r (float): Tolerance, determines the threshold for accepting matches between sequences.

    Returns:
        float: The Sample Entropy (SampEn) value.
    """
    N = len(U)  # Length of the time series

    def _phi(m):
        X = np.array(
            [U[i : i + m] for i in range(N - m)]
        )  # Create overlapping subsequences of length m
        C = np.sum(np.all(np.abs(X[:, np.newaxis] - X) <= r, axis=2), axis=0) / (
            N - m
        )  # Count matches, avoid self-matching
        return np.sum(np.log(C)) / (
            N - m
        )  # Compute the correlation sum and average log likelihood

    return _phi(m) - _phi(
        m + 1
    )  # SampEn is the difference in log likelihood for m and m+1


def calculate_max_apen(U, m, r_values):
    """
    Calculate the maximum Approximate Entropy (MaxApEn) over a range of r values.

    Args:
        U (np.array): Time series data.
        m (int): Embedding dimension, length of patterns to compare.
        r_values (np.array): Array of tolerance values to be tested.

    Returns:
        float: The maximum Approximate Entropy (MaxApEn) value.
    """
    apen_values = [
        calculate_approx_entropy(U, m, r) for r in r_values
    ]  # Compute ApEn for each r
    return max(apen_values)  # Return the maximum ApEn value


def bootstrap_max_apen(U, m, r_values, n_iter=100):
    """
    Perform bootstrap sampling to generate MaxApEn values from shuffled data.

    Args:
        U (np.array): Time series data.
        m (int): Embedding dimension, length of patterns to compare.
        r_values (np.array): Array of tolerance values to be tested.
        n_iter (int): Number of bootstrap iterations.

    Returns:
        list: List of MaxApEn values for each bootstrap iteration.
    """
    max_apen_bootstrap = []  # Initialize list to store bootstrap results
    for _ in range(n_iter):
        U_shuffled = np.random.permutation(U)  # Shuffle the time series
        max_apen_bootstrap.append(
            calculate_max_apen(U_shuffled, m, r_values)
        )  # Compute MaxApEn for shuffled data
    return max_apen_bootstrap  # Return list of MaxApEn values from bootstrapped samples


def calculate_pincus_index(U, m, r_values, n_iter=100):
    """
    Calculate the Pincus Index for a time series.

    Args:
        U (np.array): Time series data.
        m (int): Embedding dimension, length of patterns to compare.
        r_values (np.array): Array of tolerance values to be tested.
        n_iter (int): Number of bootstrap iterations.

    Returns:
        tuple: The Pincus Index, original MaxApEn, and list of bootstrap MaxApEn values.
    """
    max_apen_original = calculate_max_apen(U, m, r_values)  # MaxApEn for original data
    max_apen_bootstrap = bootstrap_max_apen(
        U, m, r_values, n_iter
    )  # MaxApEn for bootstrapped data
    pincus_index = max_apen_original / np.median(
        max_apen_bootstrap
    )  # Compute Pincus Index as a ratio
    return (
        pincus_index,
        max_apen_original,
        max_apen_bootstrap,
    )  # Return the Pincus Index and related values


# 각 윈도우의 이전 결과와 다음 결과의 차이를 계산하여 반환
def windowed_test(series, test_func, window_size=10, min_periods=1):
    """
    Apply a test function using a sliding window approach,
    comparing previous and next windows.

    Parameters:
    series (array-like): The input time series
    test_func (function): The test function to apply to each window
    window_size (int): The size of the sliding window
    min_periods (int): Minimum number of observations in window required to have a value

    Returns:
    numpy.array: Array of absolute differences between test results of adjacent windows
    """
    half_window = window_size // 2

    # Convert to pandas Series for easy rolling window operations
    series = pd.Series(series)

    # Calculate results for previous and next windows
    prev_results = series.rolling(window=half_window, min_periods=min_periods).apply(
        test_func
    )
    next_results = (
        series.rolling(window=half_window, min_periods=min_periods)
        .apply(test_func)
        .shift(-half_window)
    )

    # Calculate absolute differences
    differences = np.abs(next_results - prev_results)

    # Remove NaN values from the beginning and end
    return differences.fillna(method="ffill").fillna(0)


def mean_shift(series):
    """Detect shifts in the mean value."""
    return np.mean(series)


def std_deviation_change(series):
    """Detect changes in the standard deviation."""
    return np.std(series)


def slope_change(series):
    """Detect changes in the linear trend (slope)."""
    x = np.arange(len(series))
    slope, _, _, _, _ = stats.linregress(x, series)
    return slope


def sharpe_ratio(series):
    """Detect changes in the Sharpe ratio."""
    return np.mean(series) / np.std(series)



m = 2  # embedding dimension
w = 10  # window size

file_path = 'train_ratner_stock.csv'
df = pd.read_csv(file_path)
data_1 = df['Close Price'].pct_change().dropna()[0:88]
# data_1 = df['Close Price'].pct_change().dropna()[165:215]

r_values = np.linspace(0.01, 0.25, 20) * np.std(data_1)  # range of r values
pincus_index, max_apen_original, max_apen_bootstrap = calculate_pincus_index(
    data_1, m, r_values
)
print(f"Pincus Index: {pincus_index}")

apen_1 = calculate_approx_entropy(data_1.values, m, np.std(data_1) * 0.2)

samp_1 = calculate_sample_entropy(data_1.values, m, np.std(data_1.values) * 0.2)


print(apen_1, samp_1)


# Plotting the results
plt.hist(max_apen_bootstrap, bins=30, alpha=0.7, label="Bootstrap MaxApEn")
plt.axvline(max_apen_original, color="r", linestyle="--", label="Original MaxApEn")
plt.title(f"Pincus Index: {pincus_index:.2f}")
plt.xlabel("MaxApEn")
plt.ylabel("Frequency")
plt.legend()
plt.show()

