import numpy as np


def divide(n, m):
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[0][0] = 1
    for i in range(1, n):
        dp[i, 1] = 1

    for i in range(n+1):
        t = min(i, m)
        for j in range(t+1):
            dp[i, j] = dp[i-1, j-1] + dp[i - j, j]

    return dp[n, m]


if __name__ == "__main__":
    print(divide(10, 2))