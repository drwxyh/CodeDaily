import numpy as np


def climbing_stairs(n):
    dp = np.zeros(n + 1, dtype=int)
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]


if __name__ == "__main__":
    num = 80
    print("There are "+str(climbing_stairs(num))+" methods to climb to " + str(num) + "th floor.")


