"""
dp[i][j]:是i个字符以状态j为结尾的情况数。
next[i][j]是当前状态为i，添加了字符j后的状态值。
"""
import numpy as np


def calculate_forbidden_string(N, K, S):
    purine = ['A', 'G', 'C', 'T']
    # 预处理状态的转移
    # 生成字符串的后缀和禁止字符串的前缀匹配，除去禁止状态共有 K 个状态
    # next[i, j]: 表示当前状态为 i 的情况下，添加字符 j 后的状态值
    next = np.zeros((K, 4), dtype=int)
    for i in range(K):
        for j in range(4):
            t = S[0:i] + purine[j]
            while t not in S:
                t = t[1:]
            next[i, j] = len(t)

    # dp[i, j]: 表示长度为 i 且以 j 状态为结尾的字符串数
    dp = np.zeros((N + 1, K), dtype=int)
    dp[0, 0] = 1

    for l in range(N):
        for i in range(K):
            for j in range(4):
                # i 添加 j 后的新状态
                t = next[i, j]
                if t == K:
                    continue
                # 对于长度为 l，以 i 状态结尾的字符串而言，添加字符 j 会产生 dp[l, i] 个长度为 l+1 且以状态 next[i, j] 结尾的字符串
                dp[l + 1, t] += dp[l, i]

    res = 0
    for i in range(K):
        res += dp[N, i]
    print(res % 10009)


if __name__ == "__main__":

    calculate_forbidden_string(3, 2, 'AT')
