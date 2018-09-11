import numpy as np

INF = 65536

def dna_repair(S, P):
    L = len(S)
    N = len(P)
    purine = ['A', 'G', 'C', 'T']
    pfx = set()
    # 计算所有禁止字符串的前缀，用集合避免重复
    for p in P:
        for i in range(len(p) + 1):
            pfx.add(p[:i])
    pfx = list(pfx)
    K = len(pfx)
    # 状态转移矩阵
    state = np.zeros((K, 4), dtype=int)
    ng = [False] * K
    for i in range(K):
        # 如果禁止字符串等于某个前缀的后缀，那么该前缀状态就是禁止的
        for j in range(N):
            ng[i] |= len(P[j]) <= len(pfx[i]) and pfx[i].endswith(P[j])

        for j in range(4):
            # 反复删除第一个字符，直到等于某个新的前缀状态
            t = pfx[i] + purine[j]
            flag = 1
            new_state = 0
            while flag:
                for k in range(K):
                    if t == pfx[k]:
                        flag = 0
                        new_state = k
                        break
                if flag:
                    t = t[1:]
            state[i, j] = new_state

    dp = np.zeros((L + 1, K), dtype=int)
    dp[0, 0] = 1
    for l in range(L):
        for i in range(K):
            dp[l + 1][i] = INF

        for i in range(K):
            if ng[i]:
                continue
            for j in range(4):
                t = state[i, j]
                dp[l + 1, t] = min(dp[l + 1, t], dp[l, i] + (0 if S[l] == purine[j] else 1))

    res = INF
    for i in range(K):
        if ng[i]:
            continue
        res = min(res, dp[L][i])
    if res == INF:
        print(-1)
    else:
        print(res)


if __name__ == "__main__":
    dna_repair("AAAG", ["AAA", "AAG"])
