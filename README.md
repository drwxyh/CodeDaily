### 180821 可达矩阵

**详细描述：**给定 n 个顶点，边长为 1 的有向图的邻接矩阵。求这个图里长度为 k 的路径的总数，路径中同一条边允许通过多次。

**限制条件：**

- $$1 \le n \le 100$$
- $$1 \le k \le 10^9$$

**输入示例：**

![](https://ws3.sinaimg.cn/large/006tNbRwgy1fuiflwijubj30ng07cq3t.jpg)

**代码实现：**

```python
import numpy as np


def find_k_length_path(vertex_num, path_length, adjacency_matrix):
    """
    :param vertex_num: number of vertices in this digraph
    :param path_length: the given path length
    :param adjacency_matrix: adjacency matrix of the digraph
    :return: the number of paths with the given length in this graph
    """
    n, m = vertex_num, path_length
    if m < 1:
        print("Error, The path length is not valid!")
    else:
        arrival_matrix = adjacency_matrix
        for x in range(0, path_length):
            arrival_matrix = np.dot(adjacency_matrix, adjacency_matrix)
        res = 0
        for i in range(0, n):
            for j in range(0, n):
                res += arrival_matrix[i][j]
        print("There are " + str(res) + " paths in this digraph with length " + str(m) + ".")


if __name__ == '__main__':
    with open('180822') as fp:
        vertex_num, path_length = [int(x) for x in fp.readline().split()]
        temp_matrix = list()
        for line in fp.readlines():
            temp_matrix.append([int(x) for x in line.split()])

    adjacency_matrix = np.array(temp_matrix, dtype=int)
    find_k_length_path(vertex_num, path_length, adjacency_matrix)

```



### 180822 回文子串

**详细描述：**Given a string **s**, find the longest palindromic substring in **s**. You may assume that the maximum length of **s** is 1000.

**输入示例：**

![](https://ws1.sinaimg.cn/large/006tNbRwgy1furqcqkkkdj30o806emx1.jpg)

**代码实现：**

```python
def longestPalindrome(s):
    length = len(s)
    res = ""
    for i in range(length):
        tmp = helper(s, i, i)
        if len(res) < len(tmp):
            res = tmp
        tmp = helper(s, i, i + 1)
        if len(res) < len(tmp):
            res = tmp
    return res

def helper(s, l, r):
    while l >= 0 and r < len(s) and s[l] == s[r]:
        l -= 1
        r += 1
    return s[l + 1:r]
```



### 180823 可图理论

**详细描述：** 未名湖附近共有n 个大小湖泊 $$L_1, L_2, ..., L_n$$（其中包括未名湖），每个湖泊Li 里住着一只青蛙Fi（1≤i≤n）。如果湖泊Li 和Lj 之间有水路相连，则青蛙 Fi 和 Fj 互称为邻居。现在已知每只青蛙的邻居数目$$x_1, x_2, ..., x_n$$，请你给出每两个湖泊之间的相连关系。

**输入描述：**第一行是测试数据的组数t（0 ≤ t ≤ 20）。每组数据包括两行，第一行是整数n（2 ≤ n ≤ 10），

第二行是n 个整数，x1, x2,..., xn（0 ≤ xi < n）。

**输出描述：**对输入的每组测试数据，如果不存在可能的相连关系，输出"NO"。否则输出"YES"，并用n*n的矩阵表示湖泊间的相邻关系，即如果湖泊i 与湖泊j 之间有水路相连，则第i 行的第j 个数字为1，否则为0。每两个数字之间输出一个空格。如果存在多种可能，只需给出一种符合条件的情形。相邻两组测试数据之间输出一个空行。

**输入示例：**

![](https://ws2.sinaimg.cn/large/006tNbRwgy1furqn7nbjkj30yv0e1aa4.jpg)

**实现代码：**

```python
import numpy as np


class Lake:
    def __init__(self, id, degree):
        self.id = id
        self.degree = degree


def sort_arr(arr, start):
    temp = list()
    res = list()
    for i in range(start, len(arr)):
        temp.append(arr[i])
    temp = sorted(temp, key=lambda x: x.degree, reverse=True)
    for i in range(0, start):
        res.append(arr[i])
    for x in temp:
        res.append(x)
    return res


def is_lake_exist(lake_num, degree):
    lake_list = list()
    for i in range(0, lake_num):
        lake_list.append(Lake(i, degree[i]))
    adjacency_matrix = np.zeros((lake_num, lake_num), dtype=int)
    flag = 1
    for i in range(0, lake_num):
        lake_list = sort_arr(lake_list, i)

        u = lake_list[i].id
        d = lake_list[i].degree

        if d > lake_num - i - 1:
            flag = 0
            break

        for k in range(1, d + 1):
            v = lake_list[i + k].id
            if lake_list[i + k].degree <= 0:
                flag = 0
                break
            lake_list[i + k].degree -= 1
            adjacency_matrix[u][v] = adjacency_matrix[v][u] = 1

    if flag:
        print('Yes')
        print(adjacency_matrix)
    else:
        print('No')


if __name__ == "__main__":
    with open('1.2') as fp:
        sample_num = int(fp.readline())
        while sample_num > 0:
            lake_num = int(fp.readline())
            degree = [int(x) for x in fp.readline().split()]
            is_lake_exist(lake_num, degree)
            sample_num -= 1
```



### 180824 铺砖问题

**详细描述：**给定 n\*m 的格子，每个格子被染成黑色或者白色。现在要用 1\*2 的砖块覆盖这些格子，要求块与块之间互相不重叠，且覆盖了所有白色的格子，但不覆盖任意一个黑色的格子。求一共有多少种覆盖方法。

**限制条件：**

- $ 1 \le n \le15 $
- $ 1 \le m \le 15 $
- $ 2 \le M \le 10^9 $

**输入示例：**

![](https://ws4.sinaimg.cn/large/006tNbRwgy1furrgnkegbj30sn0fiabh.jpg)

**代码实现：**暂未使用状态压缩，稍后实现

```python
import numpy as np


def flooring(x, y, used):
    if y == m:
        return flooring(x + 1, 0, used)
    if x == n:
        return 1
    if used[x][y] or color[x][y]:
        return flooring(x, y + 1, used)

    res = 0
    used[x][y] = 1
    if y + 1 < m and not color[x][y + 1] and not used[x][y + 1]:
        used[x][y + 1] = 1
        res += flooring(x, y + 1, used)
        used[x][y + 1] = 0

    if x + 1 < n and not color[x + 1][y] and not used[x + 1][y]:
        used[x + 1][y] = 1
        res += flooring(x, y + 1, used)
        used[x + 1][y] = 0
    used[x][y] = 0

    return res


if __name__ == '__main__':

    with open('180824') as fp:
        n, m = [int(x) for x in fp.readline().split()]
        graph = list()
        for line in fp.readlines():
            graph.append([x for x in line.split()])

    used = np.zeros((n, m), dtype=int)
    color = np.zeros((n, m), dtype=int)
    for i in range(0, n):
        for j in range(0, m):
            if graph[i][j] == 'x':
                color[i][j] = 1
    res = flooring(0, 0, used)
    print(res)
```



### 180825 构造倍数

**详细描述：**编写程序，实现：给定一个自然数N，N 的范围为[0, 4999]，以及M 个不同的十进制数字X1,X2, …, XM（至少一个，即M≥1），求N 的最小的正整数倍数，满足：N 的每位数字均为X1, X2, …, XM 中的一个。

**输入描述：**输入文件包含多个测试数据，测试数据之间用空行隔开。每个测试数据的格式为：第1 行为自然数N；第2 行为正整数M；接下来有M 行，每行为一个十进制数字，分别为X1, X2, …, XM。

**输出描述：**对输入文件中的每个测试数据，输出符合条件的N 的倍数；如果不存在这样的倍数，则输出 0。

**输入示例：**

样例输入						样例输出

22							110

3

7

0

1

**代码实现：**

```python
import numpy as np
from collections import namedtuple
from queue import Queue


def find_the_smallest_multiple():
    q = Queue()
    visit = np.zeros(n, dtype=int)
    Node = namedtuple("Node", ["value", "remainder"])
    s = Node(0, 0)
    q.put(s)

    while not q.empty():
        t = q.get()

        for x in nums:
            k = t.remainder * 10 + x
            if not k:
                continue
            else:
                k %= n
                if k == 0:
                    print(t.value * 10 + x)
                    return
                if visit[k] == 0:
                    visit[k] = 1
                    new_node = Node(t.value * 10 + x, k)
                    q.put(new_node)
    print(0)


if __name__ == "__main__":
    with open("180825") as fp:
        n = int(fp.readline().strip())
        m = int(fp.readline().strip())
        nums = sorted([int(x) for x in fp.readlines()])

        find_the_smallest_multiple()
        
```



### 180826 迷宫逃生

**详细描述：**一只小狗在一个古老的迷宫里找到一根骨头，当它叼起骨头时，迷宫开始颤抖，它感觉到地面开始下沉。它才明白骨头是一个陷阱，它拼命地试着逃出迷宫。迷宫是一个N°¡M 大小的长方形，迷宫有一个门。刚开始门是关着的，并且这个门会在第T 秒钟开启，门只会开启很短的时间（少于一秒），因此小狗必须恰好在第T 秒达到门的位置。每秒钟，它可以向上、下、左或右移动一步到相邻的方格中。但一旦它移动到相邻的方格，这个方格开始下沉，而且会在下一秒消失。所以，它不能在一个方格中停留超过一秒，也不能回到经过的方格。小狗能成功逃离吗？请你帮助他。

**输入描述：**输入文件包括多个测试数据。每个测试数据的第一行为三个整数：N M T，（1<N, M<7；0<T<50），分别代表迷宫的长和宽，以及迷宫的门会在第T 秒时刻开启。接下来N 行信息给出了迷宫的格局，每行有M 个字符，这些字符可能为如下值之一：

X: 墙壁，小狗不能进入 S: 小狗所处的位置

D: 迷宫的门 			. : 空的方格

**输出描述：**对每个测试数据，如果小狗能成功逃离，则输出"YES"，否则输出"NO"。

**输入示例：**

![](https://ws1.sinaimg.cn/large/006tNbRwgy1fv1gdwkxpqj30i103ta9v.jpg)

**代码实现：**

```python
def dfs(x, y, t):
    global flag
    if (x, y) == door_pos and t == time:
        flag = 1
        return

    temp = (time - t) - abs(door_pos[0] - x) - abs(door_pos[1] - y)
    if temp < 0 or temp % 2 == 1:
        return

    for i in range(4):
        u = x + dir[i][0]
        v = y + dir[i][1]
        if 0 <= u < row and 0 <= v < col and maze_map[u][v] != 'X':
            maze_map[u][v] == 'X'
            dfs(u, v, t + 1)
            if flag == 1:
                return
            maze_map[u][v] == '.'


if __name__ == "__main__":
    with open("180826") as fp:
        row, col, time = [int(x) for x in fp.readline().split()]
        maze_map = list()
        for line in fp.readlines():
            tmp = list()
            for c in line.strip():
                tmp.append(c)
            maze_map.append(tmp)

    for i in range(row):
        for j in range(col):
            if maze_map[i][j] == 'S':
                start_pos = (i, j)
            elif maze_map[i][j] == 'D':
                door_pos = (i, j)
            else:
                pass
    flag = 0
    dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    dfs(start_pos[0], start_pos[1], 0)
    if flag:
        print("Yes, the lucky dog can get out of the maze.")
    else:
        print("No, the poor dog can't get out of the maze.")

```



### 180827 背包问题

**详细描述：**假如你是一个小偷，背着一个可以装下 4 磅东西的背包，可以偷的东西如下，音响4磅3000元，笔记本电脑3磅2000美元，吉他1磅1500元。为了让盗窃的商品价值最高，应该如何选择商品？

**代码实现：**

```python
import numpy as np


def knapsack(capacity, item_num, item_cost, item_size):
    """
    :param capacity: the capacity of the knapsack
    :param item_num: the number of the items to be chosen
    :param item_cost: the value of each item
    :param item_size: the size of each item
    :return: the maximum value of the items that the knapsack can load
    """
    A = np.zeros((item_num + 1, capacity + 1), dtype=int)
    for i in range(0, item_num + 1):
        A[i][0] = 0
    for j in range(0, capacity + 1):
        A[0][j] = 0
    for i in range(1, item_num + 1):
        for j in range(1, capacity + 1):
            if j < item_size[i - 1]:
                A[i][j] = A[i - 1][j]
            else:
                A[i][j] = max(A[i - 1][j], A[i - 1][j - item_size[i - 1]] + item_cost[i - 1])

    print("The maximum value of the items that this knapsack can load is : " + str(A[item_num][capacity]) + '$ .')


if __name__ == "__main__":
    with open("180827") as fp:
        capacity = int(fp.readline())
        item_num = int(fp.readline())
        item_cost = [int(x) for x in fp.readline().split()]
        item_size = [int(x) for x in fp.readline().split()]

    knapsack(capacity, item_num, item_cost, item_size)
```



### 180828 油田勘测

**详细描述：**GeoSurvComp 地质探测公司负责探测地下油田。每次GeoSurvComp 公司都是在一块长方形的土地上来探测油田。在探测时，他们把这块土地用网格分成若干个小方块，然后逐个分析每块土地，用探测设备探测地下是否有油田。方块土地底下有油田则称为pocket，如果两个pocket相邻，则认为是同一块油田，油田可能覆盖多个pocket。你的工作是计算长方形的土地上有多少个不同的油田。

**输入描述：**输入文件中包含多个测试数据，每个测试数据描述了一个网格。每个网格数据的第一行为两个整数：m n，分别表示网格的行和列；如果m = 0，则表示输入结束，否则1≤m≤100，1 ≤n≤100。接下来有m 行数据，每行数据有n 个字符（不包括行结束符）。每个字符代表一个小方块，如果为“*”，则代表没有石油，如果为“@”，则代表有石油，是一个pocket。

**输出描述：**对输入文件中的每个网格，输出网格中不同的油田数目。如果两块不同的pocket 在水平、垂直、或者对角线方向上相邻，则被认为属于同一块油田。每块油田所包含的pocket 数目不会超过100。

**输入示例：**

**代码实现：**

```python
def find_oil():
    for i in range(row):
        for j in range(col):
            if field[i][j] == '@':
                return i, j


def dfs(x, y):
    field[x][y] = '.'
    dir = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i in range(8):
        u = x + dir[i][0]
        v = y + dir[i][1]
        if 0 <= u < row and 0 <= v < col and field[u][v] == '@':
            dfs(u, v)


if __name__ == "__main__":
    with open("180828") as fp:
        row, col = [int(x) for x in fp.readline().split()]
        field = list()
        for line in fp.readlines():
            tmp = list()
            for c in line.strip():
                tmp.append(c)
            field.append(tmp)

    cnt = 0
    res = find_oil()
    while res:
        cnt += 1
        dfs(res[0], res[1])
        res = find_oil()
    if cnt == 1:
        print("There is only one oil field.")
    else:
        print("There are {} oil fields.".format(cnt))
        
```



### 180829 TSP问题

**详细描述：**给定一个 n 个顶点组成的带权有向图的距离矩阵 $d(i, j)$ （INF 表示没有边）。要求从顶点 0 出发，经过每个顶点恰好一次再回到顶点 0，问所经过的总权重的最小值是多少？

**限制条件：**

- $2 \le n \le 15$
- $ 0 \le d(i, j) \le 1000 $

**输入示例：**

![](https://ws3.sinaimg.cn/large/006tNbRwgy1furr1mcvgnj30sb0c574r.jpg)

**代码实现：**

```python
import numpy as np

INF = 65532

if __name__ == "__main__":
    with open("180829") as fp:
        node_num = int(fp.readline().strip())
        graph = np.zeros((node_num, node_num), dtype=int)
        for i in range(node_num):
            for j in range(node_num):
                if i == j:
                    graph[i, j] = 0
                else:
                    graph[i, j] = INF
        for line in fp.readlines():
            u, v, w = [int(x) for x in line.split()]
            graph[u, v] = w

    dp = np.zeros((node_num, 1 << (node_num - 1)), dtype=int)
    for i in range(node_num):
        for j in range(0, 1 << (node_num - 1)):
            dp[i, j] = INF

    for i in range(0, node_num):
        dp[i, 0] = graph[i, 0]

    for i in range(1, 1 << (node_num - 1)):
        for j in range(0, node_num):
            for k in range(1, node_num):
                if 1 << (k - 1) & i:
                    dp[j, i] = min(dp[j, i], graph[j, k] + dp[k, i - (1 << (k - 1))])

    print("The shortest TSP distance is: " + str(dp[0][(1 << node_num - 1) - 1]) + ".")

```



### 180830 农田灌溉

**详细描述：**有一大片农田需要灌溉。农田是一个长方形，被分割成许多小的正方形。每个正方形中都安装了水管。不同的正方形农田中可能安装了不同的水管。一共有11 种水管，分别用字母 A～K 标明，如下图所示：

![](https://ws1.sinaimg.cn/large/006tKfTcgy1fruevegd1jj30l3093weq.jpg)

**输入描述：**

输入文件中包含多个测试数据。每个测试数据的第1 行为两个整数M 和N，表示农田中有M行，每行有N 个正方形。接下来有M 行，每行有N 个字符。字符的取值为'A'～'K'，表示对应正方形农田中水管的类型。当M 或N 取负值时，表示输入文件结束；否则M 和N 的值为正数，且其取值范围是1≤M, N≤50。

**输出描述：**

对输入文件中的每个测试数据所描述的农田，输出占一行，为求得的所需水源数目的最小值。

**输入示例：**

![](https://ws4.sinaimg.cn/large/006tNbRwgy1furrmfp29uj30je087jr7.jpg)

**代码实现：**

```python
def find_non_irrigated_grid(graph, row, col):
    for i in range(row):
        for j in range(col):
            if graph[i][j] != ".":
                return i, j
    return None


def is_connected(a, b, d):
    flag = 0
    if d == 0:
        if grid[a][1] == '1' and grid[b][3] == '1':
            flag = 1
    elif d == 1:
        if grid[a][3] == '1' and grid[b][1] == '1':
            flag = 1
    elif d == 2:
        if grid[a][2] == '1' and grid[b][0] == '1':
            flag = 1
    else:
        if grid[a][0] == '1' and grid[b][2] == '1':
            flag = 1
    return flag

def dfs(x, y):
    temp = graph[x][y]
    graph[x][y] = '.'
    for i in range(4):
        u = x + dir[i][0]
        v = y + dir[i][1]
        if u < 0 or u >= row or v < 0 or v >= col:
            continue
        if graph[u][v] == '.':
            continue
        if is_connected(temp, graph[u][v], i):
            dfs(u, v)

if __name__ == "__main__":
    with open("180830") as fp:
        row, col = [int(x) for x in fp.readline().split()]
        graph = list()
        for data in fp.readlines():
            line = [x for x in data]
            graph.append(line)

    dir = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    grid = {'A': '1001', 'B': '1100', 'C': '0011', 'D': '0110', 'E': '1010', 'F': '0101', 'G': '1101', 'H': '1011',
            'I': '0111', 'J': '1110', 'K': '1111'}

    res = find_non_irrigated_grid(graph, row, col)
    cnt = 0
    while res:
        cnt += 1
        x, y = res
        dfs(x, y)
        res = find_non_irrigated_grid(graph, row, col)
    if cnt == 1:
        print("Totally one water source is needed.")
    else:
        print("Totally " + str(cnt) + " headwaters are needed.")
```



### 180831 爬楼组合

**详细描述：**一个人每次只能走一层楼梯或者两层楼梯，问走到第80层楼梯一共有多少种方法？

**代码实现：**

```python
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

```



### 180901 Gnome Tetravex 

**详细描述：**哈特近来一直在玩有趣的Gnome Tetravex 游戏。在游戏开始时，玩家会得到 n×n（n≤5）个正方形。每个正方形都被分成4 个标有数字的三角形（数字的范围是0 到9）。这四个三角形分别被称为“左三角形”、“右三角形”、“上三角形”和“下三角形”。例如，图 (a) 是2×2 的正方形的一个初始状态。

![](https://ws3.sinaimg.cn/large/006tNbRwgy1futrx6vzwaj30ft076q35.jpg)

玩家需要重排正方形，到达目标状态。在目标状态中，任何两个相邻正方形的相邻三角形上的数字都相同。图 (b)是一个目标状态的例子。看起来这个游戏并不难。但是说实话，哈特并不擅长这种游戏，他能成功地完成最简单的游戏，但是当他面对一个更复杂的游戏时，他根本无法找到解法。某一天，当哈特玩一个非常复杂的游戏的时候，他大喊到：“电脑在耍我！不可能解出这个游戏。”对于这样可怜的玩家，帮助他的最好方法是告诉他游戏是否有解。如果他知道游戏是无解的，他就不需要再把如此多的时间浪费在它上面了。

**输入描述：**输入文件中包含多个测试数据，每个测试数据描述了一个Gnome Tetravex 游戏。每个游戏的第1 行为一个整数n，1≤n≤5，表示游戏的规模，该游戏中有n×n 个正方形。接下来有n×n 行，描述了每个正方形中4 个三角形中的数字。每一行为4 个整数，依次代表上三角形、右三角形、下三角形和左三角形中的数字。

**输出描述：**对输入文件中的每个游戏，你必须判断该游戏是否有解。如果该游戏有解，输出"Possible"，否则输出"Impossible"。

**输入示例：**

![](https://ws3.sinaimg.cn/large/006tNbRwgy1futs1z3ffaj31fw0n2wep.jpg)

**代码实现：**

```python
from collections import namedtuple

def dfs(cur):
    if cur == side_length ** 2:
        return True

    for i in range(len(square_kind)):
        square = square_kind[i]
        if square_dict[square] == 0:
            continue

        if cur > 0 and cur % side_length:
            if square_kind[board[cur - 1]].r != square.l:
                continue

        if cur >= side_length:
            if square_kind[board[cur - side_length]].d != square.u:
                continue

        board[cur] = i
        square_dict[square] -= 1
        if dfs(cur + 1):
            return True
        else:
            square_dict[square] += 1

    return False

if __name__ == "__main__":
    Square = namedtuple("Square", ['u', 'r', 'd', 'l'])
    square_list = list()
    with open("180901") as fp:
        side_length = int(fp.readline())
        for data in fp.readlines():
            u, r, d, l = [int(x) for x in data.split(' ')]
            square_list.append(Square(u, r, d, l))

    square_dict = dict()
    for square in square_list:
        if square not in square_dict.items():
            square_dict[square] = 1
        else:
            square_dict[square] += 1

    square_kind = [x for x in square_dict.keys()]
    board = [-1 for i in range(side_length ** 2)]
    if dfs(0):
        print("Possible.")
    else:
        print("Impossible.")

```



### 180902 划分数

**详细描述：**有 n 个无区别的物品，将它们划分成不超过 m 组，求可以划分的方法数。

**限制条件：**

- $$ 1 \le m \le n \le 1000 $$

**代码实现：**

```python
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
```



### 180903 蛇的爬动

**详细描述：**在冬天，天气最恶劣的时期，蛇待在洞穴里冬眠。当春天来临的时候，蛇苏醒了，爬到洞穴的出口，然后爬出来，开始它的新生活。蛇的洞穴象一个迷宫，可以把它想象成一个由n×m 个正方形区域组成的长方形。每个正方形区域要么被石头占据了，要么是一块空地，蛇只能在空地间爬动。洞穴的行和列都是有编号的，行和列的编号从1 开始计起，且出口在(1,1)位置。蛇的身躯，长为L，用一块连一块的形式来表示。假设用B1(r1, c1)，B2(r2, c2)，...，BL(rL, cL)表示它的L 块身躯，其中，Bi 与Bi+1（上、下、左或右）相邻，i = 1, ..., L-1；B1 为蛇头，BL为蛇尾。为了在洞穴中爬动，蛇选择与蛇头（上、下、左或右）相邻的一个空的正方形区域，这个区域既没有被石头占据，也没有被它的身躯占据。当蛇头移动到这个空地，这时，它的身躯中其他每一块都移动它前一块身躯之前所占据的空地上。

![mark](http://oz9e7ei0y.bkt.clouddn.com/blog/180902/H2hcHjF06E.png?imageslim)

例如，图 (a) 所示的洞穴（带有深色阴影的方格为蛇头，带有浅色阴影的方格为蛇身，黑色方格为石头）中，蛇的初始位置为：B1(4,1)，B2(4,2)，B3(3,2)和B4(3,1)。在下一步，蛇头只能移动到B1'(5,1)位置。蛇头移动到B1'(5,1)位置后，则B2 移动到B1 原先所在的位置，B3 移动到B2 原先所在的位置，B4 移动到B3 原先所在的位置。因此移动一步后，蛇的身躯位于B1(5,1)，B2(4,1)，B3(4,2)和B4(3,2)，如图 (b) 所示。

**输入描述：**输入文件包含多个测试数据。每个测试数据的第1 行为3 个整数：n，m 和L，1≤n, m≤20，2≤L≤8，分别代表洞穴的行、列，以及蛇的长度；接下来有L 行，每行有一对整数，分别表示行和列，代表蛇的每一块身躯的初始位置，依顺序分别为B1(r1, c1)～BL(rL, cL)；接下来一行包含一个整数K，表示洞穴中石头的数目；接下来的K 行，每行包含一对整数，分别表示行和列，代表每一块石头的位置。

**输出描述：**r如果有解，输出蛇爬到洞穴出口所需的最少步数，如果没有解，则输出-1。

**输出示例：**

![mark](http://oz9e7ei0y.bkt.clouddn.com/blog/180902/GaLeiD38ge.png?imageslim)

**代码实现：**

```python
from collections import namedtuple
from queue import Queue
import numpy as np


def cover_self(x, y, u, v, state):
    for i in range(1, snake_len):
        k = state & 3
        state >>= 2
        x += dirs[k][0]
        y += dirs[k][1]
        if u == x and v == y:
            return True
    return False


def bfs(x, y, state):
    q = Queue()
    t = Node(x, y, state, 0)
    q.put(t)
    visit[x][y][state] = 1

    while not q.empty():
        t = q.get()

        if t.x == 0 and t.y == 0:
            return t.step

        step = t.step + 1
        for i in range(0, 4):
            u = t.x + dirs[i][0]
            v = t.y + dirs[i][1]
            if u in range(row) and v in range(col) and graph[u][v] != 1 and not cover_self(t.x, t.y, u, v, t.state):
                state = t.state
                state <<= 2
                state |= redirs[i]
                state &= base
                if visit[u][v][state] == 0:
                    cur = Node(u, v, state, step)
                    q.put(cur)
                    visit[u][v][state] = 1

    return -1


if __name__ == "__main__":
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    redirs = [2, 3, 0, 1]
    Node = namedtuple("Node", ['x', 'y', 'state', 'step'])
    with open("180903") as fp:
        row, col, snake_len = [int(x) for x in fp.readline().split()]

        snake_data = list()
        for i in range(snake_len):
            snake_data.append([int(x) - 1 for x in fp.readline().split()])

        stone_num = int(fp.readline())
        stone_data = list()
        for i in range(stone_num):
            stone_data.append([int(x) - 1 for x in fp.readline().split()])

    graph = np.zeros((row, col), dtype=int)
    for i, j in stone_data:
        graph[i][j] = 1

    base = (1 << 2 * (snake_len - 1)) - 1
    visit = np.zeros((row, col, base + 1), dtype=int)

    ans = 1
    for i in range(snake_len - 1, 0, -1):
        for j in range(4):
            if snake_data[i][0] == dirs[j][0] + snake_data[i - 1][0] and snake_data[i][1] == dirs[j][1] + snake_data[i - 1][1]:
                ans <<= 2
                ans |= j
                ans &= base
                break

    a, b = snake_data[0]
    res = bfs(a, b, ans)
    if res == -1:
        print("This poor snake can't not get out of the hole.")
    else:
        print("The minimum number of the steps that the snake should take is " + str(res) + '.')
```



### 180904 禁止字符串

**详细描述：**考虑只由 ‘A’，’G‘，’C‘，’T‘ 四种字符串组成的 DNA 字符串，给定一个长度为 k 的字符串 s。请计算长度恰好为 n 且不包含 s 的字符串个数。

**限制条件：**

- $1 \le k \le 100$
- $1 \le n \le 10000$

**输入示例：**

![](https://ws2.sinaimg.cn/large/0069RVTdgy1fv1z2ncz5gj31cq0bcmxp.jpg)

**代码实现：**

```python

```



### 180905 跳马运动

**详细描述：**给定象棋棋盘上两个位置 a 和 b，编写程序，计算马从位置 a 调到位置 b 多需要的最小值。

**输入描述：**输入文件包含多个测试数据。每个测试数据占一行，为棋盘中的两个位置，用空格隔开。棋盘位置为两个字符组成的串，第1 个字符为字母a～h，代表棋盘中的列；第2 个字符为数字字符1～8，代表棋盘中的行。

**输出描述：**对输入文件中的每个测试数据，输出一行"To get from xx to yy takes n knight moves."，xx 和 yy 分别为输入数据中的两个位置，n 为求得的最少步数。

**输入示例：**

![](https://ws2.sinaimg.cn/large/006tNbRwgy1fux9bkausbj30ib02ft8j.jpg)

**代码实现：**

```python
import numpy as np
from collections import namedtuple
from queue import Queue


def knight_move(pos1, pos2):
    board = np.zeros((8, 14), dtype=int)
    dir = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
    Pos = namedtuple("Pos", ['x', 'y'])

    origin = Pos(int(pos1[1]) - 1, ord(pos1[0]) - 97)
    destination = Pos(int(pos2[1]) - 1, ord(pos2[0]) - 97)

    q = Queue()
    q.put(origin)
    step = 0
    board[origin.x, origin.y] = 1
    rear, cursor = origin, origin
    while not q.empty():
        tmp = q.get()
        if tmp == destination:
            print("To get from {} to {} takes {} knight moves.".format(pos1, pos2, step))
            return

        for i in range(8):
            u = tmp.x + dir[i][0]
            v = tmp.y + dir[i][1]
            if 0 <= u < 8 and 0 <= v < 14 and not board[u, v]:
                next_pos = Pos(u, v)
                q.put(next_pos)
                cursor = next_pos
                board[u, v] = 1

        if tmp == rear:
            rear = cursor
            step += 1

    return -1


if __name__ == "__main__":
    test_data = [("e2", "e4"), ("a1", "b2"), ("b2", "c3"), ("a1", "h8"), ("a1", "h7"), ("h8", "a1"), ("b1", "c3"),
                 ("f6", "f6")]
    for data in test_data:
        knight_move(*data)

```

### 180906 

**详细描述：**

**输入描述：**

**输出描述：**

**输入示例：**

**代码实现：**



### 180907 简单迷宫

**详细描述：**在本题中，你需要求解一个简单的迷宫问题：

1) 迷宫由6 行6 列的方格组成；

2) 3 堵长度为1～6 的墙壁，水平或竖直地放置在迷宫中，用于分隔方格；

3) 一个起始位置和目标位置。

如下面的图2.23 描述了一个迷宫。你需要找一条从起始位置到目标位置的最短路径。从任一个方格出发，只能移动到上、下、左、右相邻方格，并且没有被墙壁所阻挡。

**输入描述：**输入文件中包含多个测试数据。每个测试数据包含 5 行：第 1 行为两个整数，表示起始位置的列号和行号；第 2 行也是两个整数，为目标位置的列号和行号，列号和行号均从 1 开始计起。第 3～5 行均为 4 个整数，描述了 3 堵墙的位置；如果墙是水平放置的，则由左、右两个端点所在的位置指定，如果墙是竖直放置的，则由上、下两个端点所在的位置指定；端点的位置由两个整数表示，第 1 个整数表示端点距离迷宫左边界的距离，第 2 个整数表示端点距离迷宫上边界的距离。

假定这3 堵墙互相不会交叉，但两堵墙可能会相邻于某个方格的顶点。从起始位置到目标位置一定存在路径。下面的样例输入数据描述了下图 所示的迷宫。

![](https://ws2.sinaimg.cn/large/0069RVTdgy1fuye2tq5vij30hi076dfv.jpg)

**输出描述：**

对输入文件中的每个测试数据，输出从起始位置到目标位置的最短路径，最短路径由代表每一步移动的字符组成（'N'表示向上移动，'E'表示向右移动，'S'表示向下移动，'W'表示向左移动）。对某个测试数据，可能存在多条最短路径，对于这种情形，只需输入任意一条最短路径即可。

**输入示例：**

样例输入						样例输出

1 6							NEEESWW

2 6

0 0 1 0

1 5 1 6

1 5 3 5

**代码实现：**

```python
from collections import namedtuple
from queue import Queue

def basic_maze():
    q = Queue()
    q.put(origin)
    step = 0
    rear, cursor = origin, origin
    while not q.empty():
        tmp = q.get()

        if tmp == destination:
            print("The minimum number of steps that needs to take is: {}.".format(step))
            print_path()
            return

        for i in range(4):
            u = tmp.x + dir[i][0]
            v = tmp.y + dir[i][1]
            if 0 <= u < 6 and 0 <= v < 6 and not graph[u][v] and not is_cross_the_wall(tmp, Pos(u, v)):
                graph[u][v] = (tmp.x, tmp.y)
                next_node = Pos(u, v)
                q.put(next_node)
                cursor = next_node

        if tmp == rear:
            step += 1
            rear = cursor

def print_path():
    res = ""
    action = ['E', 'S', 'W', 'N']
    x = destination.x
    y = destination.y
    while graph[x][y] != "origin":
        for i in range(4):
            if x == graph[x][y][0] + dir[i][0] and y == graph[x][y][1] + dir[i][1]:
                res += action[i]
        x, y = graph[x][y][0], graph[x][y][1]
    print("The route is: {}.".format(res[::-1]))

def is_cross_the_wall(a, b):
    for i in range(3):
        if walls[i].s.x == walls[i].t.x:
            if walls[i].s.x == 0 or walls[i].s.x == 6:
                continue
            else:
                if a.y == b.y and a.y in range(walls[i].s.y, walls[i].t.y):
                    if (a.x < walls[i].s.x and b.x == walls[i].s.x) or (b.x < walls[i].s.x and a.x == walls[i].s.x):
                        return True

                else:
                    continue

        if walls[i].s.y == walls[i].t.y:
            if walls[i].s.y == 0 or walls[i].s.y == 6:
                continue
            else:
                if a.x == b.x and a.x in range(walls[i].s.x, walls[i].t.x):
                    if (a.y < walls[i].s.y and b.y == walls[i].s.y) or (b.y < walls[i].s.y and a.y == walls[i].s.y):
                        return True
                    else:
                        continue
    return False


if __name__ == "__main__":
    graph = [[0] * 6 for i in range(6)]
    dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    Pos = namedtuple("Pos", ['x', 'y'])
    Wall = namedtuple("Wall", ['s', 't'])
    walls = list()

    with open("180907") as fp:
        y, x = [int(x) - 1 for x in fp.readline().split()]
        origin = Pos(x, y)
        graph[origin.x][origin.y] = "origin"
        y, x = [int(x) - 1 for x in fp.readline().split()]
        destination = Pos(x, y)
        for i in range(3):
            pos1_y, pos1_x, pos2_y, pos2_x = [int(x) for x in fp.readline().split()]
            walls.append(Wall(Pos(pos1_x, pos1_y), Pos(pos2_x, pos2_y)))

    basic_maze()

```



### 180908

**详细描述：**

**输入描述：**

**输出描述：**

**输入示例：**

**代码实现：**



### 180909 递送情报

**详细描述：**战争年代，通讯员经常要穿过敌占区去送情报。在本题中，敌占区是一个由M×N 个方格组成的网格。通讯员要从初始方格出发，送情报到达目标方格。初始时，通讯员具有一定的体力。网格中，每个方格可能为安全的方格、布有敌人暗哨的方格、埋有地雷的方格以及被敌人封锁的方格。通讯员从某个方格出发，对上、右、下、左4 个方向上的相邻方格：如果某相邻方格为安全的方格，通讯员能顺利到达，所需时间为1 个单位时间、消耗的体力为1 个单位的体力；如果某相邻方格为敌人布置的暗哨，则通讯员要消灭该暗哨才能到达该方格，所需时间为2 个单位时间，消耗的体力为2 个单位的体力；如果某相邻方格为埋有地雷的方格，通讯员要到达该方格，则必须清除地雷，所需时间为3 个单位时间，消耗的体力为1 个单位的体力。另外，从目标方格的相邻方格到达目标方格，所需时间为1 个单位时间、消耗的体力为1 个单位的体力。本题要求的是：通讯员能否到达指定的目的地，如果能到达，所需最少的时间是多少（只需要保证到达目标方格时，通讯员的体力>0 即可）。

**输入描述：**输入文件中包含多个测试数据。每个测试数据的第1 行为2 个正整数：M 和N，2<M,N<20，分别表示网格的行和列。接下来有M 行，描述了网格；每行有 N 个字符，这些字符可以是'.'、'w'、'm'、'x'、'S'、'T'，分别表示安全的方格、布有敌人暗哨的方格、埋有地雷的方格、被敌人封锁的方格（通讯员无法通过）、通讯员起始方格、目标方格，输入数据保证每个测试数据中只有一个 'S' 和 'T'。表格中各重要符号的含义及参数如表2.2 所示。每个测试数据的最后一行为一个整数P，表示通讯员初始时的体力。M = N = 0 表示输入结束。

![](https://ws2.sinaimg.cn/large/006tNbRwgy1fuzmkszm9bj30e902y747.jpg)

**输出描述：**对输入文件中的每个测试数据，如果通讯员能在体力消耗前到达目标方格，则输出所需的最少时间；如果通讯员无法到达目标方格（即体力消耗完毕或没有从起始方格到目标方格的路径），则输出No。

**输入示例：**

![](https://ws4.sinaimg.cn/large/006tNbRwgy1fv1equ1dqrj30j305rweb.jpg)

**代码实现：**

```python
from collections import namedtuple
from queue import Queue
import numpy as np


def dead_drop():
    q = Queue()
    start = Node(start_pos[0], start_pos[1], 0, hit_point)
    q.put(start)
    visit[start.x][start.y] = 1

    while not q.empty():
        tmp = q.get()

        if (tmp.x, tmp.y) == to_pos:
            print("The minimum number of time that the spy need to fulfill the task is {}.".format(tmp.t))
            return

        for i in range(4):
            u = tmp.x + dir[i][0]
            v = tmp.y + dir[i][1]
            if 0 <= u < row and 0 <= v < col and not visit[u][v] and graph[u][v] != 'x':
                if (graph[u][v] == '.' or graph[u][v] == 'T') and tmp.h - 1 > 0:
                    new_node = Node(u, v, tmp.t + 1, tmp.h - 1)
                elif graph[u][v] == 'w' and tmp.h - 2 > 0:
                    new_node = Node(u, v, tmp.t + 2, tmp.h - 2)
                elif graph[u][v] == 'm' and tmp.h - 1 > 0:
                    new_node = Node(u, v, tmp.t + 3, tmp.h - 1)
                else:
                    continue
                q.put(new_node)
                visit[u][v] = 1

    print("Unfortunately，the poor spy can't fulfill his task.")

if __name__ == "__main__":
    Node = namedtuple("Node", ['x', 'y', 't', 'h'])
    dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    with open("180909") as fp:
        row, col = [int(x) for x in fp.readline().split()]
        graph = list()
        visit = np.zeros((row, col), dtype=int)
        for i in range(row):
            line_data = fp.readline().strip()
            tmp = list()
            for data in line_data:
                tmp.append(data)
            graph.append(tmp)
        hit_point = int(fp.readline())
        for i in range(row):
            for j in range(col):
                if graph[i][j] == 'S':
                    start_pos = (i, j)
                if graph[i][j] == 'T':
                    to_pos = (i, j)

        dead_drop()

```



### 180911 最短路径（一）

**详细描述：**利用 Dijkstra 算法求图中顶点 0 到其他顶点的最短路径长度，并输出对应的最短路径。

![](https://ws2.sinaimg.cn/large/006tNbRwgy1fv1b0kbkdhj30lb05v3yn.jpg)

**输入描述：**：首先输入顶点个数n，然后输入每条边的数据。每条边的数据格式为：u v w，分别表示这条边的起点、终点和边上的权值。顶点序号从0 开始计起。

**代码实现：**

```python
import numpy as np

INF = 65536

def dijkstra(origin):
    s = np.zeros(node_num, dtype=int)
    s[origin] = 1
    path = np.zeros(node_num, dtype=int)
    dist = np.zeros(node_num, dtype=int)
    for i in range(node_num):
        dist[i] = edge[origin, i]
        if i != origin and dist[i] != INF:
            path[i] = origin
        else:
            path[i] = -1

    for i in range(node_num - 1):
        min_edge = INF
        u = origin
        for j in range(node_num):
            if not s[j] and dist[j] < min_edge:
                u = j
                min_edge = dist[j]

        s[u] = 1
        for k in range(node_num):
            if not s[k] and edge[u][k] != INF and dist[u] + edge[u, k] < dist[k]:
                path[k] = u
                dist[k] = dist[u] + edge[u, k]

    for i in range(node_num):
        if i != origin:
            shortest_path = str(i)
            t = i
            while path[t] != -1:
                shortest_path += '-' + str(path[t])
                t = path[t]

            shortest_path = shortest_path[::-1]
            res = '{} -> {}, the shortest distance is : {:<2} and the shortest path is : {}'
            print(res.format(origin, i, dist[i], shortest_path))


if __name__ == "__main__":
    with open("180911") as fp:
        node_num = int(fp.readline().strip())
        edge = [[INF] * node_num for i in range(node_num)]
        edge = np.array(edge)
        for line in fp.readlines():
            u, v, w = [int(x) for x in line.split()]
            edge[u, v] = w
        for i in range(node_num):
            edge[i, i] = 0

        dijkstra(1)

```



### 180912 多米诺骨牌

**详细描述：**你知道多米诺骨牌除了用来玩多米诺骨牌游戏外，还有其他用途吗？多米诺骨牌游戏：取一些多米诺骨牌，竖着排成连续的一行，两张骨牌之间只有很短的空隙。如果排列得很好，当你推倒第1 张骨牌，会使其他骨牌连续地倒下（这就是短语“多米诺效应”的由来）。然而当骨牌数量很少时，这种玩法就没多大意思了，所以一些人在80 年代早期开创了另一个极端的多米诺骨牌游戏：用上百万张不同颜色、不同材料的骨牌拼成一幅复杂的图案。他们开创了一种流行的艺术。在这种骨牌游戏中，通常有多行骨牌同时倒下。你的任务是编写程序，给定这样的多米诺骨牌游戏，计算最后倒下的是哪一张骨牌、在什么时间倒下。这些多米诺骨牌游戏包含一些“关键牌”，他们之间由一行普通骨牌连接。当一张关键牌倒下时，连接这张关键牌的所有行都开始倒下。当倒下的行到达其他还没倒下的关键骨牌时，则这些关键骨牌也开始倒下，同样也使得连接到它的所有行开始倒下。每一行骨牌可以从两个端点中的任何一张关键牌开始倒下，甚至两个端点的关键牌都可以分别倒下，在这种情形下，该行最后倒下的骨牌为中间的某张骨牌。假定骨牌倒下的速度一致。

**输入描述：**输入文件包含多个测试数据，每个测试数据描述了一个多米诺骨牌游戏。每个测试数据的第 1 行为两个整数：n 和m，n 表示关键牌的数目，1≤n<500；m 表示这n 张牌之间用m 行普通骨牌连接。n 张关键牌的编号为1～n。每两张关键牌之间至多有一行普通牌，并且多米诺骨牌图案是连通的，也就是说从一张骨牌可以通过一系列的行连接到其他每张骨牌。接下来有m 行，每行为3 个整数：a、b 和t，表示第a 张关键牌和第b 张关键牌之间有一行普通牌连接，这一行从一端倒向另一端需要t 秒。每个多米诺骨牌游戏都是从推倒第1 张关键牌开始的。输入文件最后一行为n = m = 0，表示输入结束。

**输出描述：**对输入文件中的每个测试数据，首先输出一行"System #k"，其中k 为测试数据的序号；然后再输出一行，首先是最后一块骨牌倒下的时间，精确到小数点后一位有效数字，然后是最后倒下骨牌的位置，这张最后倒下的骨牌要么是关键牌，要么是两张关键牌之间的某张普通牌。输出格式如样例输出所示。如果存在多个解，则输出任意一个。每个测试数据的输出之后输出一个空行。

**输入示例：**

![](https://ws3.sinaimg.cn/large/0069RVTdgy1fv3hbd403cj314w0c0dg0.jpg)

**代码实现：**

```python
import numpy as np

INF = 65536

def find_the_last_domino(n, m, edge, edges):
    s = np.zeros(n + 1, dtype=int)
    s[1] = 1
    path = [-1 for i in range(n + 1)]
    path = np.array(path)
    time = [INF for i in range(n + 1)]
    for i in range(1, n + 1):
        time[i] = edge[1, i]
        if i != 1 and time[i] != INF:
            path[i] = 1
        else:
            path[i] = -1

    for i in range(n - 1):
        min_edge = INF
        u = 1
        for j in range(1, n + 1):
            if not s[j] and time[j] < min_edge:
                min_edge = time[j]
                u = j
        s[u] = 1
        for j in range(1, n + 1):
            if not s[j] and edge[u, j] != INF and time[u] + edge[u, j] < time[j]:
                time[j] = time[u] + edge[u, j]
                path[j] = u

    max_vertex_time = 0
    max_vertex = 0
    for i in range(1, n + 1):
        if time[i] > max_vertex_time:
            max_vertex_time = time[i]
            max_vertex = i

    max_edge_time = 0
    edge_pair = [0, 0]
    for e in edges:
        if (time[e[0]] + time[e[1]] + e[2]) / 2 > max_edge_time:
            max_edge_time = (time[e[0]] + time[e[1]] + e[2]) / 2
            edge_pair[0], edge_pair[1] = e[0], e[1]

    if max_edge_time <= max_vertex_time:
        res = "The last domino falls after {:.1f} seconds, at key domino {}."
        print(res.format(max_vertex_time, max_vertex))
    else:
        res = "The last domino falls after {:.1f} seconds, between key dominoes {} and {}."
        print(res.format(max_edge_time, edge_pair[0], edge_pair[1]))


if __name__ == "__main__":
    with open("180912") as fp:
        n, m = [int(x) for x in fp.readline().split()]
        # 下标从 1 开始
        edge = [[INF] * (n + 1) for i in range(n + 1)]
        edge = np.array(edge)
        edges = list()
        for i in range(1, n + 1):
            edge[i, i] = 0
        for line in fp.readlines():
            u, v, w = [int(x) for x in line.split()]
            edge[u, v] = w
            edges.append((u, v, w))

        find_the_last_domino(n, m, edge, edges)

```

