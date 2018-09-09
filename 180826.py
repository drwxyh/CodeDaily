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
