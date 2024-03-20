def mine_sweeper(arr):
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    N = len(arr)

    for i in range(N):
        for j in range(N):
            if arr[i][j] == 1:
                print("*", end=" ")
            else:
                count = 0
                for dx, dy in directions:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < N and 0 <= ny < N and arr[nx][ny] == 1:
                        count += 1
                print(count, end=" ")
        print()
arr = []
N = int(input("N:"))
for _ in range(N):
    mine_row = input(":")
    arr.append(list(map(int, mine_row.split())))
mine_sweeper(arr)
