import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from grid import create_grid, START, TARGET
from algorithms import bfs, dfs, ucs, dls, iddfs, bidirectional
from visualization import draw

def main():
    grid = create_grid()

    print("Select Algorithm:")
    print("1: BFS")
    print("2: DFS")
    print("3: UCS")
    print("4: DLS")
    print("5: IDDFS")
    print("6: Bidirectional")

    choice = input("Choice: ")

    plt.figure(figsize=(7, 7))
    plt.gcf().canvas.manager.set_window_title("GOOD PERFORMANCE TIME APP")

    path, explored = None, []

    if choice == '1':
        path, explored = bfs(grid)
    elif choice == '2':
        path, explored = dfs(grid)
    elif choice == '3':
        path, explored = ucs(grid)
    elif choice == '4':
        path, explored = dls(grid, 20)
    elif choice == '5':
        path, explored = iddfs(grid)
    elif choice == '6':
        path, explored = bidirectional(grid)
    else:
        print("Invalid choice")
        return

    if path:
        print("\n--- PATH FOUND ---")
        print("Total Steps:", len(path))
        print("Path:", path)
        draw(grid, path=path, title="SUCCESS!")
        plt.show()
    else:
        print("No path found.")

if __name__ == "__main__":
    main()
