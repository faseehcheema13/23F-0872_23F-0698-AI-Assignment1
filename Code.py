import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import heapq

ROWS, COLS = 15, 15
START, TARGET = (0, 0), (14, 14)
DELAY = 0.01 

MOVES = [(-1, 0), (0, 1), (1, 0), (1, 1), (0, -1), (-1, -1)]

def create_grid():
    grid = np.zeros((ROWS, COLS))

    grid[5, 2:13] = 1
    grid[10, 0:10] = 1
    return grid

def draw(grid, frontier=None, explored=None, path=None, title="AI Pathfinder"):
    plt.clf()
    img = np.zeros((ROWS, COLS, 3)) + 0.9 
    img[grid == 1] = [0, 0, 0] 
    
    if explored:
        for node in explored:
            img[node[0], node[1]] = [0.6, 0.8, 1.0]
    
    if frontier:
        for node in frontier:
            
            curr_node = node[1] if isinstance(node, tuple) and len(node) == 2 else node
            if isinstance(curr_node, tuple):
                img[curr_node[0], curr_node[1]] = [1.0, 0.7, 0.0] 
            
    if path:
        for node in path:
            img[node[0], node[1]] = [0.0, 1.0, 0.0] 

    img[START[0], START[1]] = [1.0, 0.0, 0.0] 
    img[TARGET[0], TARGET[1]] = [1.0, 1.0, 0.0]
    
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.pause(DELAY) 

def reconstruct(parent, end_node):
    path, curr = [], end_node
    while curr is not None:
        path.append(curr)
        curr = parent.get(curr)
    return path[::-1]

# --- 1. BFS ---
def bfs(grid):
    q, parent, explored = deque([START]), {START: None}, []
    while q:
        curr = q.popleft()
        if curr == TARGET: return reconstruct(parent, TARGET), explored
        if curr not in explored:
            explored.append(curr)
            for dx, dy in MOVES:
                nxt = (curr[0]+dx, curr[1]+dy)
                if 0<=nxt[0]<ROWS and 0<=nxt[1]<COLS and grid[nxt]==0 and nxt not in parent:
                    parent[nxt] = curr
                    q.append(nxt)
            draw(grid, frontier=list(q), explored=explored, title="BFS Running...")
    return None, explored

# --- 2. DFS ---
def dfs(grid):
    stack, parent, explored = [START], {START: None}, []
    while stack:
        curr = stack.pop()
        if curr == TARGET: return reconstruct(parent, TARGET), explored
        if curr not in explored:
            explored.append(curr)
            for dx, dy in reversed(MOVES):
                nxt = (curr[0]+dx, curr[1]+dy)
                if 0<=nxt[0]<ROWS and 0<=nxt[1]<COLS and grid[nxt]==0 and nxt not in parent:
                    parent[nxt] = curr
                    stack.append(nxt)
            draw(grid, frontier=stack, explored=explored, title="DFS Running...")
    return None, explored

# --- 3. UCS ---
def ucs(grid):
    pq, parent, explored = [(0, START)], {START: None}, []
    cost_so_far = {START: 0}
    while pq:
        curr_cost, curr = heapq.heappop(pq)
        if curr == TARGET: return reconstruct(parent, TARGET), explored
        if curr not in explored:
            explored.append(curr)
            for dx, dy in MOVES:
                nxt = (curr[0]+dx, curr[1]+dy)
                new_cost = curr_cost + 1
                if 0<=nxt[0]<ROWS and 0<=nxt[1]<COLS and grid[nxt]==0:
                    if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                        cost_so_far[nxt] = new_cost
                        parent[nxt] = curr
                        heapq.heappush(pq, (new_cost, nxt))
            draw(grid, frontier=pq, explored=explored, title="UCS Running...")
    return None, explored

# --- 4. DLS (Depth-Limited Search) ---
def dls(grid, limit):
    def recursive_dls(node, current_limit, parent, explored):
        explored.append(node)
        draw(grid, frontier=[], explored=explored, title=f"DLS (Limit {limit}) Running...")
        if node == TARGET: return reconstruct(parent, TARGET)
        if current_limit <= 0: return None
        
        for dx, dy in MOVES:
            nxt = (node[0]+dx, node[1]+dy)
            if 0<=nxt[0]<ROWS and 0<=nxt[1]<COLS and grid[nxt]==0 and nxt not in parent:
                parent[nxt] = node
                result = recursive_dls(nxt, current_limit - 1, parent, explored)
                if result: return result
        return None

    explored = []
    parent = {START: None}
    path = recursive_dls(START, limit, parent, explored)
    return path, explored

# --- 5. IDDFS ---
def iddfs(grid):
    max_depth = ROWS * COLS
    all_explored = []
    for depth in range(max_depth):
        path, explored = dls(grid, depth)
        all_explored.extend(explored)
        if path: return path, all_explored
    return None, all_explored

# --- 6. Bidirectional Search ---
def bidirectional(grid):
    f_q, b_q = deque([START]), deque([TARGET])
    f_parent, b_parent = {START: None}, {TARGET: None}
    f_explored, b_explored = [], []

    while f_q and b_q:
        # Forward Step
        curr_f = f_q.popleft()
        f_explored.append(curr_f)
        for dx, dy in MOVES:
            nxt = (curr_f[0]+dx, curr_f[1]+dy)
            if 0<=nxt[0]<ROWS and 0<=nxt[1]<COLS and grid[nxt]==0 and nxt not in f_parent:
                f_parent[nxt] = curr_f
                f_q.append(nxt)
                if nxt in b_parent: # Meeting Point!
                    path = reconstruct(f_parent, nxt) + reconstruct(b_parent, b_parent[nxt])[::-1]
                    return path, f_explored + b_explored

        # Backward Step
        curr_b = b_q.popleft()
        b_explored.append(curr_b)
        for dx, dy in MOVES:
            nxt = (curr_b[0]+dx, curr_b[1]+dy)
            if 0<=nxt[0]<ROWS and 0<=nxt[1]<COLS and grid[nxt]==0 and nxt not in b_parent:
                b_parent[nxt] = curr_b
                b_q.append(nxt)
                if nxt in f_parent: # Meeting Point!
                    path = reconstruct(f_parent, f_parent[nxt]) + reconstruct(b_parent, nxt)[::-1]
                    return path, f_explored + b_explored
        
        draw(grid, frontier=list(f_q)+list(b_q), explored=f_explored+b_explored, title="Bidirectional Running...")
    return None, f_explored + b_explored

# --- MAIN EXECUTION ---
if _name_ == "_main_":
    grid = create_grid()
    print("Select Algorithm:\n1: BFS\n2: DFS\n3: UCS\n4: DLS\n5: IDDFS\n6: Bidirectional")
    choice = input("Choice: ")
    
    plt.figure(figsize=(7,7))
    path, explored = None, []
    
    if choice == '1': path, explored = bfs(grid)
    elif choice == '2': path, explored = dfs(grid)
    elif choice == '3': path, explored = ucs(grid)
    elif choice == '4': path, explored = dls(grid, 20) # Limit set to 20
    elif choice == '5': path, explored = iddfs(grid)
    elif choice == '6': path, explored = bidirectional(grid)

    if path:
        print("\n--- PATH FOUND ---")
        print("Total Steps:", len(path))
        print("Path Coordinates:", path)
        draw(grid, path=path, title="SUCCESS!")
        plt.show()
    else:
        print("No path found.")
