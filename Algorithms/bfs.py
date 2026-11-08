from collections import deque

def bfs(graph, origin, destinations):
    # Each queue entry contains the current node and its path.
    queue = deque([(origin, [origin])])
    nodes_created = 1

    while queue:
        current, path = queue.popleft() # popleft() removes the node entered the queue first.

        if current in destinations:
            return current, nodes_created, path

        for neighbour, cost in sorted(
        graph.get(current, []), key=lambda edge: edge[0]
        ):
            # Avoid cycles within the current path.
            if neighbour not in path:
                queue.append((neighbour, path + [neighbour])) # Add new search entries.
                nodes_created += 1

    return None, nodes_created, []