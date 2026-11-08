def dfs(graph, origin, destinations):
    # Each stack entry contains the current node and its path.
    stack = [(origin, [origin])] # The starting entry counts as one created search entry.
    nodes_created = 1

    while stack:
        current, path = stack.pop() # Remove the most recently added entry from the stack.

        if current in destinations:
            return current, nodes_created, path

        # Add neighbouring nodes to the stack. Smaller node ID is explored first.
        for neighbour, cost in reversed(
            sorted(graph.get(current, []), key=lambda edge: edge[0])
        ):
            # Avoid revisiting a node on the current path.
            if neighbour not in path:
                stack.append((neighbour, path + [neighbour]))
                nodes_created += 1

    return None, nodes_created, []
