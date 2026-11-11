import heapq

from heuristic import heuristic, get_cost_scale


def astar(nodes, graph, origin, destinations):
    cost_scale = get_cost_scale(nodes, graph)

    # Priority queue: (f, sequence, node, g, path)
    sequence = 0
    start_h = heuristic(nodes, origin, destinations, cost_scale)

    frontier = [
        (start_h, sequence, origin, 0, [origin])
    ]

    nodes_created = 1

    while frontier:
        f, _, current, g, path = heapq.heappop(frontier)

        if current in destinations:
            return current, nodes_created, path

        # Generate neighbours in ascending node-ID order.
        for neighbour, edge_cost in sorted(
            graph.get(current, []),
            key=lambda edge: edge[0]
        ):
            # Prevent cycles within the current path.
            if neighbour not in path:
                new_g = g + edge_cost

                new_h = heuristic(
                    nodes,
                    neighbour,
                    destinations,
                    cost_scale
                )

                new_f = new_g + new_h
                new_path = path + [neighbour]

                sequence += 1

                heapq.heappush(
                    frontier,
                    (
                        new_f,
                        sequence,
                        neighbour,
                        new_g,
                        new_path
                    )
                )

                nodes_created += 1

    return None, nodes_created, []