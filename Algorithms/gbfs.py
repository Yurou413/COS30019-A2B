import heapq

from heuristic import heuristic, get_cost_scale


def gbfs(nodes, graph, origin, destinations):
    cost_scale = get_cost_scale(nodes, graph)

    # Priority queue: (h, sequence, node, path)
    sequence = 0
    start_h = heuristic(nodes, origin, destinations, cost_scale)

    frontier = [
        (start_h, sequence, origin, [origin])
    ]

    nodes_created = 1

    while frontier:
        h, _, current, path = heapq.heappop(frontier)

        if current in destinations:
            return current, nodes_created, path

        # Generate neighbours in ascending node-ID order.
        for neighbour, edge_cost in sorted(
            graph.get(current, []),
            key=lambda edge: edge[0]
        ):
            # Prevent cycles within the current path.
            if neighbour not in path:
                new_h = heuristic(
                    nodes,
                    neighbour,
                    destinations,
                    cost_scale
                )

                new_path = path + [neighbour]
                sequence += 1

                heapq.heappush(
                    frontier,
                    (
                        new_h,
                        sequence,
                        neighbour,
                        new_path
                    )
                )

                nodes_created += 1

    return None, nodes_created, []