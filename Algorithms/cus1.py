import heapq
from itertools import count


def uniform_cost_search(graph, origin, destinations):
    tie_breaker = count()

    frontier = [(0, next(tie_breaker), origin, [origin])]
    best_cost = {origin: 0} # Records the lowest known cost to reach each node.
    nodes_created = 1

    while frontier:
        cost_so_far, _, current, path = heapq.heappop(frontier)

        if cost_so_far != best_cost.get(current):
            continue

        if current in destinations:
            return current, nodes_created, path

        for neighbour, edge_cost in sorted(
            graph[current], key=lambda item: item[0]
        ):
            new_cost = cost_so_far + edge_cost

            if new_cost < best_cost.get(neighbour, float("inf")): # Infinity, used when neighbour has not been reached before.
                best_cost[neighbour] = new_cost

                heapq.heappush(
                    frontier,
                    (
                        new_cost,
                        next(tie_breaker),
                        neighbour,
                        path + [neighbour]
                    )
                )

                nodes_created += 1

    return None, nodes_created, []