"""Average degree and connected components of a PilotANN index (full graph of IndexStaged), Tables 5 and 6.

Runs in the pilot environment with the patched PilotANN installed (third_party/build/pilotann.sh).

Usage: python3 pilotann_graph_stats.py --graph_path <index.ckpt>
"""
import argparse
from collections import defaultdict, deque
from typing import Dict, List

from pilot_ann.layers.staged import IndexStaged


def load_pilotann_graph(checkpoint_path: str) -> Dict[int, List[int]]:
    index = IndexStaged.load(checkpoint_path)
    index.to(device='cpu')

    # CSR adjacency of the full graph
    indptr = index.fullgraph.indptr.cpu().numpy()
    indices = index.fullgraph.indices.cpu().numpy()

    graph = defaultdict(list)
    for i in range(len(indptr) - 1):
        start, end = indptr[i], indptr[i + 1]
        neighbors = indices[start:end]
        graph[i] = [n for n in neighbors if n != -1]  # -1 marks an empty slot

    return dict(graph)


def calculate_average_degree(graph: Dict[int, List[int]]) -> float:
    total_degree = sum(len(neighbors) for neighbors in graph.values())
    num_nodes = len(graph)
    return total_degree / num_nodes if num_nodes > 0 else 0


def find_connected_components(graph: Dict[int, List[int]]) -> int:
    """Number of components found by breadth-first search over the out-edges."""
    visited = set()
    components = 0

    def bfs(start_node: int) -> None:
        queue = deque([start_node])
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    for node in graph:
        if node not in visited:
            components += 1
            visited.add(node)
            bfs(node)

    return components


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze PilotANN graph')
    parser.add_argument('--graph_path', type=str, required=True, help='Path to the graph checkpoint file')
    args = parser.parse_args()

    graph = load_pilotann_graph(args.graph_path)

    avg_degree = calculate_average_degree(graph)
    print(f'Average Degree: {avg_degree:.2f}')

    num_components = find_connected_components(graph)
    print(f'Number of Connected Components: {num_components}')


if __name__ == '__main__':
    main()
