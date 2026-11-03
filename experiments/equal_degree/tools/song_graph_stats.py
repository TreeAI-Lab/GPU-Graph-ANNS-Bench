"""Average degree and connected components of a SONG graph file (bfsg.graph), Tables 5 and 6.

bfsg.graph stores 2^shift size_t values per vertex: the out-degree, then the neighbor ids. shift is
FIXED_DEGREE_SHIFT of config.h of the build (build_song.sh prints it).

Usage: python3 song_graph_stats.py <bfsg.graph> --shift 6 [--no-components]
"""
import argparse
from collections import deque
from typing import List

import numpy as np


def analyze_graph(edges: np.ndarray, vertex_offset_shift: int) -> None:
    num_vertices = len(edges) >> vertex_offset_shift

    # the first value of every block is the degree of the vertex
    degrees = edges[::1 << vertex_offset_shift]

    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees)
    min_degree = np.min(degrees)

    print(f'Number of vertices: {num_vertices}')
    print(f'Average degree: {avg_degree:.2f}')
    print(f'Maximum degree: {max_degree}')
    print(f'Minimum degree: {min_degree}')

    unique_degrees, counts = np.unique(degrees, return_counts=True)
    print('\nDegree distribution:')
    for deg, count in zip(unique_degrees, counts):
        print(f'Degree {deg}: {count} vertices ({count / num_vertices * 100:.2f}%)')


def find_connected_components(edges: np.ndarray, num_vertices: int, vertex_offset_shift: int) -> List[List[int]]:
    adj_list = [[] for _ in range(num_vertices)]
    for i in range(num_vertices):
        offset = i << vertex_offset_shift
        degree = edges[offset]
        for j in range(degree):
            neighbor = edges[offset + j + 1]
            if neighbor < num_vertices:
                adj_list[i].append(neighbor)

    # breadth-first search over the out-edges
    visited = np.zeros(num_vertices, dtype=bool)
    components = []

    for start in range(num_vertices):
        if visited[start]:
            continue

        component = []
        queue = deque([start])
        visited[start] = True

        while queue:
            vertex = queue.popleft()
            component.append(vertex)

            for neighbor in adj_list[vertex]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        components.append(component)

    return components


def analyze_components(edges: np.ndarray, vertex_offset_shift: int) -> None:
    num_vertices = len(edges) >> vertex_offset_shift

    print(f'Total number of vertices: {num_vertices}')

    components = find_connected_components(edges, num_vertices, vertex_offset_shift)

    print(f'\nTotal number of connected components: {len(components)}')

    component_sizes = [len(comp) for comp in components]
    sorted_sizes = sorted(component_sizes, reverse=True)

    print('\nTop 10 largest components:')
    for i, size in enumerate(sorted_sizes[:10], 1):
        print(f'Component {i}: {size} vertices ({size / num_vertices * 100:.2f}% of total)')

    print('\nComponent size distribution:')
    size_counts = {}
    for size in component_sizes:
        size_counts[size] = size_counts.get(size, 0) + 1

    for size in sorted(size_counts.keys()):
        count = size_counts[size]
        print(f'Size {size}: {count} components')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('graph', help='bfsg.graph written by song build')
    parser.add_argument('--shift', type=int, required=True, help='FIXED_DEGREE_SHIFT of the build')
    parser.add_argument('--no-components', action='store_true', help='print the degrees only')
    args = parser.parse_args()

    with open(args.graph, 'rb') as f:
        data = f.read()
    # every value is a size_t (8 bytes)
    edges = np.frombuffer(data, dtype=np.uint64)

    analyze_graph(edges, args.shift)
    if not args.no_components:
        analyze_components(edges, args.shift)


if __name__ == '__main__':
    main()
