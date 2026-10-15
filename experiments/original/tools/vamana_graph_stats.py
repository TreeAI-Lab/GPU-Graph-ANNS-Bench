"""Average degree and connected components of a DiskANN disk index (<prefix>_disk.index), Tables 5 and 6.

The disk index is the Vamana graph searched by Vamana and BANG. After a 4096-byte header sector every node
holds its vector, its degree and DEGREE neighbor slots. Nodes that fit in a sector are packed several per
sector; larger nodes (for example GIST with R = 128) take whole sectors each. The layout is chosen from the
node length unless --layout is given.

Usage: python3 vamana_graph_stats.py <prefix>_disk.index <dim> <datatype: 0=int8, 1=uint8, 2=float> <degree>
"""
import argparse
import os
import struct
from collections import deque
from typing import List, Tuple

import numpy as np

SECTORLEN = 4096


def read_packed(file_to_read: str, dim: int, datatype: int, degree: int) -> Tuple[List[List[int]], int]:
    """Nodes packed SECTORLEN // NODELEN per sector."""
    datatypesize = 4 if datatype == 2 else 1
    nodelen = dim * datatypesize + 4 + degree * 4
    print(nodelen)
    total_degree = np.int64(0)

    with open(file_to_read, 'rb') as f:
        struct.unpack('<Q', f.read(8))  # file size
        total_nodes = struct.unpack('<Q', f.read(8))[0]
        struct.unpack('<Q', f.read(8))  # medoid
        struct.unpack('<Q', f.read(8))  # max node length
        struct.unpack('<Q', f.read(8))  # nodes per sector

        adj_list = [[] for _ in range(total_nodes)]

        nodes_read = 0
        filesize = os.path.getsize(file_to_read)
        num_sectors = int(filesize / SECTORLEN) - 1
        print('Total nodes:', total_nodes)
        print('num_sectors:', num_sectors)
        for i in range(num_sectors):
            sector_start = (i + 1) * SECTORLEN
            f.seek(sector_start, 0)
            nodes_in_this_sector = min(SECTORLEN // nodelen, total_nodes - nodes_read)
            for _ in range(nodes_in_this_sector):
                if nodes_read == total_nodes:
                    break
                node_bytes = f.read(nodelen)
                if len(node_bytes) < nodelen:
                    break

                offset = dim * datatypesize  # skip the vector
                d = struct.unpack('<I', node_bytes[offset:offset + 4])[0]
                offset += 4
                if i == 0:
                    print(f'Node {nodes_read} has degree {d}')
                neighbors = []
                for _ in range(d):
                    neighbor = struct.unpack('<I', node_bytes[offset:offset + 4])[0]
                    neighbors.append(neighbor)
                    offset += 4

                adj_list[nodes_read] = neighbors
                total_degree += np.int64(d)
                nodes_read += 1
            if nodes_read == total_nodes:
                break
    return adj_list, int(total_degree)


def read_multisector(file_to_read: str, dim: int, datatype: int, degree: int) -> Tuple[List[List[int]], int]:
    """Every node starts a new sector and takes ceil(NODELEN / SECTORLEN) sectors."""
    datatypesize = 4 if datatype == 2 else 1
    nodelen = dim * datatypesize + 4 + degree * 4

    with open(file_to_read, 'rb') as f:
        struct.unpack('<Q', f.read(8))
        filesize = os.path.getsize(file_to_read)
        total_nodes = struct.unpack('<Q', f.read(8))[0]
        medoid = struct.unpack('<Q', f.read(8))[0]
        struct.unpack('<Q', f.read(8))
        max_node_len = dim * datatypesize + 4 + degree * 4
        nodes_per_sec = struct.unpack('<Q', f.read(8))[0]

        print('filesize:', filesize)
        print('total_nodes:', total_nodes)
        print('Medoid is:', medoid)
        print('maxNodeLen:', max_node_len)
        print('nodesPerSec:', nodes_per_sec)
        print('NODELEN:', nodelen)
        print('Datatype =', datatype, 'Datatype size =', datatypesize)

        if nodelen > SECTORLEN:
            nsectors_per_node = int(np.ceil(nodelen / SECTORLEN))
        else:
            nsectors_per_node = 1

        adj_list = [[] for _ in range(total_nodes)]
        total_degree = 0

        nodes_read = 0
        sector_offset = SECTORLEN  # skip the metadata sector

        while nodes_read < total_nodes:
            f.seek(sector_offset, 0)
            node_bytes = b''
            for _ in range(nsectors_per_node):
                node_bytes += f.read(SECTORLEN)
            node_bytes = node_bytes[:nodelen]
            sector_offset += nsectors_per_node * SECTORLEN

            offset_in_node = dim * datatypesize
            if offset_in_node + 4 > len(node_bytes):
                print(f'Error: offset out of range at NodesRead={nodes_read}, offset={offset_in_node}, '
                      f'len={len(node_bytes)}')
                break

            d = struct.unpack('<I', node_bytes[offset_in_node:offset_in_node + 4])[0]
            offset_in_node += 4

            neighbors = []
            for _ in range(d):
                if offset_in_node + 4 > len(node_bytes):
                    print(f'Error: neighbor offset out of range at NodesRead={nodes_read}, '
                          f'offset={offset_in_node}, len={len(node_bytes)}')
                    break
                neighbor = struct.unpack('<I', node_bytes[offset_in_node:offset_in_node + 4])[0]
                neighbors.append(neighbor)
                offset_in_node += 4

            adj_list[nodes_read] = neighbors
            total_degree += d
            nodes_read += 1
            if nodes_read % 10000 == 0:
                print('NodesRead:', nodes_read)
    return adj_list, total_degree


def count_components(adj_list: List[List[int]]) -> int:
    """Breadth-first search over the out-edges; ids outside the graph are ignored."""
    total_nodes = len(adj_list)
    visited = [False] * total_nodes
    num_components = 0
    for node in range(total_nodes):
        if visited[node]:
            continue
        num_components += 1
        queue = deque([node])
        visited[node] = True
        while queue:
            u = queue.popleft()
            for neighbor in adj_list[u]:
                if neighbor < total_nodes and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
    return num_components


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('index', help='<prefix>_disk.index of DiskANN build_disk_index')
    parser.add_argument('dim', type=int)
    parser.add_argument('datatype', type=int, choices=[0, 1, 2], help='0=int8, 1=uint8, 2=float')
    parser.add_argument('degree', type=int, help='R of the build')
    parser.add_argument('--layout', choices=['auto', 'packed', 'multisector'], default='auto')
    args = parser.parse_args()

    nodelen = args.dim * (4 if args.datatype == 2 else 1) + 4 + args.degree * 4
    layout = args.layout
    if layout == 'auto':
        layout = 'multisector' if nodelen > SECTORLEN else 'packed'
    reader = read_multisector if layout == 'multisector' else read_packed
    adj_list, total_degree = reader(args.index, args.dim, args.datatype, args.degree)

    print('Average degree:', total_degree / len(adj_list))
    print('Number of connected components:', count_components(adj_list))


if __name__ == '__main__':
    main()
