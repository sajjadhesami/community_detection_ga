#!/usr/bin/env python
"""
Compute modularity (Q) for a stored community assignment.

Example:
    python scripts/compute_q.py \
        --graph data/ca-HepTh.mtx \
        --communities data/ca-HepTh.out
"""

import argparse
import ast
import pathlib
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
from networkx.algorithms.community.quality import modularity


def load_mtx(path: pathlib.Path, weighted: bool = False) -> nx.Graph:
    """Minimal Matrix Market reader (coordinate format)."""
    with path.open("r", encoding="utf-8") as handle:
        header_read = False
        graph = nx.Graph()
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if not header_read:
                # rows, cols, nnz -> ignore values, graph inferred from edges.
                header_read = True
                continue
            if len(parts) < 2:
                continue
            u = int(parts[0])
            v = int(parts[1])
            w = float(parts[2]) if (weighted and len(parts) >= 3) else 1.0
            if weighted:
                graph.add_edge(u, v, weight=w)
            else:
                graph.add_edge(u, v)
    return graph


def load_graph(path: pathlib.Path, weighted: bool = False) -> nx.Graph:
    """Load a graph from Matrix Market (.mtx) or edge list."""
    suffix = path.suffix.lower()
    if suffix == ".mtx":
        return load_mtx(path, weighted=weighted)
    if weighted:
        return nx.read_weighted_edgelist(path, nodetype=int)
    return nx.read_edgelist(path, nodetype=int)


def load_communities(path: pathlib.Path) -> List[Sequence[int]]:
    """Read community assignment stored via repr(popSpace[0].V)."""
    data = path.read_text(encoding="utf-8")
    parsed = ast.literal_eval(data)
    if isinstance(parsed, dict):
        return [list(nodes) for nodes in parsed.values()]
    if isinstance(parsed, Iterable):
        return [list(nodes) for nodes in parsed]
    raise ValueError(f"Unsupported community format in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute modularity (Q) for stored communities.")
    parser.add_argument("--graph", required=True, help="Path to the graph file (MatrixMarket or edge list).")
    parser.add_argument("--communities", required=True, help="Path to the *.out file with communities.")
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Treat the graph as weighted (uses weight column / attribute if present).",
    )
    args = parser.parse_args()

    graph_path = pathlib.Path(args.graph)
    comm_path = pathlib.Path(args.communities)

    if not graph_path.exists():
        raise FileNotFoundError(graph_path)
    if not comm_path.exists():
        raise FileNotFoundError(comm_path)

    G = load_graph(graph_path, weighted=args.weighted)
    communities = load_communities(comm_path)

    q_value = modularity(G, communities, weight="weight" if args.weighted else None)
    print(f"Graph: {graph_path.name}")
    print(f"Communities: {comm_path.name}")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of communities: {len(communities)}")
    print(f"Modularity Q = {q_value:.6f}")


if __name__ == "__main__":
    main()

