# DGAI

**DGAI** is a disk-based approximate nearest neighbor search (ANNS) system that supports **efficient dynamic updates and fast queries** at large scale.

Existing on-disk graph indexes store each vector together with its neighbor list in the same disk page. Updates that only modify graph topology still have to read and rewrite the high-dimensional vectors, which causes heavy redundant I/O. DGAI removes this redundancy by decoupling topology from vectors, and turns the resulting topology read amplification into useful prefetching through a dynamic similarity-aware layout and a search algorithm that independently schedules graph expansion and topology reads.

## ✨ Highlights

- **Decoupled storage**: topology and vectors live in separate files, so topology updates never touch raw vectors.
- **Dynamic similarity-aware layout**: similar nodes stay on the same topology page as the index evolves, without offline reordering.
- **Independent graph expansion and topology reads**: CPU and SSD work in parallel during search, without waiting for each other.
- **Lightweight system optimizations**: cache-aware PQ computation, a topology-only memory index.

## Design

### 1. Decoupled Storage

Graph topology (`*_disk.graph`) and raw vectors (`*_disk.vector`) are stored in separate files and accessed through independent readers.

| Operation | Coupled layout | DGAI |
| --- | --- | --- |
| Topology update (insert / delete repair) | Reads and rewrites vector + topology | Touches the topology file only |
| Graph traversal during search | One page per node | One page serves many nodes |
| Result reranking | Reads vectors with topology | Reads vectors only for top candidates |

### 2. Dynamic Similarity-Aware Layout

DGAI reuses the information during insert and delete to keep similar nodes co-located on the same topology page through three lightweight mechanisms:

- **Insertion-guided page decision**: a new node is placed on the page that holds most of its nearest neighbors, reusing the results of the insertion search at no extra search cost.
- **Locality-preserving page splitting**: when the candidate pages are full, the page is split by a balanced minimum cut (METIS) over PQ-weighted edges, so that similar neighbors remain together.
- **Neighbor-guided page compaction**: during deletion repair, sparse pages are merged into pages that hold their graph neighbors, bounding the space overhead of the topology file.

### 3. Independent Graph Expansion and Topology Reads

Search follows a two-stage framework:

1. **Candidate search** on the topology file. The CPU keeps expanding the nearest candidate whose page is already loaded, while asynchronous reads fetch the pages of the most promising unloaded candidates in best-first order. Neither side waits for the other.
2. **Reranking** on the vector file. Raw vectors are fetched only for the most promising candidates to compute exact distances.

### 4. System Optimizations

- **Cache-aware PQ computation**: PQ distances are evaluated in batches in subspace-first order, improving cache locality with identical results.
- **Topology-only memory index**: an in-memory navigation graph with PQ codes selects entry points without keeping raw vectors in memory.
- **Limited reverse edges (optional)**: reverse edges are added only from the top-`L_construct` candidates of the insertion search, avoiding frequent ineffective updates and lock contention on nodes near the entry point.

## Getting Started

### Build

```bash
sudo apt-get install libmetis-dev
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```