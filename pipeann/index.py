from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from .C import PyIndex as _NativeIndex
from .C import Metric as _Metric

class Metric:
    L2 = _Metric.L2
    INNER_PRODUCT = _Metric.INNER_PRODUCT
    COSINE = _Metric.COSINE


class IndexPipeANN:
    """
    A Python wrapper for the PipeANN index.
    Parameters
    ----------
    data_dim : int
        Dimensionality of the data vectors.
    data_type : str
        Data type of the vectors. Supported types: 'float32', 'int8', 'uint8'.
    metric : Metric
        Distance metric to use. Supported metrics: Metric.L2, Metric.COSINE.
    max_nthreads : int, optional
        Maximum number of threads to use for parallel operations. Default is 128.
        This only defines how many buffers are allocated, the number of threads used in search
        should be configured using `omp_set_num_threads`.
    """
    def __init__(self, data_dim: int, data_type: str | np.dtype, metric: Metric, max_nthreads: int = 128):
        # Ensure dtype objects are numpy dtypes
        params = dict()
        params['data_dim'] = int(data_dim)
        if isinstance(data_type, str):
            data_type = np.dtype(data_type)
        params['data_type'] = data_type
        params['metric'] = _Metric(metric)
        params['max_nthreads'] = int(max_nthreads)
        self._impl = _NativeIndex(params)

    def set_index_prefix(self, index_prefix: str) -> None:
        self._impl.set_index_prefix(index_prefix)

    def omp_set_num_threads(self, num_threads: int) -> None:
        self._impl.omp_set_num_threads(num_threads)

    # I/O
    def load(self, index_prefix: str) -> None:
        self._impl.load(index_prefix)

    def save(self, index_prefix: str) -> bool:
        return self._impl.save(index_prefix)

    # Build
    def build(
        self,
        data_path: str,
        index_prefix: str,
        tag_file: Optional[str] = None,
        build_mem_index: bool = False,
        max_nbrs: int = 0,
        build_L: int = 0,
        PQ_bytes: int = 32,
        memory_use_GB: int = 0,
    ) -> None:
        self._impl.build(data_path, index_prefix, tag_file, build_mem_index, max_nbrs, build_L, PQ_bytes, memory_use_GB)

    # Update & Query
    def add(self, vectors: np.ndarray, tags: np.ndarray) -> None:
        assert vectors.ndim == 2, "vectors must be 2D"
        assert tags.ndim == 1 and len(tags) == len(vectors), "tags must be 1D with same length as vectors"
        self._impl.add(np.ascontiguousarray(vectors), np.ascontiguousarray(tags.astype(np.uint32)))

    def remove(self, tags: np.ndarray) -> None:
        assert tags.ndim == 1, "tags must be 1D"
        self._impl.remove(np.ascontiguousarray(tags.astype(np.uint32)))

    def search(self, queries: np.ndarray, topk: int, L: int) -> Tuple[np.ndarray, np.ndarray]:
        assert queries.ndim == 2, "queries must be 2D"
        ids, dists = self._impl.search(np.ascontiguousarray(queries), int(topk), int(L))
        return np.asarray(ids), np.asarray(dists)

    def __repr__(self) -> str:  # pragma: no cover
        return repr(self._impl)
