import numpy as np
import os
import sys
import struct
import time
from pipeann import IndexPipeANN, Metric

def bin_write(vectors, filename):
    with open(filename, 'wb') as f:
        num_vecs, vector_dim = vectors.shape
        f.write(struct.pack('<i', num_vecs))
        f.write(struct.pack('<i', vector_dim))
        f.write(vectors.tobytes())

def bin_read(filename, dtype="float32"):
    cur_data_type_size = np.dtype(dtype).itemsize
    with open(filename, 'rb') as f:
        num_vecs = struct.unpack('<i', f.read(4))[0]
        vector_dim = struct.unpack('<i', f.read(4))[0]
        data = f.read(num_vecs * vector_dim * cur_data_type_size)
        vectors = np.frombuffer(data, dtype=dtype).reshape((num_vecs, vector_dim))
        return vectors

data_dim = 0
data_type = "uint8"
full_data_path = "/mnt/nvme/data/bigann/bigann_1M.bbin"
data_2M_path = "/mnt/nvme/data/bigann/bigann_2M.bbin"
query_path = "/mnt/nvme/data/bigann/bigann_query.bbin"
gt_path = "/mnt/nvme/indices_upd/bigann_gnd/idx_1M.ibin"
gt_2M_path = "/mnt/nvme/indices_upd/bigann_gnd/2M_topk/gt_990000.bin"
index_prefix = "/mnt/nvme/indices/bigann/1M"

def main():
    queries = bin_read(query_path, data_type)
    gt = bin_read(gt_path, "int32")
    gt_2M = bin_read(gt_2M_path, "int32")
    full_data_2M = bin_read(data_2M_path, data_type)
    print(full_data_2M.shape)

    data_dim = full_data_2M.shape[1]
    idx = IndexPipeANN(data_dim, data_type, Metric.L2)
    idx.omp_set_num_threads(32) # the number of search/insert threads.
    idx.set_index_prefix(index_prefix)
    """
print(f"Building index with prefix {index_prefix}...")
Way 1 to initialize index:
    idx.build(data_path, index_prefix) # build SSD index.
    idx.load(index_prefix) # manually load the index after building it.
Way 2: use index.add. Here we use the first half of the dataset to build the index.
    full_data = bin_read(full_data_path, data_type)
    for i in range(0, full_data.shape[0], 10000):
        print(f"Inserting data points {i} to {min(i+10000, full_data.shape[0])} ...")
        idx.add(full_data[i:min(i+10000, full_data.shape[0])], np.arange(i, min(i+10000, full_data.shape[0])))
    """

    print(f"Building index with prefix {index_prefix}...")
    for i in range(0, full_data_2M.shape[0] // 2, 10000):
        print(f"Inserting the first 1M points {i} to {min(i+10000, full_data_2M.shape[0] // 2)} ...")
        idx.add(full_data_2M[i:min(i+10000, full_data_2M.shape[0] // 2)], np.arange(i, min(i+10000, full_data_2M.shape[0] // 2)))
    # The index after adding vectors is inconsistent on disk, so we need to save it first.
    # Directly searching in it is fine.
    idx.save(index_prefix)

    print(f"Loading index with prefix {index_prefix}...")
    idx.load(index_prefix)
    topk = 10

    for L in [10, 20, 30, 40, 50]:
        print(f"Searching for {topk} nearest neighbors with L={L}...")
        t1 = time.clock_gettime(time.CLOCK_REALTIME)
        ids, dists = idx.search(queries, topk, L)
        t2 = time.clock_gettime(time.CLOCK_REALTIME)
        print(f"Search time: {t2 - t1:.4f} seconds for {len(queries)} queries, throughput: {len(queries) / (t2 - t1)} QPS.")
        recall = np.mean([
            len(set(ids[i]) & set(gt[i][:topk])) / topk
            for i in range(len(queries))
        ])
        print(f"Recall@{topk} with L={L}: {recall:.4f}")
    
    # insert vectors.
    print(f"Inserting 1M new vectors to the index ...")
    for i in range(1000000, 2000000, 10000):
        print(f"Inserting data points {i} to {min(i+10000, full_data_2M.shape[0])} ...")
        idx.add(full_data_2M[i:min(i+10000, full_data_2M.shape[0])], np.arange(i, min(i+10000, full_data_2M.shape[0])))
    print(f"Deleting the first 1M vectors from the index ...")
    idx.remove(np.arange(0, 1000000))

    # save and load.
    idx.save(index_prefix)
    idx.load(index_prefix)

    for L in [10, 20, 30, 40, 50]:
        print(f"Searching for {topk} nearest neighbors with L={L}...")
        t1 = time.clock_gettime(time.CLOCK_REALTIME)
        ids, dists = idx.search(queries, topk, L)
        t2 = time.clock_gettime(time.CLOCK_REALTIME)
        print(f"Search time: {t2 - t1:.4f} seconds for {len(queries)} queries, throughput: {len(queries) / (t2 - t1)} QPS.")
        recall = np.mean([
            len(set(ids[i]) & set(gt_2M[i][:topk])) / topk
            for i in range(len(queries))
        ])
        print(f"Recall@{topk} with L={L}: {recall:.4f}")

if __name__ == "__main__":
    main()
