import logging
import os
import json
import time
import numpy as np
import shutil

from typing import Optional

from open_webui.retrieval.vector.main import VectorItem, SearchResult, GetResult
from open_webui.env import SRC_LOG_LEVELS
from pipeann import IndexPipeANN, Metric

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])

# TODO: test this in open_webui.

class Collection:
    def __init__(self, persist_dir, collection_name):
        self.last_save_ts = 0
        self.collection_name = collection_name
        self.tag_id_map = []

        # TODO: use faster way to store tag_id_map.
        os.makedirs(f"{persist_dir}/{collection_name}", exist_ok=True)
        self.index_prefix = f"{persist_dir}/{collection_name}/index"
        self.data_file = f"{persist_dir}/{collection_name}/data.json"
        self.index_metadata_file = f"{persist_dir}/{collection_name}/index_metadata.json"
        self.tag_id_map_file = f"{persist_dir}/{collection_name}/tag_id_map.json"
        
        print("CWD is", os.getcwd())
        if os.path.exists(self.data_file):
            self.data = json.load(open(self.data_file, "r", encoding="utf-8"))
        else:
            self.data = {}
        
        if os.path.exists(self.tag_id_map_file):
            self.tag_id_map = json.load(open(self.tag_id_map_file, "r", encoding="utf-8"))
        else:
            self.tag_id_map = []
        
        if os.path.exists(self.index_metadata_file):
            index_metadata = json.load(open(self.index_metadata_file, "r", encoding="utf-8"))
            ndims = index_metadata["ndims"]
            dtype = np.dtype(index_metadata["dtype"])
            self.index = IndexPipeANN(ndims, dtype, Metric.L2)
            self.index.load(self.index_prefix)
        else:
            self.index = None
    
    def insert(self, items: list[VectorItem]):
        # Insert the items into the collection, if the collection does not exist, it will be created.
        ids = [item["id"] for item in items]

        embeddings = np.array([item["vector"] for item in items])
        if self.index is None:
            self.index = IndexPipeANN(embeddings.shape[1], embeddings.dtype, Metric.L2)
            self.index.set_index_prefix(self.index_prefix)
            # create index metadata.
            index_metadata = {
                "ndims": embeddings.shape[1],
                "dtype": str(embeddings.dtype)
            }
            json.dump(index_metadata, open(self.index_metadata_file, "w", encoding="utf-8"))


        tags = [i for i in range(len(self.tag_id_map), len(self.tag_id_map) + len(items))]
        self.index.add(embeddings, np.array(tags)) # tags = nums

        self.tag_id_map.extend(ids)
        for tag, x in zip(tags, items):
            self.data[x["id"]] = {
                "tag": tag,
                "text": x["text"], 
                "metadata": x["metadata"]
            }
    
    def upsert(self, items: list[VectorItem]):
        insert_items = []
        for x in items:
            if x["id"] in self.data:
                self.data[x["id"]] = {
                    "tag": self.data[x["id"]]["num"],
                    "text": x["text"], 
                    "metadata": x["metadata"]
                }
            else:
                insert_items.append(x)
        self.insert(insert_items)

    def delete(
        self,
        ids: Optional[list[str]] = None,
        filter: Optional[dict] = None,
    ):
        if ids is not None:
            self.index.remove(np.array([self.data[id]["tag"] for id in ids]))
            for id in ids:
                del self.data[id]
        elif filter is not None:
            print("Filter deletion not supported yet")
            # Filter:  {'file_id': 'fb06b634-0f85-478d-adeb-234dfa0c1836'}
            print("Filter: ", filter)
        self.save()
    
    def search(
        self, vectors: list[list[float | int]], limit: int
    ) -> Optional[SearchResult]:
        D, nums = self.index.search(np.array(vectors), limit)
        ids = [[self.tag_id_map[x] for x in num] for num in nums]
        docs = []
        metas = []
        for i in range(len(ids)):
            cur_docs = []
            cur_metas = []
            for id in ids[i]:
                if self.data[id]["metadata"].get("ext", None) is not None:
                    cur_docs.append(f"""{self.data[id]["metadata"]["ext"]}
                                    related segment: {self.data[id]["text"]}""")
                else:
                    cur_docs.append(self.data[id]["text"])
                cur_metas.append(self.data[id]["metadata"])
            docs.append(cur_docs) # used for generation.
            metas.append(cur_metas)

        # print("Fetched docs: ", docs)
        return SearchResult(
            **{
                "ids": ids,
                "distances": D,
                "documents": docs,
                "metadatas": metas,
            }
        )
    
    def clear(self):
        # recursive remove the dir.
        if os.path.exists(f"{self.persist_dir}/{self.collection_name}"):
            shutil.rmtree(f"{self.persist_dir}/{self.collection_name}")
    
    def save(self):
        cur_ts = int(time.time())
        if cur_ts - self.last_save_ts < 60:
            return
        if not self.collection_name.startswith("file-"):
            # skip single-file collection saving, collection with multiple files always startwith UUID.
            json.dump(self.data, open(self.data_file, "w", encoding="utf-8"))
            json.dump(self.tag_id_map, open(self.tag_id_map_file, "w", encoding="utf-8"))
            self.index.save(self.index_prefix)


collections : dict[str, Collection] = {}

# /persist_dir/{collection_name}/_disk.index
# /persist_dir/{collection_name}/_mem.index
class PipeANNClient:
    def __init__(self, persist_dir: str):
        os.makedirs(persist_dir, exist_ok=True)
        self.persist_dir = persist_dir
        collection_names = os.listdir(self.persist_dir)
        for collection_name in collection_names:
            collections[collection_name] = Collection(persist_dir, collection_name)

    def has_collection(self, collection_name: str) -> bool:
        # Check if the collection exists based on the collection name.
        return collection_name in collections.keys()

    def delete_collection(self, collection_name: str):
        # Delete the collection based on the collection name.
        # dummy currently.
        if collection_name in collections:
            collections[collection_name].clear()
        del collections[collection_name]
    
    def insert(self, collection_name: str, items: list[VectorItem]):
        if collection_name not in collections:
            collections[collection_name] = Collection(self.persist_dir, collection_name)
        ret = collections[collection_name].insert(items)
        return ret
        

    def upsert(self, collection_name: str, items: list[VectorItem]):
        if collection_name not in collections:
            collections[collection_name] = Collection(self.persist_dir, collection_name)
        return collections[collection_name].upsert(items)

    def delete(
        self,
        collection_name: str,
        ids: Optional[list[str]] = None,
        filter: Optional[dict] = None,
    ):
        print("Delete called with ", collection_name, ids, filter)
        if collection_name in collections:
            return collections[collection_name].delete(ids, filter)

    def search(
        self, collection_name: str, vectors: list[list[float | int]], limit: int
    ) -> Optional[SearchResult]:
        if collection_name not in collections:
            collections[collection_name] = Collection(self.persist_dir, collection_name)
        return collections[collection_name].search(vectors, limit)

    def query(
        self, collection_name: str, filter: dict, limit: Optional[int] = None
    ) -> Optional[GetResult]:
        # Query the items from the collection based on the filter.
        # filter search.
        print("Filter not supported yet", collection_name, filter)
        return None
        # return GetResult(
        #     **{
        #         "ids": [result["ids"]],
        #         "documents": [result["documents"]],
        #         "metadatas": [result["metadatas"]],
        #     }
        # )

    def get(self, collection_name: str) -> Optional[GetResult]:
        # Get all the items in the collection.
        if collection_name not in collections:
            collections[collection_name] = Collection(self.persist_dir, collection_name)
        data = collections[collection_name].data
        return GetResult(
            **{
                "ids": list(data.keys()),
                "documents": [data[x]["text"] for x in data.keys()],
                "metadatas": [data[x]["metadata"] for x in data.keys()],
            }
        )

    def reset(self):
        # Resets the database. This will delete all collections and item entries.
        for collection in collections:
            collections[collection].clear()
        collections.clear()
