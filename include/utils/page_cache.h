#ifndef PAGE_CACHE_H_
#define PAGE_CACHE_H_

#include <cstring>
#include <cstdint>
#include <vector>
#include "ssd_index_defs.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "utils/lock_table.h"

namespace pipeann {
  // User-space page cache for update acceleration (in fact it's a buffer)
  // only used for write-write, ensure that disk has a consistent state
  // expect a lock-free read

  struct PageCacheItem {
    uint8_t *buf;
    uint64_t ref_cnt;

    // use lock!
    uint64_t ref() {
      return ++ref_cnt;
    }

    // use lock!
    uint64_t deref() {
      return --ref_cnt;
    }
  };

  struct PageCache {
    bool get(uint64_t block_no, uint8_t *value, bool ref = false) {
      bool ret = cache.update_fn(block_no, [&](PageCacheItem &v) {
        memcpy(value, v.buf, SECTOR_LEN);
        if (ref) {
          v.ref();
        }
      });
      return ret;
    }

    bool put(uint64_t block_no, uint8_t *value, bool ref = false) {
      return cache.upsert(block_no, [&](PageCacheItem &v, libcuckoo::UpsertContext ctx) {
        if (ctx == libcuckoo::UpsertContext::NEWLY_INSERTED) {
          v = PageCacheItem{.buf = new uint8_t[SECTOR_LEN], .ref_cnt = 0};
        }
        if (ref) {
          v.ref();
        }
        memcpy(v.buf, value, SECTOR_LEN);
      });
    }

    // Read completions are snapshots, never authoritative write images. An
    // older disk read may finish after wbc_write publishes a newer page, or
    // even after that write has persisted and its cache reference was dropped.
    // Never install such a snapshot into the shared write cache. Prefer an
    // existing current write image, and only retain references actually held.
    void finish_read(uint64_t block_no, uint8_t *value, std::vector<uint64_t> *refs) {
      if (get(block_no, value, refs != nullptr) && refs != nullptr) {
        refs->push_back(block_no);
      }
    }

    bool deref(uint64_t block_no) {
      bool ret = cache.uprase_fn(block_no, [&](PageCacheItem &v, libcuckoo::UpsertContext ctx) {
        if (ctx == libcuckoo::UpsertContext::NEWLY_INSERTED) {
          LOG(ERROR) << "PageCache: deref a non-exist block_no: " << block_no;
          return true;
          __builtin_trap();
        }
        uint64_t refs = v.deref();
        if (refs == 0) {
          delete[] v.buf;
        }
        return refs == 0;
      });
      return ret;
    }

    void clear() {
      cache.clear();
    }

    // Drop one page (used after merge/compaction writes instead of clear()).
    void invalidate(uint64_t block_no) {
      cache.erase_fn(block_no, [](PageCacheItem &entry) {
        delete[] entry.buf;
        return true;
      });
    }

    void invalidate_range(uint64_t first_page, uint64_t n_pages) {
      for (uint64_t i = 0; i < n_pages; ++i) invalidate(first_page + i);
    }

    SparseLockTable<uint64_t> lock_table;
    libcuckoo::cuckoohash_map<uint64_t, PageCacheItem> cache;
  };

  inline PageCache cache;

  // Simple linear cache for the tail of the vector file.
  // Vector writes are append-only, so we only need to cache the last few
  // sectors.  This is a fixed-size circular buffer indexed by (sector_no % n).
  template<uint64_t N = 8>  // N sectors cached (8 * 4096 = 32 KB per thread)
  struct VecTailCache {
  // In decoupled mode the vec sector size == SECTOR_LEN (one sector per IO),
  // so we cache at SECTOR_LEN granularity.
  struct Entry {
    uint64_t block_no = UINT64_MAX;
    uint8_t buf[SECTOR_LEN] __attribute__((aligned(64)));
  };
  Entry entries_[N];

  // Return cached data.  Returns true on hit.
  bool get(uint64_t block_no, uint8_t *out) {
    for (auto &e : entries_) {
      if (e.block_no == block_no) {
        memcpy(out, e.buf, SECTOR_LEN);
        return true;
      }
    }
    return false;
  }

  // Insert (or update) a sector in the cache.
  void put(uint64_t block_no, const uint8_t *data) {
    uint64_t idx = next_++ % N;
    entries_[idx].block_no = block_no;
    memcpy(entries_[idx].buf, data, SECTOR_LEN);
  }

 private:
  uint64_t next_ = 0;
  };

  // Per-thread linear cache for the tail of the vector file.
  // Used by both insert (write side) and search (read side).
  // Defined in decouple_insert.cpp with thread_local.
  extern thread_local VecTailCache<8> vec_tail_cache;

}  // namespace pipeann

#endif  // PAGE_CACHE_H_