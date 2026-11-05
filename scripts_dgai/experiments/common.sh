#!/bin/bash
# Shared helpers for DGAI search / update experiment scripts.
# Sourced by scripts under scripts_dgai/experiments/.

set -u

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DGAI="$(cd "$EXP_DIR/.." && pwd)"

# env.sh derives PROJECT_PATH from $0. Callers live in experiments/, so $0 points one
# level too deep and GREATOR_ROOT="$(cd "$PROJECT_PATH/../Greator" ...)" fails. With
# set -e that failed command-substitution aborts the whole script during source.
# Temporarily disable -e while sourcing, then correct paths.
_exp_errexit_was_on=0
case $- in *e*) _exp_errexit_was_on=1; set +e ;; esac
# shellcheck source=/dev/null
source "$SCRIPTS_DGAI/env.sh"
if [ "$_exp_errexit_was_on" -eq 1 ]; then set -e; fi
unset _exp_errexit_was_on

PROJECT_PATH="$(cd "$SCRIPTS_DGAI/.." && pwd)"
DGAI_ROOT="$PROJECT_PATH"
FRESH_ROOT="${FRESH_ROOT:-$(cd "$DGAI_ROOT/../Freshdiskann" 2>/dev/null && pwd || true)}"
GREATOR_ROOT="${GREATOR_ROOT:-$(cd "$DGAI_ROOT/../Greator" 2>/dev/null && pwd || true)}"
GOVECTOR_ROOT="${GOVECTOR_ROOT:-$(cd "$DGAI_ROOT/../GoVector" 2>/dev/null && pwd || true)}"
if [ -z "${GREATOR_ROOT}" ]; then
    GREATOR_ROOT="$(cd "$DGAI_ROOT/../Greator" 2>/dev/null && pwd || true)"
fi
if [ -z "${FRESH_ROOT}" ]; then
    FRESH_ROOT="$(cd "$DGAI_ROOT/../Freshdiskann" 2>/dev/null && pwd || true)"
fi
if [ -z "${GOVECTOR_ROOT}" ]; then
    GOVECTOR_ROOT="$(cd "$DGAI_ROOT/../GoVector" 2>/dev/null && pwd || true)"
fi
OUTPUT_PATH="${DISKANN_INDEX_PATH}"

# Default experiment knobs (override via env)
K="${K:-10}"
MEM_L="${MEM_L:-10}"
L_DISK="${L_DISK:-75}"
LSEARCH="${LSEARCH:-100}"
# Update rounds: experiments always use 1 (env.sh defaults to 3).
# Override with EXP_NUM_BATCHES without changing scripts_dgai/env.sh.
NUM_BATCHES="${EXP_NUM_BATCHES:-1}"
export NUM_BATCHES
SEARCH_MODES="${SEARCH_MODES:-0 1 2 4}"
RUN_SEARCH_SYSTEMS="${RUN_SEARCH_SYSTEMS:-dgai govector}"
GOVECTOR_BEAMWIDTH="${GOVECTOR_BEAMWIDTH:-4}"
GOVECTOR_CACHE_NODES="${GOVECTOR_CACHE_NODES:-10000}"
GOVECTOR_NUM_QUERIES="${GOVECTOR_NUM_QUERIES:-0}"
GOVECTOR_DATA_TYPE="${GOVECTOR_DATA_TYPE:-float}"
GOVECTOR_DIST_FN="${GOVECTOR_DIST_FN:-l2}"
BASE_DATASETS="${BASE_DATASETS:-text2img laion msmarco}"
# Thread sweeps (search + update) default to text2img only.
THREAD_DATASETS="${THREAD_DATASETS:-text2img}"
# Batch-size sweep default to text2img only.
BATCH_DATASETS="${BATCH_DATASETS:-text2img}"
# Main paper baselines (3 base + large deep1b)
MAIN_DATASETS="${MAIN_DATASETS:-text2img laion msmarco deep1b}"
# Dedicated update experiment lists.
BASELINE_DATASETS="${BASELINE_DATASETS:-text2img laion msmarco}"
SCALE_DATASETS="${SCALE_DATASETS:-deep10m deep100m deep1b}"
SEARCH_THREAD_LIST="${SEARCH_THREAD_LIST:-1 2 4 8 16 32}"
UPDATE_THREAD_LIST="${UPDATE_THREAD_LIST:-1 2 4 8 16 32}"
BATCH_RATIOS="${BATCH_RATIOS:-0.001 0.005 0.01 0.05}"
RUN_UPDATE_SYSTEMS="${RUN_UPDATE_SYSTEMS:-odin dgai fresh greator}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
# Skip individual runs when a successful prior log already exists.
SKIP_EXISTING="${SKIP_EXISTING:-1}"

DGAI_OP_CPP="$DGAI_ROOT/tests/overall_performance.cpp"
FRESH_OP_CPP="$FRESH_ROOT/tests/overall_performance.cpp"
FRESH_MERGER_CPP="$FRESH_ROOT/src/v2/index_merger.cpp"
GREATOR_OP_CPP="$GREATOR_ROOT/tests/overall_performance.cpp"
GREATOR_MERGER_CPP="$GREATOR_ROOT/src/v2/index_merger.cpp"

# Default thread values (restored after thread sweeps)
DGAI_DEFAULT_INSERT=10
DGAI_DEFAULT_MERGE=20
FRESH_DEFAULT_INSERT=10
FRESH_DEFAULT_N=20
FRESH_OP_DEFAULT_INSERT=24
GREATOR_DEFAULT_INSERT=10
GREATOR_DEFAULT_N=20
GREATOR_OP_DEFAULT_INSERT=10
GREATOR_OP_DEFAULT_DELETE=20
GREATOR_OP_DEFAULT_MERGE=20

_exp_ts() { date +%m%d_%H%M%S; }

_exp_system_enabled() {
    local sys="$1"
    local s
    for s in $RUN_UPDATE_SYSTEMS; do
        if [ "$s" = "$sys" ]; then
            return 0
        fi
    done
    return 1
}

_exp_search_system_enabled() {
    local sys="$1"
    local s
    for s in $RUN_SEARCH_SYSTEMS; do
        if [ "$s" = "$sys" ]; then
            return 0
        fi
    done
    return 1
}

log_dir_for() {
    # log_dir_for <system> <exp_name>
    # system: dgai|odin|fresh|greator|search
    local system="$1"
    local exp_name="$2"
    local root
    case "$system" in
        dgai|odin|search) root="$DGAI_ROOT/exp_logs" ;;
        fresh)            root="$FRESH_ROOT/exp_logs" ;;
        greator)          root="$GREATOR_ROOT/exp_logs" ;;
        govector)         root="$GOVECTOR_ROOT/exp_logs" ;;
        *)
            echo "ERROR: unknown system '$system'" >&2
            return 1
            ;;
    esac
    local out="${root}/${exp_name}"
    mkdir -p "$out"
    echo "$out"
}

# L sweeps sized to cover recall ~90-99 (esp. 98); denser full / sparser for slow thread runs.
# Usage: search_L_for_dataset <data_name> [num_threads]
# density: auto sparse when num_threads<=2, or SEARCH_L_DENSITY=full|sparse
search_L_for_dataset() {
    local data_name="$1"
    local num_threads="${2:-8}"
    local density="${SEARCH_L_DENSITY:-}"

    if [ -z "$density" ]; then
        if [ "$num_threads" -le 2 ] 2>/dev/null; then
            density="sparse"
        else
            density="full"
        fi
    fi

    if [ "$density" = "sparse" ]; then
        # Fewer points for thread=1/2; still brackets 90 and 98 for interpolation.
        case "$data_name" in
            text2img|deep1m)
                echo "20 30 40 50 70 100 150 200 300 500"
                ;;
            laion|deep10m|deep100m)
                echo "30 50 80 120 200 400 600 800 1000 1200 1500"
                ;;
            msmarco)
                echo "20 40 60 100 150 200 300 500 800"
                ;;
            deep1b|*)
                echo "150 300 500 700 900 1100 1400 1800 2200 2600 3000"
                ;;
        esac
        return 0
    fi

    # Full: dense enough for curves; extended where logs showed 98 is hard.
    case "$data_name" in
        text2img|deep1m)
            # 98 around L=60-100; keep fine grid, top 600 enough for ~99.8
            echo "10 11 12 14 16 18 20 23 26 30 35 40 50 60 70 80 100 120 150 200 250 300 400 500 600"
            ;;
        laion|deep10m|deep100m)
            # 98 near L=1000 historically; extend to 1500 for margin
            echo "10 12 14 16 18 20 23 26 30 35 40 50 60 70 80 100 120 150 200 250 300 400 500 600 800 1000 1200 1500"
            ;;
        msmarco)
            # 98 around L=200-300; 1000 already tops ~99.5
            echo "10 12 14 16 18 20 23 26 30 35 40 50 60 70 80 100 120 150 200 250 300 400 500 600 800 1000"
            ;;
        deep1b|*)
            # mode4 sometimes needs L>~2000; extend to 3000 with coarser high end
            echo "100 200 300 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000 2300 2600 3000"
            ;;
    esac
}

# ---------- Build helpers ----------

build_dgai_search() {
    local use_aio="${DGAI_USE_AIO:-OFF}"
    echo "=== Building DGAI search (READ_ONLY_TESTS, USE_AIO=${use_aio}) ==="
    export ADDITIONAL_DEFINITIONS=" -DREAD_ONLY_TESTS"
    mkdir -p "$DGAI_ROOT/build"
    cd "$DGAI_ROOT/build"
    cmake .. -DUSE_AIO="$use_aio" && make -j
    cd "$DGAI_ROOT"
}

build_govector_search() {
    echo "=== Building GoVector search ==="
    if [ -z "$GOVECTOR_ROOT" ] || [ ! -d "$GOVECTOR_ROOT" ]; then
        echo "ERROR: GOVECTOR_ROOT is not set or does not exist: ${GOVECTOR_ROOT}" >&2
        return 1
    fi
    mkdir -p "$GOVECTOR_ROOT/build"
    cmake -S "$GOVECTOR_ROOT" -B "$GOVECTOR_ROOT/build" -DCMAKE_BUILD_TYPE="${GOVECTOR_BUILD_TYPE:-Release}"
    cmake --build "$GOVECTOR_ROOT/build" --target search_disk_index -j "${GOVECTOR_BUILD_THREADS:-$(nproc)}"
}

build_dgai_update() {
    local search_mode="${1:-0}"
    if [ "$search_mode" -eq 4 ]; then
        export ADDITIONAL_DEFINITIONS="-DNO_POLLUTE_ORIGINAL -DREORDER_COMPUTE_PQ"
        echo "=== Building DGAI update with REORDER_COMPUTE_PQ (mode=${search_mode}) ==="
    else
        export ADDITIONAL_DEFINITIONS="-DNO_POLLUTE_ORIGINAL"
        echo "=== Building DGAI update without REORDER_COMPUTE_PQ (mode=${search_mode}) ==="
    fi
    mkdir -p "$DGAI_ROOT/build"
    cd "$DGAI_ROOT/build"
    cmake .. && make -j
    cd "$DGAI_ROOT"
}

build_fresh() {
    local use_aio="${USE_AIO:-1}"
    echo "=== Building FreshDiskANN (USE_AIO=${use_aio}) ==="
    mkdir -p "$FRESH_ROOT/build"
    cd "$FRESH_ROOT/build"
    cmake .. -DUSE_AIO="$([[ "$use_aio" == "0" ]] && echo OFF || echo ON)"
    make -j overall_performance
    cd "$DGAI_ROOT"
}

build_greator() {
    echo "=== Building Greator ==="
    mkdir -p "$GREATOR_ROOT/build"
    cd "$GREATOR_ROOT/build"
    cmake ..
    make -j overall_performance
    cd "$DGAI_ROOT"
}

# ---------- Thread patch / restore ----------

THREAD_BACKUP_SUFFIX=".exp_threads.$$.bak"

_backup_file() {
    local f="$1"
    if [ ! -f "${f}${THREAD_BACKUP_SUFFIX}" ]; then
        cp -f "$f" "${f}${THREAD_BACKUP_SUFFIX}"
    fi
}

_restore_file() {
    local f="$1"
    if [ -f "${f}${THREAD_BACKUP_SUFFIX}" ]; then
        mv -f "${f}${THREAD_BACKUP_SUFFIX}" "$f"
    fi
}

restore_all_thread_patches() {
    _restore_file "$DGAI_OP_CPP"
    _restore_file "$FRESH_OP_CPP"
    _restore_file "$FRESH_MERGER_CPP"
    _restore_file "$GREATOR_OP_CPP"
    _restore_file "$GREATOR_MERGER_CPP"
}

# Ensure restore on exit / interrupt while thread experiments run
trap_restore_threads() {
    trap 'restore_all_thread_patches' EXIT INT TERM
}

clear_thread_trap() {
    trap - EXIT INT TERM
}

patch_dgai_threads() {
    local t="$1"
    _backup_file "$DGAI_OP_CPP"
    sed -i "s/#define NUM_INSERT_THREADS [0-9]\+/#define NUM_INSERT_THREADS ${t}/" "$DGAI_OP_CPP"
    sed -i "s/#define NUM_MERGE_THREADS [0-9]\+/#define NUM_MERGE_THREADS ${t}/" "$DGAI_OP_CPP"
    echo "Patched DGAI threads: INSERT=${t} MERGE=${t}"
}

restore_dgai_threads() {
    if [ -f "${DGAI_OP_CPP}${THREAD_BACKUP_SUFFIX}" ]; then
        _restore_file "$DGAI_OP_CPP"
    else
        sed -i "s/#define NUM_INSERT_THREADS [0-9]\+/#define NUM_INSERT_THREADS ${DGAI_DEFAULT_INSERT}/" "$DGAI_OP_CPP"
        sed -i "s/#define NUM_MERGE_THREADS [0-9]\+/#define NUM_MERGE_THREADS ${DGAI_DEFAULT_MERGE}/" "$DGAI_OP_CPP"
    fi
    echo "Restored DGAI threads to defaults"
}

patch_fresh_threads() {
    local t="$1"
    _backup_file "$FRESH_MERGER_CPP"
    _backup_file "$FRESH_OP_CPP"
    sed -i "s/#define MAX_INSERT_THREADS (uint64_t) [0-9]\+/#define MAX_INSERT_THREADS (uint64_t) ${t}/" "$FRESH_MERGER_CPP"
    sed -i "s/#define MAX_N_THREADS (uint64_t) [0-9]\+/#define MAX_N_THREADS (uint64_t) ${t}/" "$FRESH_MERGER_CPP"
    sed -i "s/#define NUM_INSERT_THREADS [0-9]\+/#define NUM_INSERT_THREADS ${t}/" "$FRESH_OP_CPP"
    # Fresh has NUM_DELETE_THREADS; set equal for same-value sweep
    if grep -q '#define NUM_DELETE_THREADS' "$FRESH_OP_CPP"; then
        sed -i "s/#define NUM_DELETE_THREADS [0-9]\+/#define NUM_DELETE_THREADS ${t}/" "$FRESH_OP_CPP"
    fi
    echo "Patched Fresh threads: MAX_INSERT/MAX_N/NUM_INSERT=${t}"
}

restore_fresh_threads() {
    _restore_file "$FRESH_MERGER_CPP"
    _restore_file "$FRESH_OP_CPP"
    echo "Restored Fresh thread macros"
}

patch_greator_threads() {
    local t="$1"
    _backup_file "$GREATOR_MERGER_CPP"
    _backup_file "$GREATOR_OP_CPP"
    sed -i "s/#define MAX_INSERT_THREADS (uint64_t) [0-9]\+/#define MAX_INSERT_THREADS (uint64_t) ${t}/" "$GREATOR_MERGER_CPP"
    sed -i "s/#define MAX_N_THREADS (uint64_t) [0-9]\+/#define MAX_N_THREADS (uint64_t) ${t}/" "$GREATOR_MERGER_CPP"
    sed -i "s/#define NUM_INSERT_THREADS [0-9]\+/#define NUM_INSERT_THREADS ${t}/" "$GREATOR_OP_CPP"
    sed -i "s/#define NUM_MERGE_THREADS [0-9]\+/#define NUM_MERGE_THREADS ${t}/" "$GREATOR_OP_CPP"
    if grep -q '#define NUM_DELETE_THREADS' "$GREATOR_OP_CPP"; then
        sed -i "s/#define NUM_DELETE_THREADS [0-9]\+/#define NUM_DELETE_THREADS ${t}/" "$GREATOR_OP_CPP"
    fi
    echo "Patched Greator threads: INSERT/MERGE/DELETE/MAX_*=${t}"
}

restore_greator_threads() {
    _restore_file "$GREATOR_MERGER_CPP"
    _restore_file "$GREATOR_OP_CPP"
    echo "Restored Greator thread macros"
}

# ---------- Index / truthset checks ----------

index_exists_for() {
    local data_name="$1"
    local prefix
    set_dataset_params "$data_name"
    R=32
    Lbuild=$L_DISK
    prefix=$(benchmark_index_prefix)
    if [ ! -f "${prefix}_disk.index" ] && [ ! -f "${prefix}_disk.graph" ]; then
        echo "WARN: missing disk index for ${data_name}: ${prefix}_disk.index (or _disk.graph)" >&2
        return 1
    fi
    return 0
}

truthset_exists_for() {
    local data_name="$1"
    local ratio="${2:-0.001}"
    set_dataset_params "$data_name"
    local gt_dir="${DATA_PATH}/${data_name}/${data_name}_${batch_npts}_gt_${ratio}"
    if [ ! -d "$gt_dir" ] && [ ! -f "${gt_dir}/gt_0.bin" ]; then
        # accept either directory or bin file prefix
        if [ ! -f "${gt_dir}" ]; then
            echo "WARN: missing truthset for ${data_name} ratio=${ratio}: ${gt_dir}" >&2
            return 1
        fi
    fi
    return 0
}

require_update_truthsets() {
    local truthset_prefix="$1"
    local update_size="$2"
    local num_batches="$3"
    local min_k="${4:-10}"
    local batch offset gt_file gt_nq gt_dim

    for ((batch = 0; batch <= num_batches; ++batch)); do
        offset=$((batch * update_size))
        gt_file="${truthset_prefix}/gt_${offset}.bin"
        if [ ! -s "$gt_file" ]; then
            echo "ERROR: missing update truthset: ${gt_file}" >&2
            return 1
        fi
        read -r gt_nq gt_dim < <(od -An -tu4 -N8 "$gt_file")
        if [ -z "${gt_nq:-}" ] || [ -z "${gt_dim:-}" ] ||
           [ "$gt_nq" -le 0 ] || [ "$gt_dim" -lt "$min_k" ]; then
            echo "ERROR: invalid update truthset header in ${gt_file}: nq=${gt_nq:-?} dim=${gt_dim:-?}" >&2
            return 1
        fi
    done
}

# ---------- Fresh env helpers (without clobbering DGAI_ROOT) ----------

_ensure_fresh_helpers() {
    if declare -F prepare_fresh_work_index >/dev/null 2>&1; then
        return 0
    fi
    # shellcheck source=/dev/null
    source "$FRESH_ROOT/scripts_dgai/env.sh"
    PROJECT_PATH="$DGAI_ROOT"
}

# ---------- Skip helpers (SKIP_EXISTING=1) ----------

_topk_log_recall_valid() {
    local f="$1"
    local top_k="${2:-}"
    local dims
    if [ -z "$top_k" ] && [[ "$(basename "$f")" =~ _K([0-9]+)_ ]]; then
        top_k="${BASH_REMATCH[1]}"
    fi
    [ -n "$top_k" ] || return 1

    # Accept any in-index GT that covers K (e.g. gt_K100.bin has #dims=100).
    # Do not require dims==10000; full-corpus K10000 GTs often contain OOB IDs.
    dims="$(grep -oE '(Metadata|GT metadata): #pts = 10000, #dims = [0-9]+' "$f" | grep -oE '[0-9]+$' | tail -n1 || true)"
    [ -n "$dims" ] || return 1
    [ "$dims" -ge "$top_k" ] || return 1

    grep -q "Recall@${top_k}" "$f" || return 1
    if [ "$top_k" -eq 100 ]; then
        grep -qE 'Decouple max candidates: 100|Search mode: (0|1|2) ' "$f" || return 1
    fi

    awk '
        $1 ~ /^[0-9]+$/ && NF >= 10 {
            value = $NF
            if (value !~ /^[0-9]+([.][0-9]+)?$/) {
                bad = 1
                next
            }
            numeric = value + 0
            if (numeric < 0 || numeric > 100) bad = 1
            if (numeric > 0) positive = 1
            if (numeric > max_recall) max_recall = numeric
            rows++
        }
        END { exit !(rows > 0 && positive && max_recall >= 1 && !bad) }
    ' "$f"
}

# True if any matching log looks successfully finished.
_log_looks_done() {
    local f="$1"
    local kind="${2:-search}"  # search|update
    [ -f "$f" ] || return 1
    [ -s "$f" ] || return 1
    if grep -qiE 'ERROR:|Segmentation fault|Aborted|Killed|Bad file descriptor|Assertion .*failed|terminate called|std::bad_alloc|Read-only file system' "$f"; then
        return 1
    fi
    case "$kind" in
        search)
            # search_disk_index always prints this on clean exit
            grep -qE 'Page cache size|Clearing scratch' "$f" || return 1
            if [[ "$f" == *search_topk* ]]; then
                _topk_log_recall_valid "$f"
            fi
            ;;
        update)
            case "$(basename "$f")" in
                *greator*) grep -qE '^Update over$' "$f" ;;
                *)         grep -qE '^Done$' "$f" ;;
            esac
            ;;
        *)
            return 1
            ;;
    esac
}

_find_done_log() {
    # _find_done_log <kind> <glob>
    local kind="$1"
    local pattern="$2"
    local f
    # nullglob-safe
    shopt -s nullglob
    for f in $pattern; do
        if _log_looks_done "$f" "$kind"; then
            shopt -u nullglob
            echo "$f"
            return 0
        fi
    done
    shopt -u nullglob
    return 1
}

# ---------- Run: search ----------

run_dgai_search() {
    local data_name="$1"
    local search_mode="$2"
    local num_threads="$3"
    local log_dir="$4"

    PROJECT_PATH="$DGAI_ROOT"
    set_dataset_params "$data_name"
    R=32
    Lbuild=$L_DISK
    set_search_width "$search_mode"

    if ! index_exists_for "$data_name"; then
        echo "SKIP search: ${data_name} mode=${search_mode} (missing index)"
        return 0
    fi

    local index_prefix query_file truthset_bin cur_mem_L mem_index_path L log_file cmd existing
    index_prefix=$(benchmark_index_prefix)
    query_file="${DATA_PATH}/${data_name}/${data_name}_query.fbin"
    if [ ! -f "$query_file" ]; then
        query_file="$(resolve_query_fbin "$data_name")" || {
            echo "SKIP search: ${data_name} (missing query)"
            return 0
        }
    fi
    truthset_bin="${DATA_PATH}/${data_name}/${data_name}_${batch_npts}_gt_0.001/gt_0.bin"
    if [ ! -f "$truthset_bin" ]; then
        echo "WARN: truthset bin missing: ${truthset_bin}" >&2
        if ! truthset_exists_for "$data_name" "0.001"; then
            echo "SKIP search: ${data_name} (missing truthset)"
            return 0
        fi
    fi

    cur_mem_L=$MEM_L
    if [ "$search_mode" -ge 4 ] && [ "$search_mode" -le 7 ]; then
        mem_index_path="${index_prefix}_mem_topo.index"
    else
        mem_index_path="${index_prefix}_mem.index"
    fi
    if [ "$cur_mem_L" -ne 0 ] && { [ ! -f "$mem_index_path" ] || [ ! -f "${mem_index_path}.data" ]; }; then
        echo "Warning: memory index missing ($mem_index_path). Set mem_L=0 for this run."
        cur_mem_L=0
    fi

    if [ "$SKIP_EXISTING" = "1" ]; then
        existing="$(_find_done_log search \
            "${log_dir}/${data_name}_search_searchmode${search_mode}_width${search_width}_mem*_thread${num_threads}_*.log" || true)"
        if [ -n "${existing:-}" ]; then
            echo "SKIP search (exists): ${data_name} mode=${search_mode} threads=${num_threads}"
            echo "  prior: ${existing}"
            return 0
        fi
    fi

    L="$(search_L_for_dataset "$data_name" "$num_threads")"
    mkdir -p "$log_dir"
    log_file="${log_dir}/${data_name}_search_searchmode${search_mode}_width${search_width}_mem${cur_mem_L}_thread${num_threads}_$(_exp_ts).log"

    cmd="$DGAI_ROOT/build/tests/search_disk_index \
        float \
        ${index_prefix} \
        ${num_threads} \
        ${search_width} \
        ${query_file} \
        ${truthset_bin} \
        ${K} \
        l2 \
        pq \
        ${search_mode} \
        ${cur_mem_L} \
        ${L}"

    echo "=== SEARCH ${data_name} mode=${search_mode} threads=${num_threads} ==="
    echo "CMD: $cmd"
    echo "LOG: $log_file"
    if ! eval "$cmd" 2>&1 | tee "$log_file"; then
        echo "ERROR: search failed for ${data_name} mode=${search_mode}" >&2
        if [ "$CONTINUE_ON_ERROR" != "1" ]; then
            return 1
        fi
    fi
    return 0
}

_govector_truth_file() {
    local data_name="$1"
    local top_k="${2:-${K:-10}}"
    local f s header nqueries ngt index_npts
    local -a candidates=()
    set_dataset_params "$data_name"
    index_npts="${batch_npts}"

    # Update-style in-index GT first when it covers K (deep1b's plain gt_K10.bin is full-corpus OOB).
    if [ "$top_k" -le 10 ]; then
        candidates+=(
            "${DATA_PATH}/${data_name}/${data_name}_${batch_npts}_gt_0.001/gt_0.bin"
        )
    fi
    # Then smallest dedicated GT that covers K (e.g. gt_K100.bin for K=100).
    for s in 1 10 100 1000 10000; do
        if [ "$s" -ge "$top_k" ]; then
            candidates+=(
                "${DATA_PATH}/${data_name}/gt_K${s}.bin"
                "${DATA_PATH}/${data_name}/${data_name}_gt_K${s}"
            )
        fi
    done

    for f in "${candidates[@]}"; do
        [ -s "$f" ] || continue
        header="$(od -An -tu4 -N8 "$f" | tr -s ' ')"
        read -r nqueries ngt <<<"$header"
        if [ -z "${nqueries:-}" ] || [ -z "${ngt:-}" ] ||
           [ "$nqueries" -ne 10000 ] || [ "$ngt" -lt "$top_k" ]; then
            continue
        fi
        # Reject GTs whose neighbor IDs mostly lie outside the indexed universe.
        if ! python3 - "$f" "$index_npts" <<'PY'
import struct, sys
path, index_npts = sys.argv[1], int(sys.argv[2])
with open(path, "rb") as f:
    nq, dim = struct.unpack("II", f.read(8))
    n_sample = min(nq, 500)
    raw = f.read(n_sample * dim * 4)
ids = memoryview(raw).cast("I")
if not ids:
    sys.exit(1)
oob = sum(1 for x in ids if x >= index_npts) / len(ids)
sys.exit(0 if oob <= 0.01 else 1)
PY
        then
            continue
        fi
        echo "$f"
        return 0
    done

    echo "null"
}

run_govector_search() {
    local data_name="$1"
    local num_threads="$2"
    local log_dir="$3"

    PROJECT_PATH="$DGAI_ROOT"
    set_dataset_params "$data_name"
    R=32
    Lbuild=$L_DISK

    if ! index_exists_for "$data_name"; then
        echo "SKIP govector search: ${data_name} (missing index)"
        return 0
    fi

    local search_bin index_prefix query_file truth_file L log_file existing
    local beamwidth cache_nodes result_dir result_prefix ts exp_name
    search_bin="$GOVECTOR_ROOT/build/apps/search_disk_index"
    if [ ! -x "$search_bin" ]; then
        echo "ERROR: GoVector search binary not found: ${search_bin}" >&2
        return 1
    fi

    index_prefix=$(benchmark_index_prefix)
    query_file=$(resolve_query_fbin "$data_name") || {
        echo "SKIP govector search: ${data_name} (missing query)"
        return 0
    }
    truth_file="$(_govector_truth_file "$data_name")"
    if [ "$truth_file" = "null" ]; then
        echo "WARN: govector truthset missing for ${data_name}; recall will not be computed" >&2
    fi

    beamwidth="$GOVECTOR_BEAMWIDTH"
    cache_nodes="$GOVECTOR_CACHE_NODES"

    if [ "$SKIP_EXISTING" = "1" ]; then
        existing="$(_find_done_log search \
            "${log_dir}/${data_name}_govector_K${K}_W${beamwidth}_cache${cache_nodes}_thread${num_threads}_*.log" || true)"
        if [ -n "${existing:-}" ]; then
            echo "SKIP govector search (exists): ${data_name} threads=${num_threads}"
            echo "  prior: ${existing}"
            return 0
        fi
    fi

    L="$(search_L_for_dataset "$data_name" "$num_threads")"
    mkdir -p "$log_dir"
    ts="$(_exp_ts)"
    exp_name="$(basename "$log_dir")"
    result_dir="${GOVECTOR_ROOT}/results/${exp_name}/${data_name}_K${K}_W${beamwidth}_cache${cache_nodes}_thread${num_threads}_${ts}"
    result_prefix="${result_dir}/result"
    mkdir -p "$result_dir"
    log_file="${log_dir}/${data_name}_govector_K${K}_W${beamwidth}_cache${cache_nodes}_thread${num_threads}_${ts}.log"

    local -a cmd
    cmd=(
        "$search_bin"
        --data_type "$GOVECTOR_DATA_TYPE"
        --dist_fn "$GOVECTOR_DIST_FN"
        --index_path_prefix "$index_prefix"
        --query_file "$query_file"
        --gt_file "$truth_file"
        -K "$K"
        -L
    )
    local l_val
    for l_val in $L; do
        cmd+=("$l_val")
    done
    cmd+=(
        --result_path "$result_prefix"
        --num_nodes_to_cache "$cache_nodes"
        --num_queries "$GOVECTOR_NUM_QUERIES"
        -T "$num_threads"
        -W "$beamwidth"
    )

    echo "=== SEARCH govector ${data_name} threads=${num_threads} ==="
    echo "CMD:"
    printf '%q ' "${cmd[@]}"
    echo
    echo "LOG: $log_file"
    echo "RESULT_PREFIX: $result_prefix"
    if ! "${cmd[@]}" 2>&1 | tee "$log_file"; then
        echo "ERROR: govector search failed for ${data_name}" >&2
        if [ "$CONTINUE_ON_ERROR" != "1" ]; then
            return 1
        fi
    fi
    return 0
}

# ---------- Run: DGAI / Odin update ----------

run_dgai_update() {
    local data_name="$1"
    local search_mode="$2"
    local update_ratio="$3"
    local log_dir="$4"
    local sys_label="${5:-}"  # odin|dgai (optional, for log name)
    local thread_tag="${6:-}" # e.g. ins8_merge8

    PROJECT_PATH="$DGAI_ROOT"
    UPDATE_RATIO="$update_ratio"
    export UPDATE_RATIO

    if [ -z "$sys_label" ]; then
        if [ "$search_mode" -eq 4 ]; then
            sys_label="dgai"
        else
            sys_label="odin"
        fi
    fi

    set_dataset_params "$data_name"
    R=32
    Lbuild=$L_DISK
    set_search_width "$search_mode"
    local beam_width=$search_width

    if ! index_exists_for "$data_name"; then
        echo "SKIP ${sys_label} update: ${data_name} (missing index)"
        return 0
    fi

    local canonical_index_prefix index_prefix data_file query_file truthset_prefix update_size log_file cmd
    data_file=$(resolve_base_fbin "$data_name" "$batch_npts") || {
        echo "SKIP ${sys_label} update: ${data_name} (missing base fbin)"
        return 0
    }
    query_file=$(resolve_query_fbin "$data_name") || {
        echo "SKIP ${sys_label} update: ${data_name} (missing query)"
        return 0
    }
    truthset_prefix="${DATA_PATH}/${data_name}/${data_name}_${batch_npts}_gt_${UPDATE_RATIO}"
    update_size=$(compute_update_size)

    mkdir -p "$log_dir"
    local existing
    if [ "$SKIP_EXISTING" = "1" ]; then
        if [ -n "$thread_tag" ]; then
            existing="$(_find_done_log update \
                "${log_dir}/${data_name}_${sys_label}_mode${search_mode}_ratio${UPDATE_RATIO}_${thread_tag}_*.log" || true)"
        else
            # Avoid matching thread-tagged logs when looking for untags
            existing=""
            shopt -s nullglob
            local cand
            for cand in "${log_dir}/${data_name}_${sys_label}_mode${search_mode}_ratio${UPDATE_RATIO}_"*.log; do
                case "$cand" in
                    *_thread*) continue ;;
                esac
                if _log_looks_done "$cand" update; then
                    existing="$cand"
                    break
                fi
            done
            shopt -u nullglob
        fi
        if [ -n "${existing:-}" ]; then
            echo "SKIP ${sys_label} update (exists): ${data_name} mode=${search_mode} ratio=${UPDATE_RATIO} ${thread_tag}"
            echo "  prior: ${existing}"
            return 0
        fi
    fi

    if ! require_update_truthsets "$truthset_prefix" "$update_size" "$NUM_BATCHES" "$K"; then
        echo "ERROR: ${sys_label} update truthset validation failed for ${data_name} ratio=${UPDATE_RATIO}" >&2
        return 1
    fi

    canonical_index_prefix=$(benchmark_index_prefix)
    if ! copy_shadow_index "$canonical_index_prefix" "$search_mode"; then
        echo "ERROR: ${sys_label} update shadow preparation failed for ${data_name}" >&2
        return 1
    fi
    index_prefix="$index_prefix_shadow"

    if [ -n "$thread_tag" ]; then
        log_file="${log_dir}/${data_name}_${sys_label}_mode${search_mode}_ratio${UPDATE_RATIO}_${thread_tag}_$(_exp_ts).log"
    else
        log_file="${log_dir}/${data_name}_${sys_label}_mode${search_mode}_ratio${UPDATE_RATIO}_$(_exp_ts).log"
    fi

    cmd="$DGAI_ROOT/build/tests/overall_performance \
        float \
        ${data_file} \
        ${L_DISK} \
        ${index_prefix} \
        ${query_file} \
        ${truthset_prefix} \
        ${K} \
        ${beam_width} \
        ${update_size} \
        ${NUM_BATCHES} \
        ${search_mode} \
        ${LSEARCH}"

    echo "=== UPDATE ${sys_label} ${data_name} mode=${search_mode} ratio=${UPDATE_RATIO} update_size=${update_size} ==="
    echo "CMD: $cmd"
    echo "LOG: $log_file"
    if ! eval "$cmd" 2>&1 | tee "$log_file"; then
        echo "ERROR: ${sys_label} update failed for ${data_name}" >&2
        if [ "$CONTINUE_ON_ERROR" != "1" ]; then
            return 1
        fi
    fi
    return 0
}

# ---------- Run: Fresh update ----------

run_fresh_update() {
    local data_name="$1"
    local update_ratio="$2"
    local log_dir="$3"
    local thread_tag="${4:-}"

    _ensure_fresh_helpers
    PROJECT_PATH="$FRESH_ROOT"
    UPDATE_RATIO="$update_ratio"
    export UPDATE_RATIO

    local metric=l2
    local ckpt="${CKPT:-0}"
    local beam_width=4
    local R_DISK=32
    local L_DISK_LOCAL=$L_DISK
    local ALPHA_DISK=1.2
    local NODES_TO_CACHE=0
    local NUM_START=0

    set_dataset_params "$data_name"
    R=$R_DISK
    Lbuild=$L_DISK_LOCAL

    if ! index_exists_for "$data_name"; then
        echo "SKIP fresh update: ${data_name} (missing index)"
        PROJECT_PATH="$DGAI_ROOT"
        return 0
    fi

    local update_size merge_size data_type data_file query_file truthset_prefix canonical_index_prefix save_path log_file cmd skip_search_env
    update_size=$(compute_fresh_update_size)
    merge_size=$(compute_fresh_merge_size)
    data_type=$(fresh_data_type "$data_name")
    data_file=$(resolve_full_base_fbin "$data_name") || {
        echo "SKIP fresh update: ${data_name} (missing full base)"
        PROJECT_PATH="$DGAI_ROOT"
        return 0
    }
    query_file=$(resolve_query_fbin "$data_name") || {
        echo "SKIP fresh update: ${data_name} (missing query)"
        PROJECT_PATH="$DGAI_ROOT"
        return 0
    }
    truthset_prefix=$(fresh_truthset_prefix)
    canonical_index_prefix=$(benchmark_index_prefix)

    PROJECT_PATH="$FRESH_ROOT"
    if ! prepare_fresh_work_index; then
        echo "SKIP fresh update: ${data_name} (shadow prepare failed)"
        PROJECT_PATH="$DGAI_ROOT"
        return 0
    fi
    save_path="$index_prefix_shadow"

    mkdir -p "$log_dir"
    local existing
    if [ "$SKIP_EXISTING" = "1" ]; then
        if [ -n "$thread_tag" ]; then
            existing="$(_find_done_log update \
                "${log_dir}/${data_name}_fresh_ratio${UPDATE_RATIO}_${thread_tag}_*.log" || true)"
        else
            existing=""
            shopt -s nullglob
            local cand
            for cand in "${log_dir}/${data_name}_fresh_ratio${UPDATE_RATIO}_"*.log; do
                case "$cand" in
                    *_thread*) continue ;;
                esac
                if _log_looks_done "$cand" update; then
                    existing="$cand"
                    break
                fi
            done
            shopt -u nullglob
        fi
        if [ -n "${existing:-}" ]; then
            echo "SKIP fresh update (exists): ${data_name} ratio=${UPDATE_RATIO} ${thread_tag}"
            echo "  prior: ${existing}"
            PROJECT_PATH="$DGAI_ROOT"
            return 0
        fi
    fi

    if [ -n "$thread_tag" ]; then
        log_file="${log_dir}/${data_name}_fresh_ratio${UPDATE_RATIO}_${thread_tag}_$(_exp_ts).log"
    else
        log_file="${log_dir}/${data_name}_fresh_ratio${UPDATE_RATIO}_$(_exp_ts).log"
    fi

    skip_search_env=""
    if [ "${UPDATE_SKIP_SEARCH:-1}" = "1" ]; then
        skip_search_env="SKIP_UPDATE_SEARCH=1 "
    fi

    cmd="${skip_search_env}$FRESH_ROOT/build/tests/overall_performance \
        ${data_type} \
        ${data_file} \
        ${L_DISK_LOCAL} \
        ${R_DISK} \
        ${ALPHA_DISK} \
        ${NUM_START} \
        ${NODES_TO_CACHE} \
        ${save_path} \
        ${query_file} \
        ${truthset_prefix} \
        ${K} \
        ${beam_width} \
        ${update_size} \
        ${ckpt} \
        ${NUM_BATCHES} \
        ${merge_size} \
        ${LSEARCH}"

    echo "=== UPDATE fresh ${data_name} ratio=${UPDATE_RATIO} update_size=${update_size} ==="
    echo "CMD: $cmd"
    echo "LOG: $log_file"
    local rc=0
    if ! eval "$cmd" 2>&1 | tee "$log_file"; then
        echo "ERROR: fresh update failed for ${data_name}" >&2
        rc=1
    fi
    PROJECT_PATH="$DGAI_ROOT"
    if [ "$rc" -ne 0 ] && [ "$CONTINUE_ON_ERROR" != "1" ]; then
        return 1
    fi
    return 0
}

# ---------- Run: Greator update ----------

run_greator_update() {
    local data_name="$1"
    local update_ratio="$2"
    local log_dir="$3"
    local thread_tag="${4:-}"

    PROJECT_PATH="$GREATOR_ROOT"
    UPDATE_RATIO="$update_ratio"
    export UPDATE_RATIO

    local metric="${METRIC:-l2}"
    validate_metric "$metric" || return 1
    local search_mode=0
    local id_map=2
    local L_mem=75
    local R_mem=32
    local alpha_mem=1.2
    local R_disk=32
    local alpha_disk=1.2
    local num_shards=100
    local num_nodes_to_cache=10000

    set_dataset_params "$data_name"
    local num_pq_chunks=${n_pq_chunks:-$pq_dim}
    R=32
    Lbuild=$L_DISK
    set_search_width "$search_mode"
    local beam_width=$search_width

    if ! index_exists_for "$data_name"; then
        echo "SKIP greator update: ${data_name} (missing index)"
        PROJECT_PATH="$DGAI_ROOT"
        return 0
    fi

    local canonical_index_prefix index_prefix data_file query_file truthset_prefix update_size merge_th log_file cmd skip_search_env
    canonical_index_prefix=$(benchmark_index_prefix)
    if ! copy_greator_shadow_index "$canonical_index_prefix" "$search_mode"; then
        echo "SKIP greator update: ${data_name} (shadow/sidecars failed)"
        PROJECT_PATH="$DGAI_ROOT"
        return 0
    fi
    index_prefix="$index_prefix_shadow"

    data_file=$(resolve_base_fbin "$data_name" "$batch_npts") || {
        echo "SKIP greator update: ${data_name} (missing base)"
        PROJECT_PATH="$DGAI_ROOT"
        return 0
    }
    query_file=$(resolve_query_fbin "$data_name") || {
        echo "SKIP greator update: ${data_name} (missing query)"
        PROJECT_PATH="$DGAI_ROOT"
        return 0
    }
    truthset_prefix=$(truthset_prefix_path)
    update_size=$(compute_update_size)
    merge_th=$update_size

    mkdir -p "$log_dir"
    local existing
    if [ "$SKIP_EXISTING" = "1" ]; then
        if [ -n "$thread_tag" ]; then
            existing="$(_find_done_log update \
                "${log_dir}/${data_name}_greator_ratio${UPDATE_RATIO}_${thread_tag}_*.log" || true)"
        else
            existing=""
            shopt -s nullglob
            local cand
            for cand in "${log_dir}/${data_name}_greator_ratio${UPDATE_RATIO}_"*.log; do
                case "$cand" in
                    *_thread*) continue ;;
                esac
                if _log_looks_done "$cand" update; then
                    existing="$cand"
                    break
                fi
            done
            shopt -u nullglob
        fi
        if [ -n "${existing:-}" ]; then
            echo "SKIP greator update (exists): ${data_name} ratio=${UPDATE_RATIO} ${thread_tag}"
            echo "  prior: ${existing}"
            PROJECT_PATH="$DGAI_ROOT"
            return 0
        fi
    fi

    if [ -n "$thread_tag" ]; then
        log_file="${log_dir}/${data_name}_greator_ratio${UPDATE_RATIO}_${thread_tag}_$(_exp_ts).log"
    else
        log_file="${log_dir}/${data_name}_greator_ratio${UPDATE_RATIO}_$(_exp_ts).log"
    fi

    skip_search_env=""
    if [ "${UPDATE_SKIP_SEARCH:-1}" = "1" ]; then
        skip_search_env="SKIP_UPDATE_SEARCH=1 "
    fi

    cmd="${skip_search_env}$GREATOR_ROOT/build/tests/overall_performance \
        float ${data_file} ${L_mem} ${R_mem} ${alpha_mem} \
        ${L_DISK} ${R_disk} ${alpha_disk} ${batch_npts} ${num_shards} ${num_pq_chunks} ${num_nodes_to_cache} \
        ${index_prefix} true false ${data_file} ${query_file} ${truthset_prefix} \
        ${K} ${metric} ${LSEARCH} ${beam_width} null 1 ${NUM_BATCHES} ${id_map} ${merge_th}"

    echo "=== UPDATE greator ${data_name} ratio=${UPDATE_RATIO} update_size=${update_size} ==="
    echo "CMD: $cmd"
    echo "LOG: $log_file"
    local rc=0
    if ! eval "$cmd" 2>&1 | tee "$log_file"; then
        echo "ERROR: greator update failed for ${data_name}" >&2
        rc=1
    fi
    PROJECT_PATH="$DGAI_ROOT"
    if [ "$rc" -ne 0 ] && [ "$CONTINUE_ON_ERROR" != "1" ]; then
        return 1
    fi
    return 0
}

# Run one update combo across enabled systems.
# Usage: run_update_all_systems <data_name> <ratio> <log_exp_name> [thread_tag]
run_update_all_systems() {
    local data_name="$1"
    local ratio="$2"
    local exp_name="$3"
    local thread_tag="${4:-}"
    local log_dir

    if _exp_system_enabled odin; then
        log_dir="$(log_dir_for odin "$exp_name")"
        run_dgai_update "$data_name" 0 "$ratio" "$log_dir" "odin" "$thread_tag" || return 1
    fi
    if _exp_system_enabled dgai; then
        log_dir="$(log_dir_for dgai "$exp_name")"
        run_dgai_update "$data_name" 4 "$ratio" "$log_dir" "dgai" "$thread_tag" || return 1
    fi
    if _exp_system_enabled fresh; then
        log_dir="$(log_dir_for fresh "$exp_name")"
        run_fresh_update "$data_name" "$ratio" "$log_dir" "$thread_tag" || return 1
    fi
    if _exp_system_enabled greator; then
        log_dir="$(log_dir_for greator "$exp_name")"
        run_greator_update "$data_name" "$ratio" "$log_dir" "$thread_tag" || return 1
    fi
    return 0
}

# Build update binaries for currently enabled systems.
# Prefer rebuilding DGAI once per mode when needed by callers.
build_enabled_update_systems() {
    local need_odin=0 need_dgai=0
    _exp_system_enabled odin && need_odin=1
    _exp_system_enabled dgai && need_dgai=1

    # Caller typically builds odin/dgai around mode loops; here only non-DGAI.
    if _exp_system_enabled fresh; then
        build_fresh || return 1
    fi
    if _exp_system_enabled greator; then
        build_greator || return 1
    fi
    return 0
}

echo_exp_banner() {
    local name="$1"
    shift
    echo "============================================================"
    echo " Experiment: ${name}"
    echo " $*"
    echo " CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}"
    echo " SKIP_EXISTING=${SKIP_EXISTING}"
    echo " RUN_UPDATE_SYSTEMS=${RUN_UPDATE_SYSTEMS}"
    echo "============================================================"
}
