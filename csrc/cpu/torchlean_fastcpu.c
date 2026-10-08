/*
 * fastcpu: native CPU kernels for TorchLean's eager `Float` runtime.
 *
 * Each kernel computes exactly the IEEE binary64 operations of the Lean
 * specification it replaces, in the same order:
 *
 * - every output element is produced by one thread;
 * - each reduction runs sequentially in index order from +0.0, as
 *   `List.foldl (fun s k => s + a k * b k) 0` does;
 * - products and sums are rounded separately (no FMA contraction; this file
 *   is compiled with -ffp-contract=off and its SIMD paths only use separate
 *   multiply and add instructions).
 *
 * SIMD lanes and threads only ever hold *different* output elements, so the
 * results are bit-identical to the Lean reference for every non-NaN result,
 * whatever the thread count. NaN payload selection is not specified by Lean
 * either and may differ.
 *
 * Environment:
 *   TORCHLEAN_FASTCPU=0        use the Lean reference everywhere
 *   TORCHLEAN_NUM_THREADS=N    worker count (default: online CPUs)
 *   TORCHLEAN_FASTCPU_STATS=1  print per-kernel call counts at exit
 */

#define _GNU_SOURCE
#include <lean/lean.h>

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#include <immintrin.h>
#define TL_HAVE_X86 1
#else
#define TL_HAVE_X86 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#define TL_WEAK __attribute__((weak))
#else
#define TL_WEAK
#endif

/* ---------------------------------------------------------------------- */
/* Configuration and statistics                                           */
/* ---------------------------------------------------------------------- */

/* Statistics slots; must match `TorchLean.FastCpu.Op.*`. */
enum {
  TL_OP_MATMUL = 0,
  TL_OP_MATMUL_BACKWARD = 1,
  TL_OP_ADD = 2,
  TL_OP_SUB = 3,
  TL_OP_MUL = 4,
  TL_OP_SCALE = 5,
  TL_OP_EXP = 6,
  TL_OP_ABS = 7,
  TL_OP_INV = 8,
  TL_OP_SIGN = 9,
  TL_OP_RELU = 10,
  TL_OP_RELU_DERIV = 11,
  TL_OP_BROADCAST = 12,
  TL_OP_REDUCE_BROADCAST = 13,
  TL_OP_SWAP = 14,
  TL_OP_SOFTMAX = 15,
  TL_OP_SOFTMAX_BACKWARD = 16,
  TL_OP_LAYER_NORM = 17,
  TL_OP_LAYER_NORM_BACKWARD = 18,
  TL_OP_COUNT_MAX = 64
};

static const char *tl_op_names[TL_OP_COUNT_MAX] = {
    [TL_OP_MATMUL] = "matmul",
    [TL_OP_MATMUL_BACKWARD] = "matmul_backward",
    [TL_OP_ADD] = "add",
    [TL_OP_SUB] = "sub",
    [TL_OP_MUL] = "mul",
    [TL_OP_SCALE] = "scale",
    [TL_OP_EXP] = "exp",
    [TL_OP_ABS] = "abs",
    [TL_OP_INV] = "inv",
    [TL_OP_SIGN] = "sign",
    [TL_OP_RELU] = "relu",
    [TL_OP_RELU_DERIV] = "relu_deriv",
    [TL_OP_BROADCAST] = "broadcast_to",
    [TL_OP_REDUCE_BROADCAST] = "reduce_from_broadcast",
    [TL_OP_SWAP] = "swap_adjacent_axes",
    [TL_OP_SOFTMAX] = "softmax_last",
    [TL_OP_SOFTMAX_BACKWARD] = "softmax_last_backward",
    [TL_OP_LAYER_NORM] = "layer_norm",
    [TL_OP_LAYER_NORM_BACKWARD] = "layer_norm_backward",
};

static int tl_enabled_flag = 1;
static int tl_stats_flag = 0;
static int tl_debug_flag = 0;
static size_t tl_thread_count = 1;
static atomic_size_t tl_native_calls[TL_OP_COUNT_MAX];
static atomic_size_t tl_fallback_calls[TL_OP_COUNT_MAX];
static pthread_once_t tl_config_once = PTHREAD_ONCE_INIT;

static void tl_print_stats(void) {
  fprintf(stderr, "[fastcpu] threads=%zu enabled=%d\n", tl_thread_count,
          tl_enabled_flag);
  for (int op = 0; op < TL_OP_COUNT_MAX; ++op) {
    size_t native = atomic_load(&tl_native_calls[op]);
    size_t fallback = atomic_load(&tl_fallback_calls[op]);
    if (native == 0 && fallback == 0) {
      continue;
    }
    const char *name = tl_op_names[op] ? tl_op_names[op] : "?";
    fprintf(stderr, "[fastcpu] %-20s native %zu  reference %zu\n", name,
            native, fallback);
  }
}

static void tl_config_init(void) {
  const char *enabled = getenv("TORCHLEAN_FASTCPU");
  if (enabled != NULL && strcmp(enabled, "0") == 0) {
    tl_enabled_flag = 0;
  }
  long cpus = sysconf(_SC_NPROCESSORS_ONLN);
  tl_thread_count = cpus > 0 ? (size_t)cpus : 1;
  const char *threads = getenv("TORCHLEAN_NUM_THREADS");
  if (threads != NULL) {
    long value = strtol(threads, NULL, 10);
    if (value >= 1 && value <= 1024) {
      tl_thread_count = (size_t)value;
    }
  }
  const char *debug = getenv("TORCHLEAN_FASTCPU_DEBUG");
  if (debug != NULL && strcmp(debug, "0") != 0) {
    tl_debug_flag = 1;
  }
  const char *stats = getenv("TORCHLEAN_FASTCPU_STATS");
  if (stats != NULL && strcmp(stats, "0") != 0) {
    tl_stats_flag = 1;
    atexit(tl_print_stats);
  }
}

static void tl_config(void) { pthread_once(&tl_config_once, tl_config_init); }

/* Lean: `TorchLean.FastCpu.enabled : Unit → Bool`. */
LEAN_EXPORT uint8_t torchlean_fastcpu_enabled(lean_obj_arg unit) {
  (void)unit;
  tl_config();
  return (uint8_t)tl_enabled_flag;
}

/*
 * Lean: `TorchLean.FastCpu.noteFallback (op : UInt32) (value : β) : β`.
 * Logically the identity; counts a dispatch that used the Lean reference.
 */
LEAN_EXPORT lean_obj_res torchlean_fastcpu_note_fallback(uint32_t op,
                                                         lean_obj_arg value) {
  tl_config();
  if (op < TL_OP_COUNT_MAX) {
    atomic_fetch_add(&tl_fallback_calls[op], 1);
  }
  return value;
}

static void tl_note_native(int op) {
  if (op >= 0 && op < TL_OP_COUNT_MAX) {
    atomic_fetch_add(&tl_native_calls[op], 1);
  }
}

/*
 * Lean: `TorchLean.FastCpu.isPositiveZero (x : @& α) : Bool`.
 * True only for a boxed binary64 +0.0. Callers have already established that
 * `α` is `Float`; the extra shape checks only defend the FFI boundary.
 */
LEAN_EXPORT uint8_t torchlean_fastcpu_is_positive_zero(b_lean_obj_arg value) {
  if (lean_is_scalar(value) || lean_ptr_tag(value) != 0 ||
      lean_ctor_num_objs(value) != 0) {
    return 0;
  }
  double x = lean_ctor_get_float(value, 0);
  uint64_t bits;
  memcpy(&bits, &x, sizeof bits);
  return bits == 0;
}

/* ---------------------------------------------------------------------- */
/* Thread pool                                                            */
/* ---------------------------------------------------------------------- */

typedef void (*tl_task_fn)(void *ctx, size_t task);

typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t wake;
  pthread_cond_t done;
  size_t generation;
  tl_task_fn fn;
  void *ctx;
  size_t tasks;
  atomic_size_t next;
  size_t active;
  size_t workers;
} tl_pool_t;

static tl_pool_t tl_pool = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .wake = PTHREAD_COND_INITIALIZER,
    .done = PTHREAD_COND_INITIALIZER,
};
static pthread_mutex_t tl_pool_owner = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t tl_pool_once = PTHREAD_ONCE_INIT;

static void tl_run_tasks(tl_task_fn fn, void *ctx, size_t tasks) {
  for (;;) {
    size_t task = atomic_fetch_add(&tl_pool.next, 1);
    if (task >= tasks) {
      return;
    }
    fn(ctx, task);
  }
}

static void *tl_worker(void *arg) {
  (void)arg;
  size_t seen = 0;
  for (;;) {
    pthread_mutex_lock(&tl_pool.mutex);
    while (tl_pool.generation == seen) {
      pthread_cond_wait(&tl_pool.wake, &tl_pool.mutex);
    }
    seen = tl_pool.generation;
    tl_task_fn fn = tl_pool.fn;
    void *ctx = tl_pool.ctx;
    size_t tasks = tl_pool.tasks;
    pthread_mutex_unlock(&tl_pool.mutex);

    tl_run_tasks(fn, ctx, tasks);

    pthread_mutex_lock(&tl_pool.mutex);
    if (--tl_pool.active == 0) {
      pthread_cond_signal(&tl_pool.done);
    }
    pthread_mutex_unlock(&tl_pool.mutex);
  }
  return NULL;
}

static void tl_pool_init(void) {
  tl_config();
  size_t workers = tl_thread_count > 1 ? tl_thread_count - 1 : 0;
  size_t started = 0;
  for (size_t i = 0; i < workers; ++i) {
    pthread_t thread;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    if (pthread_create(&thread, &attr, tl_worker, NULL) == 0) {
      ++started;
    }
    pthread_attr_destroy(&attr);
  }
  tl_pool.workers = started;
}

/*
 * Run fn(ctx, 0..tasks-1) on the pool; the caller participates. Concurrent or
 * nested calls run serially on the calling thread instead of waiting.
 */
static void tl_parallel(size_t tasks, tl_task_fn fn, void *ctx) {
  tl_config();
  if (tasks == 0) {
    return;
  }
  if (tasks == 1 || tl_thread_count <= 1 ||
      pthread_mutex_trylock(&tl_pool_owner) != 0) {
    for (size_t task = 0; task < tasks; ++task) {
      fn(ctx, task);
    }
    return;
  }
  pthread_once(&tl_pool_once, tl_pool_init);
  if (tl_pool.workers == 0) {
    pthread_mutex_unlock(&tl_pool_owner);
    for (size_t task = 0; task < tasks; ++task) {
      fn(ctx, task);
    }
    return;
  }
  pthread_mutex_lock(&tl_pool.mutex);
  tl_pool.fn = fn;
  tl_pool.ctx = ctx;
  tl_pool.tasks = tasks;
  atomic_store(&tl_pool.next, 0);
  tl_pool.active = tl_pool.workers;
  tl_pool.generation++;
  pthread_cond_broadcast(&tl_pool.wake);
  pthread_mutex_unlock(&tl_pool.mutex);

  tl_run_tasks(fn, ctx, tasks);

  pthread_mutex_lock(&tl_pool.mutex);
  while (tl_pool.active != 0) {
    pthread_cond_wait(&tl_pool.done, &tl_pool.mutex);
  }
  pthread_mutex_unlock(&tl_pool.mutex);
  pthread_mutex_unlock(&tl_pool_owner);
}

/* ---------------------------------------------------------------------- */
/* Buffers                                                                */
/* ---------------------------------------------------------------------- */

static lean_obj_res tl_alloc_floats(size_t size) {
  if (lean_alloc_sarray_would_overflow(sizeof(double), size)) {
    lean_internal_panic_out_of_memory();
  }
  return lean_alloc_sarray(sizeof(double), size, size);
}

static size_t tl_mul_checked(size_t a, size_t b) {
  if (a != 0 && b > SIZE_MAX / a) {
    lean_internal_panic("torchlean fastcpu: size overflow");
  }
  return a * b;
}

/* Largest element index touched by a strided [batch, rows, cols] view. */
static size_t tl_view_extent(size_t batch, size_t rows, size_t cols,
                             size_t batch_stride, size_t row_stride,
                             size_t col_stride) {
  if (batch == 0 || rows == 0 || cols == 0) {
    return 0;
  }
  size_t last = tl_mul_checked(batch - 1, batch_stride);
  last += tl_mul_checked(rows - 1, row_stride);
  last += tl_mul_checked(cols - 1, col_stride);
  return last + 1;
}

static void tl_check_view(b_lean_obj_arg buffer, size_t batch, size_t rows,
                          size_t cols, size_t batch_stride, size_t row_stride,
                          size_t col_stride) {
  size_t extent = tl_view_extent(batch, rows, cols, batch_stride, row_stride,
                                 col_stride);
  if (extent > lean_sarray_size(buffer)) {
    lean_internal_panic("torchlean fastcpu: strided view out of bounds");
  }
}

/* ---------------------------------------------------------------------- */
/* Matrix multiplication                                                  */
/* ---------------------------------------------------------------------- */

/*
 * out[b, i, j] = sum_{k < K} L(b, i, k) * R(b, k, j), summed in increasing k
 * from +0.0. L is an arbitrary strided view; R must have unit column stride
 * (callers pack it first otherwise). out is contiguous [batch, M, N].
 */
typedef struct {
  const double *l;
  size_t l_bs, l_rs, l_cs;
  const double *r;
  size_t r_bs, r_rs;
  double *out;
  size_t m, k, n;
  size_t row_block, col_block;
  size_t row_blocks, col_blocks;
} tl_mm_job_t;

/*
 * Tiles cover contraction indices [k0, k1). With `accumulate` the partial sums
 * stored in `out` by the previous block are continued, so splitting the
 * contraction into blocks keeps each output's sequential summation order.
 */
static void tl_mm_scalar_tile(const tl_mm_job_t *job, const double *l,
                              const double *r, double *out, size_t i0,
                              size_t i1, size_t j0, size_t j1, size_t k0,
                              size_t k1, int accumulate) {
  for (size_t i = i0; i < i1; ++i) {
    const double *li = l + i * job->l_rs;
    for (size_t j = j0; j < j1; ++j) {
      double acc = accumulate ? out[i * job->n + j] : 0.0;
      for (size_t k = k0; k < k1; ++k) {
        acc = acc + li[k * job->l_cs] * r[k * job->r_rs + j];
      }
      out[i * job->n + j] = acc;
    }
  }
}

#if TL_HAVE_X86
#define TL_AVX2 __attribute__((target("avx2")))

/* 4 rows x 8 columns: eight 4-lane accumulators. */
#define TL_INIT(acc, ptr) ((acc) ? _mm256_loadu_pd(ptr) : _mm256_setzero_pd())

TL_AVX2 static void tl_mm_avx2_4x8(const double *l, size_t l_rs, size_t l_cs,
                                    const double *r, size_t r_rs, double *out,
                                    size_t ldo, size_t kk, int acc) {
  __m256d c00 = TL_INIT(acc, out), c01 = TL_INIT(acc, out + 4);
  __m256d c10 = TL_INIT(acc, out + ldo), c11 = TL_INIT(acc, out + ldo + 4);
  __m256d c20 = TL_INIT(acc, out + 2 * ldo);
  __m256d c21 = TL_INIT(acc, out + 2 * ldo + 4);
  __m256d c30 = TL_INIT(acc, out + 3 * ldo);
  __m256d c31 = TL_INIT(acc, out + 3 * ldo + 4);
  const double *l0 = l, *l1 = l + l_rs, *l2 = l + 2 * l_rs,
               *l3 = l + 3 * l_rs;
  for (size_t k = 0; k < kk; ++k) {
    const double *rk = r + k * r_rs;
    __m256d b0 = _mm256_loadu_pd(rk);
    __m256d b1 = _mm256_loadu_pd(rk + 4);
    size_t lk = k * l_cs;
    __m256d a = _mm256_broadcast_sd(l0 + lk);
    c00 = _mm256_add_pd(c00, _mm256_mul_pd(a, b0));
    c01 = _mm256_add_pd(c01, _mm256_mul_pd(a, b1));
    a = _mm256_broadcast_sd(l1 + lk);
    c10 = _mm256_add_pd(c10, _mm256_mul_pd(a, b0));
    c11 = _mm256_add_pd(c11, _mm256_mul_pd(a, b1));
    a = _mm256_broadcast_sd(l2 + lk);
    c20 = _mm256_add_pd(c20, _mm256_mul_pd(a, b0));
    c21 = _mm256_add_pd(c21, _mm256_mul_pd(a, b1));
    a = _mm256_broadcast_sd(l3 + lk);
    c30 = _mm256_add_pd(c30, _mm256_mul_pd(a, b0));
    c31 = _mm256_add_pd(c31, _mm256_mul_pd(a, b1));
  }
  _mm256_storeu_pd(out, c00);
  _mm256_storeu_pd(out + 4, c01);
  _mm256_storeu_pd(out + ldo, c10);
  _mm256_storeu_pd(out + ldo + 4, c11);
  _mm256_storeu_pd(out + 2 * ldo, c20);
  _mm256_storeu_pd(out + 2 * ldo + 4, c21);
  _mm256_storeu_pd(out + 3 * ldo, c30);
  _mm256_storeu_pd(out + 3 * ldo + 4, c31);
}

/* 4 rows x 4 columns. */
TL_AVX2 static void tl_mm_avx2_4x4(const double *l, size_t l_rs, size_t l_cs,
                                    const double *r, size_t r_rs, double *out,
                                    size_t ldo, size_t kk, int acc) {
  __m256d c0 = TL_INIT(acc, out), c1 = TL_INIT(acc, out + ldo);
  __m256d c2 = TL_INIT(acc, out + 2 * ldo), c3 = TL_INIT(acc, out + 3 * ldo);
  const double *l0 = l, *l1 = l + l_rs, *l2 = l + 2 * l_rs,
               *l3 = l + 3 * l_rs;
  for (size_t k = 0; k < kk; ++k) {
    __m256d b = _mm256_loadu_pd(r + k * r_rs);
    size_t lk = k * l_cs;
    c0 = _mm256_add_pd(c0, _mm256_mul_pd(_mm256_broadcast_sd(l0 + lk), b));
    c1 = _mm256_add_pd(c1, _mm256_mul_pd(_mm256_broadcast_sd(l1 + lk), b));
    c2 = _mm256_add_pd(c2, _mm256_mul_pd(_mm256_broadcast_sd(l2 + lk), b));
    c3 = _mm256_add_pd(c3, _mm256_mul_pd(_mm256_broadcast_sd(l3 + lk), b));
  }
  _mm256_storeu_pd(out, c0);
  _mm256_storeu_pd(out + ldo, c1);
  _mm256_storeu_pd(out + 2 * ldo, c2);
  _mm256_storeu_pd(out + 3 * ldo, c3);
}

/* 1 row x 8 columns. */
TL_AVX2 static void tl_mm_avx2_1x8(const double *l, size_t l_cs,
                                    const double *r, size_t r_rs, double *out,
                                    size_t kk, int acc) {
  __m256d c0 = TL_INIT(acc, out), c1 = TL_INIT(acc, out + 4);
  for (size_t k = 0; k < kk; ++k) {
    const double *rk = r + k * r_rs;
    __m256d a = _mm256_broadcast_sd(l + k * l_cs);
    c0 = _mm256_add_pd(c0, _mm256_mul_pd(a, _mm256_loadu_pd(rk)));
    c1 = _mm256_add_pd(c1, _mm256_mul_pd(a, _mm256_loadu_pd(rk + 4)));
  }
  _mm256_storeu_pd(out, c0);
  _mm256_storeu_pd(out + 4, c1);
}

/* 1 row x 4 columns. */
TL_AVX2 static void tl_mm_avx2_1x4(const double *l, size_t l_cs,
                                    const double *r, size_t r_rs, double *out,
                                    size_t kk, int acc) {
  __m256d c0 = TL_INIT(acc, out);
  for (size_t k = 0; k < kk; ++k) {
    __m256d a = _mm256_broadcast_sd(l + k * l_cs);
    c0 = _mm256_add_pd(c0, _mm256_mul_pd(a, _mm256_loadu_pd(r + k * r_rs)));
  }
  _mm256_storeu_pd(out, c0);
}

TL_AVX2 static void tl_mm_avx2_tile(const tl_mm_job_t *job, const double *l,
                                     const double *r, double *out, size_t i0,
                                     size_t i1, size_t j0, size_t j1, size_t k0,
                                     size_t k1, int acc) {
  size_t kk = k1 - k0, ldo = job->n;
  const double *lk = l + k0 * job->l_cs;
  const double *rk = r + k0 * job->r_rs;
  size_t i = i0;
  for (; i + 4 <= i1; i += 4) {
    const double *li = lk + i * job->l_rs;
    size_t j = j0;
    for (; j + 8 <= j1; j += 8) {
      tl_mm_avx2_4x8(li, job->l_rs, job->l_cs, rk + j, job->r_rs,
                     out + i * ldo + j, ldo, kk, acc);
    }
    for (; j + 4 <= j1; j += 4) {
      tl_mm_avx2_4x4(li, job->l_rs, job->l_cs, rk + j, job->r_rs,
                     out + i * ldo + j, ldo, kk, acc);
    }
    if (j < j1) {
      tl_mm_scalar_tile(job, l, r, out, i, i + 4, j, j1, k0, k1, acc);
    }
  }
  for (; i < i1; ++i) {
    const double *li = lk + i * job->l_rs;
    size_t j = j0;
    for (; j + 8 <= j1; j += 8) {
      tl_mm_avx2_1x8(li, job->l_cs, rk + j, job->r_rs, out + i * ldo + j, kk,
                     acc);
    }
    for (; j + 4 <= j1; j += 4) {
      tl_mm_avx2_1x4(li, job->l_cs, rk + j, job->r_rs, out + i * ldo + j, kk,
                     acc);
    }
    if (j < j1) {
      tl_mm_scalar_tile(job, l, r, out, i, i + 1, j, j1, k0, k1, acc);
    }
  }
}

/* CPUID + XGETBV directly, so the object needs no libgcc CPU-model symbols. */
static int tl_detect_avx2(void) {
  unsigned a, b, c, d;
  if (!__get_cpuid(1, &a, &b, &c, &d)) {
    return 0;
  }
  if (!((c >> 27) & 1) || !((c >> 28) & 1)) { /* OSXSAVE, AVX */
    return 0;
  }
  unsigned xcr0_lo, xcr0_hi;
  __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
  (void)xcr0_hi;
  if ((xcr0_lo & 6) != 6) { /* XMM and YMM state enabled by the OS */
    return 0;
  }
  if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
    return 0;
  }
  return (b >> 5) & 1;
}

static int tl_cpu_has_avx2(void) {
  static atomic_int cached = -1;
  int value = atomic_load_explicit(&cached, memory_order_relaxed);
  if (value < 0) {
    value = tl_detect_avx2();
    atomic_store_explicit(&cached, value, memory_order_relaxed);
  }
  return value;
}
#endif

static void tl_mm_task(void *ctx, size_t task) {
  const tl_mm_job_t *job = ctx;
  size_t per_batch = job->row_blocks * job->col_blocks;
  size_t b = task / per_batch;
  size_t rest = task % per_batch;
  size_t rb = rest / job->col_blocks;
  size_t cb = rest % job->col_blocks;
  size_t i0 = rb * job->row_block;
  size_t i1 = i0 + job->row_block < job->m ? i0 + job->row_block : job->m;
  size_t j0 = cb * job->col_block;
  size_t j1 = j0 + job->col_block < job->n ? j0 + job->col_block : job->n;
  const double *l = job->l + b * job->l_bs;
  const double *r = job->r + b * job->r_bs;
  double *out = job->out + b * job->m * job->n;
  /* Contraction blocks keep a tile's operands cache-resident for long k. */
  size_t kc = job->k > 1024 ? 256 : (job->k ? job->k : 1);
  for (size_t k0 = 0; k0 < job->k || k0 == 0; k0 += kc) {
    size_t k1 = k0 + kc < job->k ? k0 + kc : job->k;
    int acc = k0 > 0;
#if TL_HAVE_X86
    if (tl_cpu_has_avx2()) {
      tl_mm_avx2_tile(job, l, r, out, i0, i1, j0, j1, k0, k1, acc);
    } else
#endif
    {
      tl_mm_scalar_tile(job, l, r, out, i0, i1, j0, j1, k0, k1, acc);
    }
    if (k1 >= job->k) {
      break;
    }
  }
}

/* Copy a strided [batch, rows, cols] view into contiguous storage. */
typedef struct {
  const double *src;
  size_t bs, rs, cs;
  double *dst;
  size_t rows, cols;
} tl_pack_job_t;

static void tl_pack_task(void *ctx, size_t b) {
  const tl_pack_job_t *job = ctx;
  const double *src = job->src + b * job->bs;
  double *dst = job->dst + b * job->rows * job->cols;
  for (size_t i = 0; i < job->rows; ++i) {
    for (size_t j = 0; j < job->cols; ++j) {
      dst[i * job->cols + j] = src[i * job->rs + j * job->cs];
    }
  }
}

static void tl_choose_blocks(tl_mm_job_t *job, size_t batch) {
  /* Aim for >= 8 tasks per thread, each with >= ~64k multiply-adds. */
  size_t threads = tl_thread_count;
  size_t target_tasks = threads * 8;
  if (job->k > 1024) {
    /* Long reductions cannot be split; use small output blocks instead. */
    target_tasks = threads * 2;
  }
  size_t row_block = job->m;
  size_t col_block = job->n;
  size_t tasks = batch;
  while (tasks < target_tasks) {
    size_t work = row_block * col_block * (job->k ? job->k : 1);
    if (work < 2 * 65536) {
      break;
    }
    if (row_block >= 8 && (row_block >= col_block || col_block < 16)) {
      row_block = (row_block / 2 + 3) / 4 * 4;
    } else if (col_block >= 16) {
      col_block = (col_block / 2 + 7) / 8 * 8;
    } else {
      break;
    }
    size_t rbs = (job->m + row_block - 1) / row_block;
    size_t cbs = (job->n + col_block - 1) / col_block;
    tasks = batch * rbs * cbs;
  }
  job->row_block = row_block ? row_block : 1;
  job->col_block = col_block ? col_block : 1;
  job->row_blocks = (job->m + job->row_block - 1) / job->row_block;
  job->col_blocks = (job->n + job->col_block - 1) / job->col_block;
}

static lean_obj_res tl_bmm(int op, b_lean_obj_arg lhs, size_t l_bs, size_t l_rs,
                           size_t l_cs, b_lean_obj_arg rhs, size_t r_bs,
                           size_t r_rs, size_t r_cs, size_t batch, size_t m,
                           size_t k, size_t n) {
  tl_config();
  tl_check_view(lhs, batch, m, k, l_bs, l_rs, l_cs);
  tl_check_view(rhs, batch, k, n, r_bs, r_rs, r_cs);
  size_t out_size = tl_mul_checked(tl_mul_checked(batch, m), n);
  lean_object *out = tl_alloc_floats(out_size);
  tl_note_native(op);
  if (out_size == 0) {
    return out;
  }
  const double *r = lean_float_array_cptr(rhs);
  double *packed = NULL;
  if (r_cs != 1) {
    size_t packed_batches = r_bs == 0 ? 1 : batch;
    packed = malloc(tl_mul_checked(tl_mul_checked(packed_batches, k), n) *
                    sizeof(double));
    if (packed == NULL) {
      lean_internal_panic_out_of_memory();
    }
    tl_pack_job_t pack = {r, r_bs, r_rs, r_cs, packed, k, n};
    tl_parallel(packed_batches, tl_pack_task, &pack);
    r = packed;
    r_bs = r_bs == 0 ? 0 : k * n;
    r_rs = n;
  }
  tl_mm_job_t job = {
      .l = lean_float_array_cptr(lhs),
      .l_bs = l_bs,
      .l_rs = l_rs,
      .l_cs = l_cs,
      .r = r,
      .r_bs = r_bs,
      .r_rs = r_rs,
      .out = lean_float_array_cptr(out),
      .m = m,
      .k = k,
      .n = n,
  };
  tl_choose_blocks(&job, batch);
  tl_parallel(batch * job.row_blocks * job.col_blocks, tl_mm_task, &job);
  free(packed);
  return out;
}

/*
 * Lean: `TorchLean.FastCpu.bmmNative`.
 * out[b] = L[b] @ R[b] for strided views; all strides in elements.
 */
LEAN_EXPORT lean_obj_res torchlean_fastcpu_bmm(
    b_lean_obj_arg lhs, size_t l_bs, size_t l_rs, size_t l_cs,
    b_lean_obj_arg rhs, size_t r_bs, size_t r_rs, size_t r_cs, size_t batch,
    size_t m, size_t k, size_t n) {
  return tl_bmm(TL_OP_MATMUL, lhs, l_bs, l_rs, l_cs, rhs, r_bs, r_rs, r_cs,
                batch, m, k, n);
}

/* Interpreter entry point; compiled callers use the unboxed symbol. */
LEAN_EXPORT TL_WEAK lean_obj_res lp_TorchLean_TorchLean_FastCpu_bmmNative___boxed(
    lean_obj_arg lhs, lean_obj_arg l_bs, lean_obj_arg l_rs, lean_obj_arg l_cs,
    lean_obj_arg rhs, lean_obj_arg r_bs, lean_obj_arg r_rs, lean_obj_arg r_cs,
    lean_obj_arg batch, lean_obj_arg m, lean_obj_arg k, lean_obj_arg n) {
  lean_object *result = torchlean_fastcpu_bmm(
      lhs, lean_unbox_usize(l_bs), lean_unbox_usize(l_rs),
      lean_unbox_usize(l_cs), rhs, lean_unbox_usize(r_bs),
      lean_unbox_usize(r_rs), lean_unbox_usize(r_cs), lean_unbox_usize(batch),
      lean_unbox_usize(m), lean_unbox_usize(k), lean_unbox_usize(n));
  lean_dec(l_bs);
  lean_dec(l_rs);
  lean_dec(l_cs);
  lean_dec(r_bs);
  lean_dec(r_rs);
  lean_dec(r_cs);
  lean_dec(batch);
  lean_dec(m);
  lean_dec(k);
  lean_dec(n);
  lean_dec_ref(rhs);
  lean_dec_ref(lhs);
  return result;
}

/*
 * Lean: `TorchLean.FastCpu.bmmBackwardNative`; counted separately so the
 * statistics distinguish forward and backward products.
 */
LEAN_EXPORT lean_obj_res torchlean_fastcpu_bmm_backward(
    b_lean_obj_arg lhs, size_t l_bs, size_t l_rs, size_t l_cs,
    b_lean_obj_arg rhs, size_t r_bs, size_t r_rs, size_t r_cs, size_t batch,
    size_t m, size_t k, size_t n) {
  return tl_bmm(TL_OP_MATMUL_BACKWARD, lhs, l_bs, l_rs, l_cs, rhs, r_bs, r_rs,
                r_cs, batch, m, k, n);
}

LEAN_EXPORT TL_WEAK lean_obj_res lp_TorchLean_TorchLean_FastCpu_bmmBackwardNative___boxed(
    lean_obj_arg lhs, lean_obj_arg l_bs, lean_obj_arg l_rs, lean_obj_arg l_cs,
    lean_obj_arg rhs, lean_obj_arg r_bs, lean_obj_arg r_rs, lean_obj_arg r_cs,
    lean_obj_arg batch, lean_obj_arg m, lean_obj_arg k, lean_obj_arg n) {
  lean_object *result = torchlean_fastcpu_bmm_backward(
      lhs, lean_unbox_usize(l_bs), lean_unbox_usize(l_rs),
      lean_unbox_usize(l_cs), rhs, lean_unbox_usize(r_bs),
      lean_unbox_usize(r_rs), lean_unbox_usize(r_cs), lean_unbox_usize(batch),
      lean_unbox_usize(m), lean_unbox_usize(k), lean_unbox_usize(n));
  lean_dec(l_bs);
  lean_dec(l_rs);
  lean_dec(l_cs);
  lean_dec(r_bs);
  lean_dec(r_rs);
  lean_dec(r_cs);
  lean_dec(batch);
  lean_dec(m);
  lean_dec(k);
  lean_dec(n);
  lean_dec_ref(rhs);
  lean_dec_ref(lhs);
  return result;
}

/* Interpreter entry points for the dispatch helpers. */
LEAN_EXPORT TL_WEAK lean_obj_res
lp_TorchLean_TorchLean_FastCpu_enabled___boxed(lean_obj_arg unit) {
  return lean_box(torchlean_fastcpu_enabled(unit));
}

LEAN_EXPORT TL_WEAK lean_obj_res
lp_TorchLean_TorchLean_FastCpu_noteFallback___boxed(lean_obj_arg type,
                                                    lean_obj_arg op,
                                                    lean_obj_arg value) {
  (void)type;
  return torchlean_fastcpu_note_fallback(lean_unbox_uint32(op), value);
}

LEAN_EXPORT TL_WEAK lean_obj_res
lp_TorchLean_TorchLean_FastCpu_isPositiveZero___boxed(lean_obj_arg type,
                                                      lean_obj_arg value) {
  (void)type;
  uint8_t result = torchlean_fastcpu_is_positive_zero(value);
  lean_dec(value);
  return lean_box(result);
}

/* ---------------------------------------------------------------------- */
/* Backend identification                                                 */
/* ---------------------------------------------------------------------- */

#include <dlfcn.h>
#include <math.h>

/*
 * Structural equality of two runtime values: the same pointer, or closures
 * with the same code pointer, arity and (recursively equal) captured
 * arguments, or constructor objects with the same tag, equal object fields and
 * equal scalar bytes. Equal runtime representations evaluate identically, so
 * this is a sound sufficient test that two scalar dictionaries are the same
 * Lean value; anything unrecognised (or too deep) compares unequal.
 */
static int tl_same(b_lean_obj_arg a, b_lean_obj_arg b, int depth) {
  if (a == b) {
    return 1;
  }
  if (lean_is_scalar(a) || lean_is_scalar(b) || depth == 0) {
    return 0;
  }
  unsigned tag = lean_ptr_tag(a);
  if (tag != lean_ptr_tag(b)) {
    return 0;
  }
  if (tag == LeanClosure) {
    if (lean_closure_fun(a) != lean_closure_fun(b) ||
        lean_closure_arity(a) != lean_closure_arity(b) ||
        lean_closure_num_fixed(a) != lean_closure_num_fixed(b)) {
      return 0;
    }
    lean_object **fa = lean_closure_arg_cptr(a);
    lean_object **fb = lean_closure_arg_cptr(b);
    for (unsigned i = 0; i < lean_closure_num_fixed(a); ++i) {
      if (!tl_same(fa[i], fb[i], depth - 1)) {
        return 0;
      }
    }
    return 1;
  }
  if (tag <= LeanMaxCtorTag) {
    unsigned objs = lean_ctor_num_objs(a);
    if (objs != lean_ctor_num_objs(b)) {
      return 0;
    }
    size_t size_a = lean_object_byte_size(a);
    if (size_a != lean_object_byte_size(b)) {
      return 0;
    }
    for (unsigned i = 0; i < objs; ++i) {
      if (!tl_same(lean_ctor_get(a, i), lean_ctor_get(b, i), depth - 1)) {
        return 0;
      }
    }
    size_t offset = sizeof(lean_ctor_object) + objs * sizeof(void *);
    return size_a <= offset ||
           memcmp((const char *)a + offset, (const char *)b + offset,
                  size_a - offset) == 0;
  }
  return 0;
}

static void tl_describe(const char *side, b_lean_obj_arg value) {
  if (lean_is_scalar(value)) {
    fprintf(stderr, "    %s: scalar %zu\n", side, lean_unbox(value));
    return;
  }
  unsigned tag = lean_ptr_tag(value);
  if (tag == LeanClosure) {
    Dl_info info;
    void *fun = lean_closure_fun(value);
    const char *name = dladdr(fun, &info) && info.dli_sname ? info.dli_sname
                                                            : "?";
    fprintf(stderr, "    %s: closure %s (%p) arity %u fixed %u\n", side, name,
            fun, lean_closure_arity(value), lean_closure_num_fixed(value));
  } else if (tag <= LeanMaxCtorTag) {
    fprintf(stderr, "    %s: ctor tag %u objs %u bytes %zu at %p\n", side, tag,
            lean_ctor_num_objs(value), lean_object_byte_size(value),
            (void *)value);
  } else {
    fprintf(stderr, "    %s: object tag %u\n", side, tag);
  }
}

/* Lean: `TorchLean.FastCpu.sameValue (a : @& α) (b : @& β) : Bool`. */
LEAN_EXPORT uint8_t torchlean_fastcpu_same(b_lean_obj_arg a,
                                           b_lean_obj_arg b) {
  tl_config();
  int same = tl_same(a, b, 8);
  if (!same && tl_debug_flag) {
    fprintf(stderr, "[fastcpu] dictionary mismatch\n");
    tl_describe("candidate", a);
    tl_describe("reference", b);
    if (!lean_is_scalar(a) && !lean_is_scalar(b) &&
        lean_ptr_tag(a) <= LeanMaxCtorTag &&
        lean_ptr_tag(a) == lean_ptr_tag(b)) {
      unsigned objs = lean_ctor_num_objs(a);
      for (unsigned i = 0; i < objs && i < lean_ctor_num_objs(b); ++i) {
        if (!tl_same(lean_ctor_get(a, i), lean_ctor_get(b, i), 7)) {
          fprintf(stderr, "    first differing field %u\n", i);
          tl_describe("candidate field", lean_ctor_get(a, i));
          tl_describe("reference field", lean_ctor_get(b, i));
          break;
        }
      }
    }
  }
  return (uint8_t)same;
}

LEAN_EXPORT TL_WEAK lean_obj_res
lp_TorchLean_TorchLean_FastCpu_sameValue___boxed(lean_obj_arg ta,
                                                 lean_obj_arg tb,
                                                 lean_obj_arg a,
                                                 lean_obj_arg b) {
  (void)ta;
  (void)tb;
  uint8_t result = torchlean_fastcpu_same(a, b);
  lean_dec(a);
  lean_dec(b);
  return lean_box(result);
}

/* ---------------------------------------------------------------------- */
/* Elementwise kernels                                                    */
/* ---------------------------------------------------------------------- */

#define TL_CHUNK ((size_t)1 << 15)

static size_t tl_chunks(size_t n) { return (n + TL_CHUNK - 1) / TL_CHUNK; }

static double tl_float_of_boxed(b_lean_obj_arg value) {
  if (lean_is_scalar(value) || lean_ptr_tag(value) != 0 ||
      lean_ctor_num_objs(value) != 0) {
    lean_internal_panic("torchlean fastcpu: expected a boxed Float");
  }
  return lean_ctor_get_float(value, 0);
}

/*
 * libm's exp, with the underflow branch short-cut: for x < -1000 the exact
 * result rounds to +0.0, which is what exp returns (after a slow errno path).
 * errno is not observable from Lean.
 */
static inline double tl_exp(double x) { return x < -1000.0 ? 0.0 : exp(x); }

typedef struct {
  int op;
  const double *a, *b;
  double *out;
  size_t n;
  double c;
} tl_ew_job_t;

/* Each case is the scalar body of the corresponding Lean specification. */
static void tl_ew_task(void *ctx, size_t task) {
  const tl_ew_job_t *job = ctx;
  size_t i0 = task * TL_CHUNK;
  size_t i1 = i0 + TL_CHUNK < job->n ? i0 + TL_CHUNK : job->n;
  const double *a = job->a, *b = job->b;
  double *out = job->out;
  switch (job->op) {
  case TL_OP_ADD: /* addSpec: x + y */
    for (size_t i = i0; i < i1; ++i) out[i] = a[i] + b[i];
    break;
  case TL_OP_SUB: /* subSpec: x - y */
    for (size_t i = i0; i < i1; ++i) out[i] = a[i] - b[i];
    break;
  case TL_OP_MUL: /* mulSpec: x * y */
    for (size_t i = i0; i < i1; ++i) out[i] = a[i] * b[i];
    break;
  case TL_OP_SCALE: /* scaleSpec: x * c */
    for (size_t i = i0; i < i1; ++i) out[i] = a[i] * job->c;
    break;
  case TL_OP_EXP: /* expSpec: Float.exp */
    for (size_t i = i0; i < i1; ++i) out[i] = tl_exp(a[i]);
    break;
  case TL_OP_ABS: /* absSpec: Float.abs */
    for (size_t i = i0; i < i1; ++i) out[i] = fabs(a[i]);
    break;
  case TL_OP_INV: /* invSpec: 1 / x */
    for (size_t i = i0; i < i1; ++i) out[i] = 1.0 / a[i];
    break;
  case TL_OP_SIGN: /* signSpec: if x > 0 then 1 else if x < 0 then -1 else 0 */
    for (size_t i = i0; i < i1; ++i)
      out[i] = a[i] > 0.0 ? 1.0 : (a[i] < 0.0 ? -1.0 : 0.0);
    break;
  case TL_OP_RELU: /* reluSpec: if x == 0 then 0 else max x 0 */
    for (size_t i = i0; i < i1; ++i) {
      double x = a[i];
      out[i] = x == 0.0 ? 0.0 : (x <= 0.0 ? 0.0 : x);
    }
    break;
  case TL_OP_RELU_DERIV: /* reluDerivSpec: if x > 0 then 1 else 0 */
    for (size_t i = i0; i < i1; ++i) out[i] = a[i] > 0.0 ? 1.0 : 0.0;
    break;
  default:
    break;
  }
}

static lean_obj_res tl_elementwise(int op, b_lean_obj_arg a, b_lean_obj_arg b,
                                   double c) {
  tl_config();
  size_t n = lean_sarray_size(a);
  if (b != NULL && lean_sarray_size(b) < n) {
    n = lean_sarray_size(b);
  }
  lean_object *out = tl_alloc_floats(n);
  tl_note_native(op);
  tl_ew_job_t job = {op, lean_float_array_cptr(a),
                     b ? lean_float_array_cptr(b) : NULL,
                     lean_float_array_cptr(out), n, c};
  tl_parallel(tl_chunks(n), tl_ew_task, &job);
  return out;
}

/* Lean: `TorchLean.FastCpu.zipNative (op : UInt32) (a b : @& FloatArray)`. */
LEAN_EXPORT lean_obj_res torchlean_fastcpu_zip(uint32_t op, b_lean_obj_arg a,
                                               b_lean_obj_arg b) {
  if (op != TL_OP_ADD && op != TL_OP_SUB && op != TL_OP_MUL) {
    lean_internal_panic("torchlean fastcpu: unknown binary op");
  }
  return tl_elementwise((int)op, a, b, 0.0);
}

/* Lean: `TorchLean.FastCpu.mapNative (op : UInt32) (a : @& FloatArray)`. */
LEAN_EXPORT lean_obj_res torchlean_fastcpu_map(uint32_t op, b_lean_obj_arg a) {
  if (op < TL_OP_EXP || op > TL_OP_RELU_DERIV) {
    lean_internal_panic("torchlean fastcpu: unknown unary op");
  }
  return tl_elementwise((int)op, a, NULL, 0.0);
}

/* Lean: `TorchLean.FastCpu.scaleNative (a : @& FloatArray) (c : @& α)`. */
LEAN_EXPORT lean_obj_res torchlean_fastcpu_scale(b_lean_obj_arg a,
                                                 b_lean_obj_arg c) {
  return tl_elementwise(TL_OP_SCALE, a, NULL, tl_float_of_boxed(c));
}

/* ---------------------------------------------------------------------- */
/* Shapes: broadcast, its adjoint, adjacent-axis swap                      */
/* ---------------------------------------------------------------------- */

#define TL_MAX_RANK 16

/* Read a `List Nat` shape; returns its rank, or -1 if unsupported. */
static int tl_read_shape(b_lean_obj_arg list, size_t *dims) {
  int rank = 0;
  while (!lean_is_scalar(list)) {
    lean_object *head = lean_ctor_get(list, 0);
    if (rank >= TL_MAX_RANK || !lean_is_scalar(head)) {
      return -1;
    }
    dims[rank++] = lean_unbox(head);
    list = lean_ctor_get(list, 1);
  }
  return rank;
}

static size_t tl_shape_size(const size_t *dims, int rank) {
  size_t size = 1;
  for (int i = 0; i < rank; ++i) {
    size = tl_mul_checked(size, dims[i]);
  }
  return size;
}

/*
 * Lean: `TorchLean.FastCpu.broadcastNative (x : @& FloatArray)
 * (source target : @& List Nat)`. `broadcastTo` pads the source with leading
 * singleton axes and replicates singleton axes; this is a pure copy.
 * Returns an empty array when the shapes are not a valid broadcast.
 */
LEAN_EXPORT lean_obj_res torchlean_fastcpu_broadcast(b_lean_obj_arg x,
                                                     b_lean_obj_arg source,
                                                     b_lean_obj_arg target) {
  tl_config();
  size_t src[TL_MAX_RANK], dst[TL_MAX_RANK], stride[TL_MAX_RANK];
  int sr = tl_read_shape(source, src), dr = tl_read_shape(target, dst);
  if (sr < 0 || dr < 0 || sr > dr ||
      tl_shape_size(src, sr) != lean_sarray_size(x)) {
    return tl_alloc_floats(0);
  }
  /* Right-aligned source strides; 0 on padded or replicated axes. */
  size_t running = 1;
  for (int i = dr - 1; i >= 0; --i) {
    int si = i - (dr - sr);
    if (si < 0) {
      stride[i] = 0;
    } else if (src[si] == dst[i]) {
      stride[i] = running;
      running *= src[si];
    } else if (src[si] == 1) {
      stride[i] = 0;
    } else {
      return tl_alloc_floats(0);
    }
  }
  size_t n = tl_shape_size(dst, dr);
  lean_object *out = tl_alloc_floats(n);
  tl_note_native(TL_OP_BROADCAST);
  const double *in = lean_float_array_cptr(x);
  double *o = lean_float_array_cptr(out);
  size_t inner = dr > 0 ? dst[dr - 1] : 1;
  size_t inner_stride = dr > 0 ? stride[dr - 1] : 0;
  size_t idx[TL_MAX_RANK] = {0};
  for (size_t base = 0; base < n; base += inner) {
    size_t offset = 0;
    for (int i = 0; i < dr - 1; ++i) {
      offset += idx[i] * stride[i];
    }
    for (size_t j = 0; j < inner; ++j) {
      o[base + j] = in[offset + j * inner_stride];
    }
    for (int i = dr - 2; i >= 0; --i) {
      if (++idx[i] < dst[i]) {
        break;
      }
      idx[i] = 0;
    }
  }
  return out;
}

/*
 * Lean: `TorchLean.FastCpu.reduceLeadingNative (x : @& FloatArray)
 * (lead : @& List Nat) (inner : USize)`. `reduceFromBroadcastTo` when the
 * target is `lead ++ source`: sum the outermost axis away, one axis at a
 * time, each output in increasing index order from +0.0.
 */
typedef struct {
  const double *in;
  double *out;
  size_t outer, inner;
} tl_reduce_job_t;

static void tl_reduce_task(void *ctx, size_t task) {
  const tl_reduce_job_t *job = ctx;
  size_t j0 = task * TL_CHUNK;
  size_t j1 = j0 + TL_CHUNK < job->inner ? j0 + TL_CHUNK : job->inner;
  for (size_t j = j0; j < j1; ++j) {
    job->out[j] = 0.0;
  }
  for (size_t i = 0; i < job->outer; ++i) {
    const double *row = job->in + i * job->inner;
    for (size_t j = j0; j < j1; ++j) {
      job->out[j] = job->out[j] + row[j];
    }
  }
}

LEAN_EXPORT lean_obj_res torchlean_fastcpu_reduce_leading(b_lean_obj_arg x,
                                                          b_lean_obj_arg lead,
                                                          size_t inner) {
  tl_config();
  size_t dims[TL_MAX_RANK];
  int rank = tl_read_shape(lead, dims);
  if (rank < 0 ||
      tl_mul_checked(tl_shape_size(dims, rank), inner) !=
          lean_sarray_size(x)) {
    lean_internal_panic("torchlean fastcpu: reduce shape mismatch");
  }
  tl_note_native(TL_OP_REDUCE_BROADCAST);
  size_t rest = tl_shape_size(dims, rank) * inner;
  double *current = malloc((rest ? rest : 1) * sizeof(double));
  if (current == NULL) {
    lean_internal_panic_out_of_memory();
  }
  memcpy(current, lean_float_array_cptr(x), rest * sizeof(double));
  for (int axis = 0; axis < rank; ++axis) {
    size_t outer = dims[axis];
    size_t next = outer == 0 ? 0 : rest / outer;
    if (outer == 0) {
      /* An empty leading axis sums to zeros of the remaining shape. */
      next = tl_shape_size(dims + axis + 1, rank - axis - 1) * inner;
    }
    double *reduced = malloc((next ? next : 1) * sizeof(double));
    if (reduced == NULL) {
      lean_internal_panic_out_of_memory();
    }
    tl_reduce_job_t job = {current, reduced, outer, next};
    tl_parallel(tl_chunks(next), tl_reduce_task, &job);
    free(current);
    current = reduced;
    rest = next;
  }
  lean_object *out = tl_alloc_floats(rest);
  memcpy(lean_float_array_cptr(out), current, rest * sizeof(double));
  free(current);
  return out;
}

/*
 * Lean: `TorchLean.FastCpu.swapNative (x : @& FloatArray)
 * (outer a b inner : USize)`: [outer, a, b, inner] -> [outer, b, a, inner].
 */
typedef struct {
  const double *in;
  double *out;
  size_t outer, a, b, inner;
} tl_swap_job_t;

static void tl_swap_task(void *ctx, size_t task) {
  const tl_swap_job_t *job = ctx;
  size_t o = task / job->b, j = task % job->b;
  const double *in = job->in + o * job->a * job->b * job->inner;
  double *out = job->out + (o * job->b + j) * job->a * job->inner;
  for (size_t i = 0; i < job->a; ++i) {
    memcpy(out + i * job->inner, in + (i * job->b + j) * job->inner,
           job->inner * sizeof(double));
  }
}

LEAN_EXPORT lean_obj_res torchlean_fastcpu_swap(b_lean_obj_arg x,
                                                size_t outer, size_t a,
                                                size_t b, size_t inner) {
  tl_config();
  size_t n = tl_mul_checked(tl_mul_checked(tl_mul_checked(outer, a), b), inner);
  if (n != lean_sarray_size(x)) {
    lean_internal_panic("torchlean fastcpu: swap shape mismatch");
  }
  lean_object *out = tl_alloc_floats(n);
  tl_note_native(TL_OP_SWAP);
  tl_swap_job_t job = {lean_float_array_cptr(x), lean_float_array_cptr(out),
                       outer, a, b, inner};
  tl_parallel(outer * b, tl_swap_task, &job);
  return out;
}

/* ---------------------------------------------------------------------- */
/* Last-axis softmax                                                      */
/* ---------------------------------------------------------------------- */

typedef struct {
  const double *x, *dy;
  double *out;
  size_t rows, n;
  int backward;
} tl_softmax_job_t;

/*
 * softmaxVecSpec: m = foldl max x[0] x (max a v = if a <= v then v else a);
 * e = exp(x - m); d = foldl (+) 0 e; y = e / d.
 * softmaxInnermostBackwardSpec: s = foldl (+) 0 (dy * y); dx = y * (dy - s).
 */
static void tl_softmax_row(const tl_softmax_job_t *job, size_t r) {
  const double *x = job->x + r * job->n;
  double *y = job->out + r * job->n;
  size_t n = job->n;
  double m = x[0];
  for (size_t i = 0; i < n; ++i) {
    m = m <= x[i] ? x[i] : m;
  }
  double denom = 0.0;
  for (size_t i = 0; i < n; ++i) {
    y[i] = tl_exp(x[i] - m);
    denom = denom + y[i];
  }
  for (size_t i = 0; i < n; ++i) {
    y[i] = y[i] / denom;
  }
  if (job->backward) {
    const double *dy = job->dy + r * job->n;
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
      s = s + dy[i] * y[i];
    }
    for (size_t i = 0; i < n; ++i) {
      y[i] = y[i] * (dy[i] - s);
    }
  }
}

static void tl_softmax_task(void *ctx, size_t task) {
  const tl_softmax_job_t *job = ctx;
  size_t per = TL_CHUNK / (job->n ? job->n : 1) + 1;
  size_t r0 = task * per;
  size_t r1 = r0 + per < job->rows ? r0 + per : job->rows;
  for (size_t r = r0; r < r1; ++r) {
    tl_softmax_row(job, r);
  }
}

static lean_obj_res tl_softmax(int op, b_lean_obj_arg x, b_lean_obj_arg dy,
                               size_t rows, size_t n) {
  tl_config();
  size_t total = tl_mul_checked(rows, n);
  if (n == 0 || lean_sarray_size(x) != total ||
      (dy != NULL && lean_sarray_size(dy) != total)) {
    lean_internal_panic("torchlean fastcpu: softmax shape mismatch");
  }
  lean_object *out = tl_alloc_floats(total);
  tl_note_native(op);
  tl_softmax_job_t job = {lean_float_array_cptr(x),
                          dy ? lean_float_array_cptr(dy) : NULL,
                          lean_float_array_cptr(out), rows, n,
                          dy != NULL};
  size_t per = TL_CHUNK / n + 1;
  tl_parallel((rows + per - 1) / per, tl_softmax_task, &job);
  return out;
}

LEAN_EXPORT lean_obj_res torchlean_fastcpu_softmax(b_lean_obj_arg x,
                                                   size_t rows, size_t n) {
  return tl_softmax(TL_OP_SOFTMAX, x, NULL, rows, n);
}

LEAN_EXPORT lean_obj_res torchlean_fastcpu_softmax_backward(
    b_lean_obj_arg x, b_lean_obj_arg dy, size_t rows, size_t n) {
  return tl_softmax(TL_OP_SOFTMAX_BACKWARD, x, dy, rows, n);
}

/* ---------------------------------------------------------------------- */
/* Layer normalization over the last axis of [rows, width]                */
/* ---------------------------------------------------------------------- */

/*
 * Spec.layerNorm, per row, with E = (width : Float):
 *   mean = (foldl (+) 0 x) / E;            c = x - mean
 *   mean2 = (foldl (+) 0 c) / E;           q = (c - mean2) * (c - mean2)
 *   var = (foldl (+) 0 q) / E;             var = if var <= 0 then 0 else var
 *   std = sqrt(var + eps);                 xhat = c / std
 *   y = xhat * gamma + beta
 * (reduceVar centres its already-centred input again; this is reproduced.)
 */
typedef struct {
  const double *x, *gamma, *beta, *dy;
  double *out, *norm, *inv_std;
  size_t rows, width;
  double eps;
  int backward;
} tl_ln_job_t;

static double tl_ln_stats(const double *x, double *c, size_t width,
                          double eps) {
  double e = (double)width;
  double sum = 0.0;
  for (size_t j = 0; j < width; ++j) sum = sum + x[j];
  double mean = sum / e;
  for (size_t j = 0; j < width; ++j) c[j] = x[j] - mean;
  double sum2 = 0.0;
  for (size_t j = 0; j < width; ++j) sum2 = sum2 + c[j];
  double mean2 = sum2 / e;
  double sq = 0.0;
  for (size_t j = 0; j < width; ++j) {
    double d = c[j] - mean2;
    sq = sq + d * d;
  }
  double var = sq / e;
  var = var <= 0.0 ? 0.0 : var;
  return sqrt(var + eps);
}

static void tl_ln_task(void *ctx, size_t task) {
  const tl_ln_job_t *job = ctx;
  size_t per = TL_CHUNK / job->width + 1;
  size_t r0 = task * per;
  size_t r1 = r0 + per < job->rows ? r0 + per : job->rows;
  size_t width = job->width;
  double e = (double)width;
  for (size_t r = r0; r < r1; ++r) {
    const double *x = job->x + r * width;
    if (!job->backward) {
      double *y = job->out + r * width;
      double std = tl_ln_stats(x, y, width, job->eps);
      for (size_t j = 0; j < width; ++j) {
        y[j] = (y[j] / std) * job->gamma[j] + job->beta[j];
      }
      continue;
    }
    /*
     * Spec.layerNormBackward input gradient, per row:
     *   invStd = 1 / std;  xhat = c / std;  g = dy * gamma
     *   mg = (foldl (+) 0 g) / E;  mgx = (foldl (+) 0 (g * xhat)) / E
     *   dx = invStd * ((g - mg) - xhat * mgx)
     */
    double *xhat = job->norm + r * width;
    double *dx = job->out + r * width;
    const double *dy = job->dy + r * width;
    double std = tl_ln_stats(x, xhat, width, job->eps);
    double inv_std = 1.0 / std;
    for (size_t j = 0; j < width; ++j) xhat[j] = xhat[j] / std;
    double sg = 0.0, sgx = 0.0;
    for (size_t j = 0; j < width; ++j) {
      double g = dy[j] * job->gamma[j];
      dx[j] = g;
      sg = sg + g;
    }
    for (size_t j = 0; j < width; ++j) sgx = sgx + dx[j] * xhat[j];
    double mg = sg / e, mgx = sgx / e;
    for (size_t j = 0; j < width; ++j) {
      dx[j] = inv_std * ((dx[j] - mg) - xhat[j] * mgx);
    }
  }
}

/* Column sums over rows in increasing row order (reduceSum 0). */
typedef struct {
  const double *a, *b; /* b may be NULL: sum a; else sum a * b */
  double *out;
  size_t rows, width;
} tl_colsum_job_t;

static void tl_colsum_task(void *ctx, size_t task) {
  const tl_colsum_job_t *job = ctx;
  size_t j0 = task * 8;
  size_t j1 = j0 + 8 < job->width ? j0 + 8 : job->width;
  double acc[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (size_t r = 0; r < job->rows; ++r) {
    const double *a = job->a + r * job->width;
    if (job->b == NULL) {
      for (size_t j = j0; j < j1; ++j) acc[j - j0] = acc[j - j0] + a[j];
    } else {
      const double *b = job->b + r * job->width;
      for (size_t j = j0; j < j1; ++j)
        acc[j - j0] = acc[j - j0] + a[j] * b[j];
    }
  }
  for (size_t j = j0; j < j1; ++j) job->out[j] = acc[j - j0];
}

LEAN_EXPORT lean_obj_res torchlean_fastcpu_layer_norm(
    b_lean_obj_arg x, b_lean_obj_arg gamma, b_lean_obj_arg beta, size_t rows,
    size_t width, b_lean_obj_arg eps) {
  tl_config();
  size_t total = tl_mul_checked(rows, width);
  if (width == 0 || lean_sarray_size(x) != total ||
      lean_sarray_size(gamma) != width || lean_sarray_size(beta) != width) {
    lean_internal_panic("torchlean fastcpu: layer norm shape mismatch");
  }
  lean_object *out = tl_alloc_floats(total);
  tl_note_native(TL_OP_LAYER_NORM);
  tl_ln_job_t job = {lean_float_array_cptr(x), lean_float_array_cptr(gamma),
                     lean_float_array_cptr(beta), NULL,
                     lean_float_array_cptr(out), NULL, NULL, rows, width,
                     tl_float_of_boxed(eps), 0};
  size_t per = TL_CHUNK / width + 1;
  tl_parallel((rows + per - 1) / per, tl_ln_task, &job);
  return out;
}

/* Returns (dx, (dgamma, dbeta)). */
LEAN_EXPORT lean_obj_res torchlean_fastcpu_layer_norm_backward(
    b_lean_obj_arg x, b_lean_obj_arg gamma, b_lean_obj_arg dy, size_t rows,
    size_t width, b_lean_obj_arg eps) {
  tl_config();
  size_t total = tl_mul_checked(rows, width);
  if (width == 0 || lean_sarray_size(x) != total ||
      lean_sarray_size(gamma) != width || lean_sarray_size(dy) != total) {
    lean_internal_panic("torchlean fastcpu: layer norm shape mismatch");
  }
  lean_object *dx = tl_alloc_floats(total);
  lean_object *dgamma = tl_alloc_floats(width);
  lean_object *dbeta = tl_alloc_floats(width);
  tl_note_native(TL_OP_LAYER_NORM_BACKWARD);
  double *xhat = malloc((total ? total : 1) * sizeof(double));
  if (xhat == NULL) {
    lean_internal_panic_out_of_memory();
  }
  tl_ln_job_t job = {lean_float_array_cptr(x), lean_float_array_cptr(gamma),
                     NULL, lean_float_array_cptr(dy),
                     lean_float_array_cptr(dx), xhat, NULL, rows, width,
                     tl_float_of_boxed(eps), 1};
  size_t per = TL_CHUNK / width + 1;
  tl_parallel((rows + per - 1) / per, tl_ln_task, &job);
  /* scaleGradient = reduceSum 0 (dy * xhat); biasGradient = reduceSum 0 dy */
  tl_colsum_job_t sg = {lean_float_array_cptr(dy), xhat,
                        lean_float_array_cptr(dgamma), rows, width};
  tl_parallel((width + 7) / 8, tl_colsum_task, &sg);
  tl_colsum_job_t sb = {lean_float_array_cptr(dy), NULL,
                        lean_float_array_cptr(dbeta), rows, width};
  tl_parallel((width + 7) / 8, tl_colsum_task, &sb);
  free(xhat);
  lean_object *pair = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(pair, 0, dgamma);
  lean_ctor_set(pair, 1, dbeta);
  lean_object *triple = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(triple, 0, dx);
  lean_ctor_set(triple, 1, pair);
  return triple;
}

/* Lean: `TorchLean.FastCpu.fillNative (n : USize) (value : @& α)`. */
LEAN_EXPORT lean_obj_res torchlean_fastcpu_fill(size_t n,
                                                b_lean_obj_arg value) {
  double v = tl_float_of_boxed(value);
  lean_object *out = tl_alloc_floats(n);
  double *o = lean_float_array_cptr(out);
  for (size_t i = 0; i < n; ++i) {
    o[i] = v;
  }
  return out;
}

/*
 * Generic code over `[Context α]` passes `DecidableRel (· > ·)` as a partial
 * application of TorchLean's low-priority instance `instDecidableRelGt_nN`
 * (`Context.decidableGT`) to the type and the context. Recognise exactly that
 * closure, with a context structurally equal to `context`. The symbol is weak
 * so this object still links without TorchLean's generated code.
 */
extern lean_object *lp_TorchLean_instDecidableRelGt__nN___boxed(
    lean_object *, lean_object *, lean_object *, lean_object *)
    __attribute__((weak));

/* Lean: `TorchLean.FastCpu.isGtOfContext (gt : @& α) (context : @& β)`. */
LEAN_EXPORT uint8_t torchlean_fastcpu_is_gt_of_context(b_lean_obj_arg gt,
                                                       b_lean_obj_arg context) {
  void *instance = (void *)lp_TorchLean_instDecidableRelGt__nN___boxed;
  if (instance == NULL || lean_is_scalar(gt) ||
      lean_ptr_tag(gt) != LeanClosure || lean_closure_fun(gt) != instance ||
      lean_closure_arity(gt) != 4 || lean_closure_num_fixed(gt) != 2) {
    return 0;
  }
  return (uint8_t)tl_same(lean_closure_arg_cptr(gt)[1], context, 8);
}
