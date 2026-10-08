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

enum {
  TL_OP_MATMUL = 0,
  TL_OP_MATMUL_BACKWARD,
  TL_OP_COUNT_MAX = 64
};

static const char *tl_op_names[TL_OP_COUNT_MAX] = {
    [TL_OP_MATMUL] = "matmul",
    [TL_OP_MATMUL_BACKWARD] = "matmul_backward",
};

static int tl_enabled_flag = 1;
static int tl_stats_flag = 0;
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

static void tl_mm_scalar_tile(const tl_mm_job_t *job, const double *l,
                              const double *r, double *out, size_t i0,
                              size_t i1, size_t j0, size_t j1) {
  for (size_t i = i0; i < i1; ++i) {
    const double *li = l + i * job->l_rs;
    for (size_t j = j0; j < j1; ++j) {
      double acc = 0.0;
      for (size_t k = 0; k < job->k; ++k) {
        acc = acc + li[k * job->l_cs] * r[k * job->r_rs + j];
      }
      out[i * job->n + j] = acc;
    }
  }
}

#if TL_HAVE_X86
#define TL_AVX2 __attribute__((target("avx2")))

/* 4 rows x 8 columns: eight 4-lane accumulators. */
TL_AVX2 static void tl_mm_avx2_4x8(const double *l, size_t l_rs, size_t l_cs,
                                    const double *r, size_t r_rs, double *out,
                                    size_t ldo, size_t kk) {
  __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
  __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
  __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
  __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
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
                                    size_t ldo, size_t kk) {
  __m256d c0 = _mm256_setzero_pd(), c1 = _mm256_setzero_pd();
  __m256d c2 = _mm256_setzero_pd(), c3 = _mm256_setzero_pd();
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
                                    size_t kk) {
  __m256d c0 = _mm256_setzero_pd(), c1 = _mm256_setzero_pd();
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
                                    size_t kk) {
  __m256d c0 = _mm256_setzero_pd();
  for (size_t k = 0; k < kk; ++k) {
    __m256d a = _mm256_broadcast_sd(l + k * l_cs);
    c0 = _mm256_add_pd(c0, _mm256_mul_pd(a, _mm256_loadu_pd(r + k * r_rs)));
  }
  _mm256_storeu_pd(out, c0);
}

TL_AVX2 static void tl_mm_avx2_tile(const tl_mm_job_t *job, const double *l,
                                     const double *r, double *out, size_t i0,
                                     size_t i1, size_t j0, size_t j1) {
  size_t kk = job->k, ldo = job->n;
  size_t i = i0;
  for (; i + 4 <= i1; i += 4) {
    const double *li = l + i * job->l_rs;
    size_t j = j0;
    for (; j + 8 <= j1; j += 8) {
      tl_mm_avx2_4x8(li, job->l_rs, job->l_cs, r + j, job->r_rs,
                     out + i * ldo + j, ldo, kk);
    }
    for (; j + 4 <= j1; j += 4) {
      tl_mm_avx2_4x4(li, job->l_rs, job->l_cs, r + j, job->r_rs,
                     out + i * ldo + j, ldo, kk);
    }
    if (j < j1) {
      tl_mm_scalar_tile(job, l, r, out, i, i + 4, j, j1);
    }
  }
  for (; i < i1; ++i) {
    const double *li = l + i * job->l_rs;
    size_t j = j0;
    for (; j + 8 <= j1; j += 8) {
      tl_mm_avx2_1x8(li, job->l_cs, r + j, job->r_rs, out + i * ldo + j, kk);
    }
    for (; j + 4 <= j1; j += 4) {
      tl_mm_avx2_1x4(li, job->l_cs, r + j, job->r_rs, out + i * ldo + j, kk);
    }
    if (j < j1) {
      tl_mm_scalar_tile(job, l, r, out, i, i + 1, j, j1);
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
#if TL_HAVE_X86
  if (tl_cpu_has_avx2()) {
    tl_mm_avx2_tile(job, l, r, out, i0, i1, j0, j1);
    return;
  }
#endif
  tl_mm_scalar_tile(job, l, r, out, i0, i1, j0, j1);
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
