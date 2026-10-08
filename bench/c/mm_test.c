/*
 * Standalone check of the fastcpu matmul kernels: bit-exactness against the
 * naive sequential reference, and throughput. Builds without the Lean
 * runtime (only the raw-buffer job functions are exercised):
 *
 *   cc -O3 -ffp-contract=off -pthread -I$(lean --print-prefix)/include \
 *     bench/c/mm_test.c -o mm_test -Wl,--unresolved-symbols=ignore-all
 */
#include "../../csrc/cpu/torchlean_fastcpu.c"

#include <math.h>
#include <time.h>

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint64_t rng = 88172645463325252ull;
static double rnd(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  /* Mix magnitudes and signs, including exact zeros and negative zeros. */
  uint64_t r = rng % 1000;
  if (r == 0) return 0.0;
  if (r == 1) return -0.0;
  double x = (double)(rng >> 11) / 9007199254740992.0 * 2.0 - 1.0;
  return x * ldexp(1.0, (int)(rng % 21) - 10);
}

/* Reference: out[b,i,j] = foldl (+) 0 [L(b,i,k) * R(b,k,j) | k <- 0..K-1]. */
static void reference(const double *l, size_t l_bs, size_t l_rs, size_t l_cs,
                      const double *r, size_t r_bs, size_t r_rs, size_t r_cs,
                      double *out, size_t batch, size_t m, size_t k,
                      size_t n) {
  for (size_t b = 0; b < batch; ++b)
    for (size_t i = 0; i < m; ++i)
      for (size_t j = 0; j < n; ++j) {
        volatile double acc = 0.0;
        for (size_t kk = 0; kk < k; ++kk) {
          double prod = l[b * l_bs + i * l_rs + kk * l_cs] *
                        r[b * r_bs + kk * r_rs + j * r_cs];
          acc = acc + prod;
        }
        out[(b * m + i) * n + j] = acc;
      }
}

static void run(const double *l, size_t l_bs, size_t l_rs, size_t l_cs,
                const double *r, size_t r_bs, size_t r_rs, size_t r_cs,
                double *out, size_t batch, size_t m, size_t k, size_t n) {
  double *packed = NULL;
  if (r_cs != 1) {
    size_t pb = r_bs == 0 ? 1 : batch;
    packed = malloc(pb * k * n * sizeof(double));
    tl_pack_job_t pack = {r, r_bs, r_rs, r_cs, packed, k, n};
    tl_parallel(pb, tl_pack_task, &pack);
    r = packed;
    r_bs = r_bs == 0 ? 0 : k * n;
    r_rs = n;
  }
  tl_mm_job_t job = {.l = l, .l_bs = l_bs, .l_rs = l_rs, .l_cs = l_cs,
                     .r = r, .r_bs = r_bs, .r_rs = r_rs, .out = out,
                     .m = m, .k = k, .n = n};
  tl_choose_blocks(&job, batch);
  tl_parallel(batch * job.row_blocks * job.col_blocks, tl_mm_task, &job);
  free(packed);
}

typedef struct {
  const char *name;
  size_t batch, m, k, n;
  int mode; /* 0: A@B, 1: dC@B^T (dA), 2: A^T@dC (dB), 3: shared B */
} case_t;

static int check_case(const case_t *c, int timing) {
  size_t batch = c->batch, m = c->m, k = c->k, n = c->n;
  size_t l_bs, l_rs, l_cs, r_bs, r_rs, r_cs, l_size, r_size;
  switch (c->mode) {
  case 0: /* L = A[b] (m x k), R = B[b] (k x n) */
    l_bs = m * k; l_rs = k; l_cs = 1; r_bs = k * n; r_rs = n; r_cs = 1;
    l_size = batch * m * k; r_size = batch * k * n; break;
  case 1: /* L = dC[b] (m x k), R(kk, j) = B[b][j][kk], B is (n x k) */
    l_bs = m * k; l_rs = k; l_cs = 1; r_bs = n * k; r_rs = 1; r_cs = k;
    l_size = batch * m * k; r_size = batch * n * k; break;
  case 2: /* L(i, kk) = A[b][kk][i], A is (k x m); R = dC[b] (k x n) */
    l_bs = k * m; l_rs = 1; l_cs = m; r_bs = k * n; r_rs = n; r_cs = 1;
    l_size = batch * k * m; r_size = batch * k * n; break;
  default: /* shared R */
    l_bs = m * k; l_rs = k; l_cs = 1; r_bs = 0; r_rs = n; r_cs = 1;
    l_size = batch * m * k; r_size = k * n; break;
  }
  double *l = malloc(l_size * sizeof(double));
  double *r = malloc(r_size * sizeof(double));
  double *out = malloc(batch * m * n * sizeof(double));
  double *ref = malloc(batch * m * n * sizeof(double));
  for (size_t i = 0; i < l_size; ++i) l[i] = rnd();
  for (size_t i = 0; i < r_size; ++i) r[i] = rnd();
  run(l, l_bs, l_rs, l_cs, r, r_bs, r_rs, r_cs, out, batch, m, k, n);
  int ok = 1;
  if (!timing) {
    reference(l, l_bs, l_rs, l_cs, r, r_bs, r_rs, r_cs, ref, batch, m, k, n);
    if (memcmp(out, ref, batch * m * n * sizeof(double)) != 0) {
      ok = 0;
      for (size_t i = 0; i < batch * m * n; ++i)
        if (memcmp(&out[i], &ref[i], sizeof(double)) != 0) {
          printf("  mismatch at %zu: %a vs %a\n", i, out[i], ref[i]);
          break;
        }
    }
    printf("%-28s %s\n", c->name, ok ? "bit-exact" : "MISMATCH");
  } else {
    int reps = 5;
    double best = 1e30;
    for (int rep = 0; rep < reps; ++rep) {
      double t0 = now();
      run(l, l_bs, l_rs, l_cs, r, r_bs, r_rs, r_cs, out, batch, m, k, n);
      double t = now() - t0;
      if (t < best) best = t;
    }
    double flops = 2.0 * batch * m * k * n;
    printf("%-28s %8.3f ms  %7.1f GFLOP/s\n", c->name, best * 1e3,
           flops / best / 1e9);
  }
  free(l); free(r); free(out); free(ref);
  return ok;
}

int main(int argc, char **argv) {
  (void)argv;
  tl_config();
  printf("threads=%zu avx2=%d\n", tl_thread_count, tl_cpu_has_avx2());
  case_t small[] = {
      {"fwd 1x7x5x3", 1, 7, 5, 3, 0},     {"fwd 3x13x9x21", 3, 13, 9, 21, 0},
      {"fwd 1x150x24x150", 1, 150, 24, 150, 0},
      {"dA 2x37x19x29", 2, 37, 19, 29, 1}, {"dB 2x37x19x29", 2, 37, 19, 29, 2},
      {"shared 4x33x17x12", 4, 33, 17, 12, 3},
      {"fwd 1x1x1x1", 1, 1, 1, 1, 0},       {"fwd 1x5x0x7 (k=0)", 1, 5, 0, 7, 0},
      {"fwd 1x300x96x96", 1, 300, 96, 96, 0},
      {"dB 1x96x2000x96", 1, 96, 2000, 96, 2},
      {"dA 5x150x150x24", 5, 150, 150, 24, 1},
  };
  int ok = 1;
  for (size_t i = 0; i < sizeof small / sizeof small[0]; ++i)
    ok &= check_case(&small[i], 0);
  if (argc > 1) {
    case_t big[] = {
        {"fwd [28800x96]@[96x96]", 1, 28800, 96, 96, 0},
        {"dA [28800x96]@[96x96]^T", 1, 28800, 96, 96, 1},
        {"dB [96x28800]@[28800x96]", 1, 96, 28800, 96, 2},
        {"fwd [28800x96]@[96x192]", 1, 28800, 96, 192, 0},
        {"fwd [28800x192]@[192x96]", 1, 28800, 192, 96, 0},
        {"bmm 768x[150x24]@[24x150]", 768, 150, 24, 150, 0},
        {"bmm 768x[150x150]@[150x24]", 768, 150, 150, 24, 0},
        {"bmm dA 768x[150x150]@^T", 768, 150, 150, 24, 1},
        {"bmm dB 768x^T[24x150]", 768, 24, 150, 150, 2},
    };
    for (size_t i = 0; i < sizeof big / sizeof big[0]; ++i)
      check_case(&big[i], 1);
  }
  return ok ? 0 : 1;
}
