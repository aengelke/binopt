
#include <binopt.h>

#include <getopt.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int64_t xdiff, ydiff;
    float factor;
} StencilPoint;

typedef struct {
    size_t points;
    const StencilPoint* p;
} Stencil;

typedef void(*StencilFunction)(const Stencil*, float* restrict, float* restrict, size_t);

inline __attribute__((always_inline))
static void stencil_generic(const Stencil* stencil_par, float* restrict in_mat,
                            float* restrict out_mat, size_t mat_size) {
    for (size_t y = 1; y < mat_size - 1; ++y) {
        for (size_t x = 1; x < mat_size - 1; ++x) {
            size_t index = y * mat_size + x;
            float res = 0;
            for(size_t i = 0; i < stencil_par->points; i++) {
                const StencilPoint* p = &stencil_par->p[i];
                res += p->factor * in_mat[index + p->ydiff*mat_size + p->xdiff];
            }
            out_mat[index] = res;
        }
    }
}

static const StencilPoint stencil_pts[4] = {{1, 0, 0.25}, {-1, 0, 0.25}, {0, 1, 0.25}, {0, -1, 0.25}};
static const Stencil stencil = {sizeof(stencil_pts)/sizeof(StencilPoint), stencil_pts};

static void stencil_special(const Stencil* stencil_par, float* restrict in_mat,
                            float* restrict out_mat, size_t mat_size) {
    (void) stencil_par;
    stencil_generic(&stencil, in_mat, out_mat, mat_size);
}

static void init_matrix(size_t mat_size, float** matrix_in, float** matrix_out) {
    float* b = malloc(sizeof(float) * mat_size * mat_size);
    for (size_t i = 0; i < mat_size; i++) {
        for (size_t j = 0; j < mat_size; j++) {
            size_t index = i * mat_size + j;
            if (i == 0) // First Row
                b[index] = 1.0 - (j * 1.0 / (mat_size - 1));
            else if (i == (mat_size - 1)) // Last Row
                b[index] = j * 1.0 / (mat_size - 1);
            else if (j == 0) // First Column
                b[index] = 1.0 - (i * 1.0 / (mat_size - 1));
            else if (j == (mat_size - 1)) // Last Column
                b[index] = i * 1.0 / (mat_size - 1);
            else
                b[index] = 0;
        }
    }
    *matrix_out = malloc(sizeof(float) * mat_size * mat_size);
    memcpy(*matrix_out, b, sizeof(float) * mat_size * mat_size);

    *matrix_in = b;
}

static void print_matrix(size_t mat_size, float* matrix) {
    printf("Matrix:\n");
    if (mat_size < 9 || ((mat_size - 9) & 7))
        return;
    size_t stride = ((mat_size - 9) / 8) + 1;
    for (size_t y = 0; y < 9; y++) {
        for (size_t x = 0; x < 9; x++) {
            printf("%7.4f", matrix[y * stride * mat_size + x * stride]);
        }
        printf("\n");
    }
}

static struct timespec time_get(void) {
    struct timespec ret;
    clock_gettime(CLOCK_MONOTONIC, &ret);
    return ret;
}

static double time_diff_secs(struct timespec* first, struct timespec* last) {
    time_t diff_secs = last->tv_sec - first->tv_sec;
    long diff_nsecs = last->tv_nsec - first->tv_nsec;
    return diff_secs + diff_nsecs * 1e-9;
}

int main(int argc, char** argv) {
    bool use_binopt = false;
    size_t run_count = 10000;
    size_t interlines = 20;

    int opt;
    while ((opt = getopt(argc, argv, "on:i:")) != -1) {
        switch (opt) {
        case 'o': use_binopt = true; break;
        case 'n': run_count = strtoul(optarg, NULL, 0); break;
        case 'i': interlines = strtoul(optarg, NULL, 0); break;
        default:
            fprintf(stderr, "usage: %s [-o] [-n run_count] [-i interlines]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }


    size_t mat_size = 8 * interlines + 9;
    float* matrix_a;
    float* matrix_b;
    init_matrix(mat_size, &matrix_a, &matrix_b);

    BinoptHandle boh;
    StencilFunction stencil_fn;

    struct timespec time_start = time_get();

    if (!use_binopt) {
        printf("Using compiler: " __VERSION__ "\n");
        stencil_fn = stencil_special;
    } else {
        printf("Using rewriter: %s\n", binopt_driver());

        boh = binopt_init();
        BinoptCfgRef bcfg = binopt_cfg_new(boh, (BinoptFunc) stencil_generic);
        binopt_cfg_type(bcfg, 4, BINOPT_TY_VOID, BINOPT_TY_PTR_NOALIAS, BINOPT_TY_PTR_NOALIAS,
                        BINOPT_TY_PTR_NOALIAS, BINOPT_TY_UINT64);
        binopt_cfg_set(bcfg, BINOPT_F_FASTMATH, true);
        binopt_cfg_set_parami(bcfg, 3, mat_size);
        binopt_cfg_set_paramp(bcfg, 0, &stencil, sizeof(stencil), BINOPT_MEM_CONST);
        binopt_cfg_mem(bcfg, stencil.p, sizeof(StencilPoint)*stencil.points, BINOPT_MEM_CONST);

        *((BinoptFunc*) &stencil_fn) = binopt_spec_create(bcfg);
    }

    struct timespec time_exec = time_get();

    for (size_t i = 0; i < run_count; i++) {
        stencil_fn(&stencil, matrix_a, matrix_b, mat_size);
        float* tmp = matrix_a;
        matrix_a = matrix_b;
        matrix_b = tmp;
    }

    struct timespec time_end = time_get();

    print_matrix(mat_size, matrix_a);

    printf("Time (preparation): %8.4lf\n", time_diff_secs(&time_start, &time_exec));
    printf("Time (execution):   %8.4lf\n", time_diff_secs(&time_exec, &time_end));
    printf("Time (total):       %8.4lf\n", time_diff_secs(&time_start, &time_end));

    if (use_binopt)
        binopt_fini(boh);

    return 0;
}

