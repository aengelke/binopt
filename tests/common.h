
#ifndef BINOPT_TEST_COMMON_H
#define BINOPT_TEST_COMMON_H

#include <binopt.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int optimize;

static BinoptHandle test_init(int argc, char** argv) {
    if (argc != 3) {
        printf("usage: %s driver opt\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    optimize = strtol(argv[2], NULL, 0);

    const char* driver = binopt_driver();
    if (strcmp(argv[1], driver)) {
        printf("error: expected driver %s got %s\n", argv[1], driver);
        exit(EXIT_FAILURE);
    }

    return binopt_init();
}

static void test_eq_i32(int32_t a, int32_t no_opt, int32_t opt) {
    int32_t b = optimize ? opt : no_opt;
    if (a != b) {
        printf("error: got %d != %d\n", a, b);
        exit(EXIT_FAILURE);
    }
}

__attribute__((noreturn))
static void test_fini(void) {
    exit(EXIT_SUCCESS);
}

#endif
