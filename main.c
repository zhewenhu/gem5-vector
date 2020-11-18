#include <stdio.h>
#include <gem5/m5ops.h>

int main() {
    printf("Hello, World!");
    m5_reset_stats(0,0);
    return 0;
}