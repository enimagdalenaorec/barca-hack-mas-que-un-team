#include <stdlib.h>
/* decode.viterbi3 with a per-frame switch cost; identical comparisons and tie-breaks. */
void vit3(int n, const double *eh, const double *ea, const double *ed, const double *cost,
          double st, double rs, signed char *path) {
    unsigned char *back = (unsigned char *)malloc(3 * (size_t)n);
    double h = eh[0], a = ea[0], d = ed[0];
    for (int t = 1; t < n; t++) {
        double sw = cost[t];
        double nh = h; unsigned char bh = 0;
        if (a - sw > nh) { nh = a - sw; bh = 1; }
        if (d - rs > nh) { nh = d - rs; bh = 2; }
        double na = a; unsigned char ba = 1;
        if (h - sw > na) { na = h - sw; ba = 0; }
        if (d - rs > na) { na = d - rs; ba = 2; }
        double nd = d; unsigned char bd = 2;
        if (h - st > nd) { nd = h - st; bd = 0; }
        if (a - st > nd) { nd = a - st; bd = 1; }
        h = nh + eh[t]; a = na + ea[t]; d = nd + ed[t];
        back[3 * t] = bh; back[3 * t + 1] = ba; back[3 * t + 2] = bd;
    }
    int state = 0; double best = h;
    if (a > best) { best = a; state = 1; }
    if (d > best) { best = d; state = 2; }
    for (int t = n - 1; t >= 0; t--) { path[t] = (signed char)state; state = back[3 * t + state]; }
    free(back);
}
