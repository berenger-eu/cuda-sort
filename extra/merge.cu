template <typename T>
__global__ void mergeKernel(const T* A, T* C, int run_size, int N) {
    // position globale de début de ce bloc
    int start = (2*blockIdx.x) * run_size;
    if (start >= N) return;  // rien à faire

    // taille de la première run (runA) et de la deuxième (runB) au bord
    int sizeA = min(run_size, N - start);
    int sizeB = min(run_size, max(0, N - start - sizeA));

    const T* runA = A + start;
    const T* runB = A + start + sizeA;
    T*       out  = C + start;

    int totalLen = sizeA + sizeB;  // longueur à fusionner

    // chaque thread fusionne une tranche [D_start..D_end) dans [0..totalLen)
    int t = threadIdx.x, Tn = blockDim.x;
    int D_start = (t * totalLen) / Tn;
    int D_end   = ((t+1) * totalLen) / Tn;

    // --- binary search pour trouver p_start, q_start ---
    int p_lo = max(0, D_start - sizeB);
    int p_hi = min(sizeA, D_start);
    while (p_lo < p_hi) {
        int p_mid = (p_lo + p_hi) >> 1;
        int q_mid = D_start - p_mid;
        if (p_mid < sizeA && q_mid > 0 && runA[p_mid] < runB[q_mid-1])
            p_lo = p_mid + 1;
        else
            p_hi = p_mid;
    }
    int p_start = p_lo;
    int q_start = D_start - p_lo;

    // --- même pour D_end ---
    p_lo = max(0, D_end - sizeB);
    p_hi = min(sizeA, D_end);
    while (p_lo < p_hi) {
        int p_mid = (p_lo + p_hi) >> 1;
        int q_mid = D_end - p_mid;
        if (p_mid < sizeA && q_mid > 0 && runA[p_mid] < runB[q_mid-1])
            p_lo = p_mid + 1;
        else
            p_hi = p_mid;
    }
    int p_end = p_lo;
    int q_end = D_end - p_lo;

    // --- fusion séquentielle de la tranche [D_start..D_end) ---
    int i = p_start, j = q_start;
    for (int k = D_start; k < D_end; ++k) {
        // si runB épuisée ou runA plus petite, prendre runA
        if (j >= sizeB || (i < sizeA && runA[i] <= runB[j]))
            out[k] = runA[i++];
        else
            out[k] = runB[j++];
    }
}

