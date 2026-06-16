#include <iostream>
#include <vector>

int main() {
    const int N = 40;          // grid size (40 x 40)
    const int STEPS = 2000;    // number of time steps
    const double ALPHA = 0.2;  // how fast heat spreads (keep below 0.25)

    // Two grids. Read from 'curr', write into 'next', then swap them.
    std::vector<std::vector<double>> curr(N, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> next = curr;

    // Heat source: the whole top row is hot (100 degrees).
    for (int j = 0; j < N; ++j) curr[0][j] = 100.0;

    // Run the simulation.
    for (int step = 0; step < STEPS; ++step) {
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                // Each cell moves toward the average of its 4 neighbors.
                next[i][j] = curr[i][j] + ALPHA * (
                    curr[i-1][j] + curr[i+1][j] +
                    curr[i][j-1] + curr[i][j+1] - 4 * curr[i][j]);
            }
        }
        for (int j = 0; j < N; ++j) next[0][j] = 100.0;  // keep top row hot
        curr.swap(next);
    }

    // Show the result as a text heatmap (space = cold, @ = hot).
    const char* shades = " .:-=+*#%@";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int level = (int)(curr[i][j] / 100.0 * 9);
            if (level < 0) level = 0;
            if (level > 9) level = 9;
            std::cout << shades[level];
        }
        std::cout << "\n";
    }
    return 0;
}
