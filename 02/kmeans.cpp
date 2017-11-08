#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

using namespace std;

typedef vector<double> Point;
typedef vector<Point> Points;

unsigned int UniformRandom(unsigned int max_value);
double Distance(const Point& point1, const Point& point2);
size_t FindNearestCentroid(const Points& centroids, const Point& point);
Point GetRandomPosition(const Points& centroids);

vector<size_t> KMeans(const Points& data, size_t K) {
    size_t ths_num = omp_get_max_threads();
    size_t data_size = data.size();
    size_t dimensions = data[0].size();

    vector<vector<size_t>> ths_clusters(
        ths_num, vector<size_t>(data_size / ths_num + 1, 0));

    vector<size_t> clusters(data_size);

    // Initialize centroids randomly at data points
    Points centroids(K);
    vector<Points> ths_centroids(ths_num, Points(K, Point(dimensions)));

    vector<vector<size_t>> ths_clusters_sizes(ths_num, vector<size_t>(K));
    vector<size_t> clusters_sizes(K);

    for (size_t i = 0; i < K; ++i) {
        centroids[i] = data[UniformRandom(data_size - 1)];
    }

    vector<int> ths_converged(ths_num);

    bool converged = false;

    size_t i, j, th_num, th_start_i, th_end_i, th_converged, nearest_cluster, d;

    while (!converged) {
        converged = true;

#pragma omp parallel shared(data) private(i, j, th_num, th_start_i, th_end_i,  \
                                          th_converged, nearest_cluster, d)
        {
            th_num = omp_get_thread_num();
            vector<size_t>& th_clusters = ths_clusters[th_num];
            vector<Point>& th_centroids = ths_centroids[th_num];
            vector<size_t>& th_clusters_sizes = ths_clusters_sizes[th_num];
            th_start_i = th_num * data_size / ths_num;
            th_end_i = (th_num + 1) * data_size / ths_num;
            th_converged = true;

            th_clusters.assign(K, 0);

            vector<double> distances(K);

            // #pragma omp for reduction(& : converged) private(j)
            for (size_t k = 1; k < K; ++k) {

                for (j = 0, i = th_start_i; i < th_end_i; ++i, ++j) {
                    // nearest_cluster = FindNearestCentroid(centroids,
                    // data[i]);

                    // const Points& centroids, const Point& point
                    const Point& point = data[i];

                    double min_distance = Distance(point, centroids[0]);

                    double distance = Distance(point, centroids[k]);
                    if (distance < min_distance) {
                        min_distance = distance;
                        nearest_cluster = k;
                    }
                    if (th_clusters[j] != nearest_cluster) {
                        th_clusters[j] = nearest_cluster;
                        // converged = false;
                        th_converged = false;
                    }
                }
            }

            ths_converged[th_num] = th_converged;

            th_clusters_sizes.assign(K, 0);
            th_centroids.assign(K, Point(dimensions));

            // #pragma omp for
            for (i = th_start_i, j = 0; i < th_end_i; ++i, ++j) {
                for (d = 0; d < dimensions; ++d) {
                    th_centroids[th_clusters[j]][d] += data[i][d];
                }
                ++th_clusters_sizes[th_clusters[j]];
            }
        }
        // pragma omp parallel end

        centroids.assign(K, Point(dimensions));
        clusters_sizes.assign(K, 0);

        // agregate
        converged = true;

        for (j = 0; j < ths_num; ++j) {
            for (i = 0; i < K; ++i) {
                for (d = 0; d < dimensions; ++d) {
                    centroids[i][d] += ths_centroids[j][i][d];
                }
                clusters_sizes[i] += ths_clusters_sizes[j][i];
            }
            converged &= ths_converged[j];
        }

        // original end

        for (i = 0; i < K; ++i) { // not parallel
            if (clusters_sizes[i] != 0) {
                for (d = 0; d < dimensions; ++d) {
                    centroids[i][d] /= clusters_sizes[i];
                }
            }
        }

        for (i = 0; i < K; ++i) { // not parallel
            if (clusters_sizes[i] == 0) {
                centroids[i] = GetRandomPosition(centroids);
            }
        }
    }

    for (th_num = 0; th_num < ths_num; ++th_num) {
        th_start_i = th_num * data_size / ths_num;
        th_end_i = (th_num + 1) * data_size / ths_num;
        for (i = th_start_i, j = 0; i < th_end_i; ++i, ++j) {
            clusters[i] = ths_clusters[th_num][j];
        }
    }

    return clusters;
}

// void ReadPoints(Points* data, ifstream& input) {
//     size_t data_size;
//     size_t dimensions;
//     input >> data_size >> dimensions;
//     data->assign(data_size, Point(dimensions));
//     for (size_t i = 0; i < data_size; ++i) {
//         for (size_t d = 0; d < dimensions; ++d) {
//             double coord;
//             input >> coord;
//             (*data)[i][d] = coord;
//         }
//     }
// }

// optimized version using atof()
void ReadPoints(Points* data, ifstream& input) {
    size_t data_size;
    size_t dimensions;
    input >> data_size >> dimensions;
    data->assign(data_size, Point(dimensions));
    string s;
    for (size_t i = 0; i < data_size; ++i) {
        for (size_t d = 0; d < dimensions; ++d) {
            input >> s;
            (*data)[i][d] = atof(s.c_str());
        }
    }
}

void WriteOutput(const vector<size_t>& clusters, ofstream& output) {
    for (size_t i = 0; i < clusters.size(); ++i) {
        output << clusters[i] << '\n';
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::printf("Usage: %s number_of_clusters input_file output_file\n",
                    argv[0]);
        return 1;
    }
    size_t K = atoi(argv[1]);

    char* input_file = argv[2];
    ifstream input;
    input.open(input_file, ifstream::in);
    if (!input) {
        cerr << "Error: input file could not be opened\n";
        return 1;
    }

    Points data;
    ReadPoints(&data, input);
    input.close();

    char* output_file = argv[3];
    ofstream output;
    output.open(output_file, ifstream::out);
    if (!output) {
        cerr << "Error: output file could not be opened\n";
        return 1;
    }

    srand(123); // for reproducible results

    vector<size_t> clusters = KMeans(data, K);

    WriteOutput(clusters, output);
    output.close();

    return 0;
}

Point GetRandomPosition(const Points& centroids) {
    size_t K = centroids.size();
    int c1 = rand() % K;
    int c2 = rand() % K;
    int c3 = rand() % K;
    size_t dimensions = centroids[0].size();
    Point new_position(dimensions);
    for (size_t d = 0; d < dimensions; ++d) {
        new_position[d] =
            (centroids[c1][d] + centroids[c2][d] + centroids[c3][d]) / 3;
    }
    return new_position;
}

// Gives random number in range [0..max_value]
unsigned int UniformRandom(unsigned int max_value) {
    unsigned int rnd = ((static_cast<unsigned int>(rand()) % 32768) << 17) |
                       ((static_cast<unsigned int>(rand()) % 32768) << 2) |
                       rand() % 4;
    return ((max_value + 1 == 0) ? rnd : rnd % (max_value + 1));
}

double Distance(const Point& point1, const Point& point2) {
    double distance_sqr = 0;
    size_t dimensions = point1.size();
    for (size_t i = 0; i < dimensions; ++i) {
        distance_sqr += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return distance_sqr;
}

size_t FindNearestCentroid(const Points& centroids, const Point& point) {
    double min_distance = Distance(point, centroids[0]);
    size_t centroid_index = 0;
    size_t centroids_size = centroids.size();
    for (size_t i = 1; i < centroids_size; ++i) {
        double distance = Distance(point, centroids[i]);
        if (distance < min_distance) {
            min_distance = distance;
            centroid_index = i;
        }
    }
    return centroid_index;
}
