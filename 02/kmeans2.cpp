#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

typedef vector<double> Point;
typedef vector<Point> Points;

void selectStartCentroids(Points& centroids, const Points& data, size_t K,
                          size_t data_size);

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

// Calculates new centroid position as mean of positions of 3 random centroids
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

vector<size_t> KMeans(const Points& data, size_t K) {
    size_t data_size = data.size();
    size_t dimensions = data[0].size();
    vector<size_t> clusters(data_size);

    // Initialize centroids randomly at data points
    Points centroids(K);
    selectStartCentroids(centroids, data, K, data_size);

    bool converged = false;
    while (!converged) {
        converged = true;

        size_t nearest_cluster;

#pragma omp parallel for private(nearest_cluster) reduction(& : converged)
        for (size_t i = 0; i < data_size; ++i) {
            nearest_cluster = FindNearestCentroid(centroids, data[i]);
            if (clusters[i] != nearest_cluster) {
                clusters[i] = nearest_cluster;
                converged = false;
            }
        }
        converged = converged;
        if (converged) {
            break;
        }
        vector<size_t> clusters_sizes(K);
        centroids.assign(K, Point(dimensions));

        size_t d;

        for (size_t i = 0; i < data_size; ++i) {
            for (d = 0; d < dimensions; ++d) {
                centroids[clusters[i]][d] += data[i][d];
            }
            ++clusters_sizes[clusters[i]];
        }

        for (size_t i = 0; i < K; ++i) { // not parallel
            if (clusters_sizes[i] != 0) {
                for (d = 0; d < dimensions; ++d) {
                    centroids[i][d] /= clusters_sizes[i];
                }
            }
        }

        for (size_t i = 0; i < K; ++i) { // not parallel
            if (clusters_sizes[i] == 0) {
                centroids[i] = GetRandomPosition(centroids);
            }
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

void selectStartCentroids(Points& centroids, const Points& data, size_t K,
                          size_t data_size) {
    // for (size_t i = 0; i < K; ++i) {
    //     centroids[i] = data[UniformRandom(data_size - 1)];
    // }

    vector<double> distances(data_size);
    vector<size_t> centroids_indexes(K);

    centroids_indexes[0] = UniformRandom(data_size - 1);
    centroids[0] = data[centroids_indexes[0]];
    double sum = 0;
    double distance = 0;
    for (size_t i = 0; i < data_size; ++i) {
        distance = Distance(data[i], data[centroids_indexes[0]]);
        distances[i] = distance;
        sum += distance;
    }

    for (size_t i = 1; i < K; ++i) {
        // choice
        double choice_value = rand() * sum;
        double cur_sum = 0;
        size_t j = 0;
        for (; j < data_size && cur_sum < choice_value; ++j) {
            cur_sum += distances[j];
        }
        --j;

        centroids_indexes[i] = j;
        centroids[i] = data[j];

        // recalc
        double new_sum = 0;
        for (size_t j = 0; j < data_size; ++j) {
            distance = Distance(data[j], data[centroids_indexes[i]]);
            if (distance < distances[j]) {
                distances[j] = centroids_indexes[i];
            }
            new_sum += distances[j];
                }
        sum = new_sum;
    }
}
