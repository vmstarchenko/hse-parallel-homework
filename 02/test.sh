#!/bin/bash

K=30
d=2
N=800000

# ./datagen $d $N $K data.in


# g++ kmeans_original.cpp -o kmeans_original -O2
# echo -ne "Base:"
# time ./kmeans_original $K data.in data_original.out
# echo ""


# g++ -std=c++11 -fopenmp kmeans.cpp -o kmeans -O2 -fsanitize=address|| exit 1
g++ -std=c++11 -fopenmp kmeans.cpp -o kmeans -O2 || exit 1
echo -ne "Parallel:"
time ./kmeans $K data.in data.out
diff -q data_original.out data.out
