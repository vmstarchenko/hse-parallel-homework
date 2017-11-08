gen:
	./datagen 2 1000000 30 data.in

compile_original:
	g++ kmeans.cpp -o kmeans_original

run_original:
	time ./kmeans_original 5 data.in data_original.out

compile:
	g++ kmeans.cpp -o kmeans -fopenmp

run:
	./kmeans 5 data.in data.out

check_diff:
	diff data_original.out data.out && echo "Ok" || echo "Diff error"

# make compile_original && time make run_original
# make compile && time make run && make check_diff
