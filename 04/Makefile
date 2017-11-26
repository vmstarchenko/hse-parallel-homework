N=10000
S=4
I=5

cdatagen:
	mpicc datagen.c -O2 -o tmp/datagen.a

datagen: cdatagen
	./tmp/datagen.a $N tmp/data$N.in

clife:
	mpicc  life.c -o tmp/tmp.a -lm

life: clife
	time mpirun -np $S ./tmp/tmp.a $N tmp/data$N.in $I tmp/data$N.res

corigin_life:
	gcc origin_life.c -O2 -o tmp/origin_life.a

origin_life: corigin_life
	time ./tmp/origin_life.a $N tmp/data$N.in $I tmp/data$N.out

utest: datagen life origin_life
	diff tmp/data$N.out tmp/data$N.res -q
