gcc -x c 01.cu -fsanitize=leak,address && time ./a.out
