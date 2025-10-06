output:main.o
	g++ -g -o output main.o
main.o: main.cpp i2c.h
	g++ -c -g main.cpp -libi2c

clean:main.o output
	rm output main.o