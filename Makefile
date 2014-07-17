.PHONY: all test check clean cleanall

all:
	cd src; make

test:
	cd src; make test

check:
	cd src; make check

clean:
	rm -f src/*/*.o src/*.o
	rm -rf exe/*.dSYM
