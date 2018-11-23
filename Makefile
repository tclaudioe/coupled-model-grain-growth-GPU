clean:
	rm main.o result.err result.out debug.txt timeline.prof || true
	
compile: clean
	nvcc main.cu -o main.o

compile-local: clean
	optirun nvcc main.cu -o main.o -ccbin g++53

send: compile
	qsub submission.sh

debug: clean
	nvcc -g -G -D VDEBUG --compiler-options -Wall -lineinfo main.cu -o main.o
	qsub submission_memtest.sh

memtest: clean
	nvcc --compiler-options -Wall -lineinfo main.cu -o main.o
	qsub submission_memtest.sh

profile: clean
	nvcc main.cu -o main.o
	qsub submission_profiling.sh

timeline: clean
	nvcc main.cu -o main.o
	qsub submission_timeline.sh
