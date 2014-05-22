CC=g++
CFLAGS=-std=c++11 -fPIC -Wall -I. -c
LDFLAGS=-std=c++11 -shared 
LDFLAGSEXTRA=-install_name @rpath/DigitRecog/lib/
OBJDIR=build
LIBDIR=lib

CXX_FILES := $(wildcard src/*.cxx)
OBJ_FILES := $(addprefix build/,$(notdir $(CXX_FILES:.cxx=.o)))

all: $(OBJDIR) $(LIBDIR) python/_DigitRecog.so

$(OBJDIR):
	mkdir $(OBJDIR)

$(LIBDIR):
	mkdir $(LIBDIR) 

python/_DigitRecog.so: lib/libDigitRecog.so python/DigitRecog_wrap.o
	$(CC) $(LDFLAGS) python/DigitRecog_wrap.o `python2.7-config --ldflags` -o python/_DigitRecog.so -Wl,-rpath,'$$ORIGIN/../lib' -Llib/ -lDigitRecog 

python/DigitRecog_wrap.o: python/DigitRecog_wrap.cxx
	$(CC) $(CFLAGS) python/DigitRecog_wrap.cxx -I../simplefwk-services `python2.7-config --cflags` -o python/DigitRecog_wrap.o

python/DigitRecog_wrap.cxx:
	swig -Wall -c++ -python python/DigitRecog.i

lib/libDigitRecog.so: $(OBJ_FILES)
	$(CC) $(LDFLAGS) $^ -L../simplefwk-services/lib -lServices -L../simplefwk-utilitytoolsinterfaces/lib -lObjectHolder -lopencv_ml -lopencv_core `root-config --libs`  -Wl,-rpath,'$$ORIGIN/../../simplefwk-services/lib' -Wl,-rpath,'$$ORIGIN/../../simplefwk-utilitytoolsinterfaces/lib' -o $@

build/%.o: src/%.cxx
	$(CC) $(CFLAGS) -I../simplefwk-services -I../simplefwk-utilitytoolsinterfaces `root-config --cflags` $< -o $@

clean:
	rm -r build/
	rm -r lib/
	rm python/DigitRecog_wrap*
	rm python/DigitRecog.py*
	rm python/_DigitRecog.so
