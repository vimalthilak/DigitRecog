CC=g++-mp-4.8
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
	$(CC) $(LDFLAGS) python/DigitRecog_wrap.o -L/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/config -ldl -lpython2.7 -Llib/ -lDigitRecog -Xlinker -rpath -Xlinker `dirname \`pwd\`` -o python/_DigitRecog.so

python/DigitRecog_wrap.o: python/DigitRecog_wrap.cxx
	$(CC) $(CFLAGS) python/DigitRecog_wrap.cxx -I../Services `python2.7-config --cflags` -o python/DigitRecog_wrap.o

python/DigitRecog_wrap.cxx:
	swig -Wall -c++ -python python/DigitRecog.i

lib/libDigitRecog.so: $(OBJ_FILES)
	$(CC) $(LDFLAGS) $(LDFLAGSEXTRA)libDigitRecog.so $^ -L../Services/lib -lServices -L../UtilityToolsInterfaces/lib -lObjectHolder -L/opt/local/lib/ -lopencv_ml -lopencv_core `root-config --libs`  -Xlinker -rpath -Xlinker `dirname \`pwd\`` -o $@

build/%.o: src/%.cxx
	$(CC) $(CFLAGS) -I../Services -I../UtilityToolsInterfaces `root-config --cflags` $< -o $@

clean:
	rm -r build/
	rm -r lib/
	rm python/DigitRecog_wrap*
	rm python/DigitRecog.py*
	rm python/_DigitRecog.so
