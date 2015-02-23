.SUFFIXES: .cpp

CXX = clang++  

SRCS = main.cpp XBuilder.cpp sfm.cpp


OBJS = $(SRCS:.cpp=.o)

CFLAGS = -O2 \
         -I/usr/local/opencv-2.4.9/include \

LIBS =  ./3rdparty/SSBA-3.0/build/libV3D.a \
        ./3rdparty/SSBA-3.0/build/libcolamd.a \
        /usr/local/lib/libopencv* \

sfm: $(OBJS)
	$(CXX) $(CFLAGS) -o $@ $(OBJS) $(LIBS)

.cpp.o:
	$(CXX) -c $(CFLAGS) $<

clean:
	rm -rf *.o
