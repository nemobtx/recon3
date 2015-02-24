.SUFFIXES: .cpp

CXX = clang++  

SRCS = main.cpp XBuilder.cpp sfm.cpp fmatrix_matching.cpp triangulation.cpp
OBJS = $(SRCS:.cpp=.o)

CFLAGS = -O2 \
         -I/usr/local/opencv-2.4.9/include \

LIBS =  ../sfm/3rdparty/SSBA-3.0/build/libV3D.a \
        ../sfm/3rdparty/SSBA-3.0/build/libcolamd.a \
        /usr/local/lib/libopencv* \

xBulder: $(SRCS) XBuilder.h Makefile
	$(CXX) $(CFLAGS) -o $@ $(SRCS) $(LIBS)

.cpp.o:
	$(CXX) -c $(CFLAGS) $<

clean:
	rm -rf *.o
