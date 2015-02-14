.SUFFIXES: .cpp

CXX = clang++  

SRCS = AbstractFeatureMatcher.cpp BundleAdjuster.cpp Common.cpp Distance.cpp FindCameraMatrices.cpp \
       MultiCameraDistance.cpp MultiCameraPnP.cpp OFFeatureMatcher.cpp RichFeatureMatcher.cpp \
       Triangulation.cpp \
       file_Interface.cpp \
       main.cpp


OBJS = $(SRCS:.cpp=.o)

CFLAGS = -O2 \
         -I/usr/local/opencv-2.4.9/include \
         -I ./3rdparty/SSBA-3.0/ 

LIBS =  ./3rdparty/SSBA-3.0/build/libV3D.a \
        ./3rdparty/SSBA-3.0/build/libcolamd.a \
        /usr/local/lib/libopencv* \

sfm: $(OBJS)
	$(CXX) $(CFLAGS) -o $@ $(OBJS) $(LIBS)

.cpp.o:
	$(CXX) -c $(CFLAGS) $<

clean:
	rm -rf *.o
