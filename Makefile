CAFFE2_PREFIX=/home/xieyi/opt/caffe2
CAFFE2_HELPER_PREFIX=/home/xieyi/opt/caffe2_helper
CXXFLAGS=`pkg-config --cflags opencv dlib-1 eigen3` -I. -I${CAFFE2_PREFIX}/include \
-I${CAFFE2_HELPER_PREFIX}/include -std=c++14 -O2
LIBS=-lboost_program_options \
-L${CAFFE2_HELPER_PREFIX}/lib -lcaffe2_cpp -lcaffe2_cpp_gpu \
-L${CAFFE2_PREFIX}/lib -lcaffe2_gpu -lcaffe2 \
`pkg-config --libs opencv dlib-1 eigen3` \
-lglog -lprotobuf -lcudart -lcurand \
-lboost_filesystem -lboost_system -lboost_thread -lboost_regex -lpthread
OBJS=$(patsubst %.cpp,%.o,$(wildcard *.cpp))

all: createLMDB train_MobileID train_MobileNet

createLMDB: createLMDB.o
	$(CXX) $^ $(LIBS) -o ${@}

train_MobileID: train_MobileID.o
	$(CXX) $^ $(LIBS) -o ${@}

train_MobileNet: train_MobileNet.o
	$(CXX) $^ $(LIBS) -o ${@}

clean:
	$(RM) createLMDB train_MobileID train_MobileNet $(OBJS)
