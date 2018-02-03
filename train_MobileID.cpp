#include <cstdlib>
#include <iostream>
#include <boost/filesystem.hpp>
#include <caffe2/core/init.h>
#include <caffe2/core/context_gpu.h>
#include <caffe2/util/blob.h>
#include <caffe2/util/model.h>
#include <caffe2/util/net.h>
#include <cvplot/cvplot.h>

#define NDEBUG
#define WITH_CUDA
#define TRAINSIZE 350418
#define BATCHSIZE 80

using namespace std;
using namespace boost::filesystem;
using namespace caffe2;
using namespace cvplot;

void setupTrainNet(NetDef & init, NetDef & predict);
void setupSaveNet(NetDef & save);

unique_ptr<NetBase> predict_net;
unique_ptr<NetBase> save_net;


void atexit_handler()
{
	cout<<"saving params"<<endl;
	remove_all("MobileID_params");
	save_net->Run();
}

int main(int argc,char ** argv)
{
	NetDef init,predict,save;
	setupTrainNet(init,predict);
	setupSaveNet(save);
#ifdef WITH_CUDA
	auto device = CUDA;
#else
	auto device = CPU;
#endif
	init.mutable_device_option()->set_device_type(device);
	predict.mutable_device_option()->set_device_type(device);
	save.mutable_device_option()->set_device_type(device);
	Workspace workspace(nullptr);
	workspace.RunNetOnce(init);
	predict_net = CreateNet(predict,&workspace);
	save_net = CreateNet(save,&workspace);
	atexit(atexit_handler);
#ifndef NDEBUG
	//show loss degradation
	cvplot::window("loss revolution");
	cvplot::move("loss",300,300);
	cvplot::resize("loss",500,300);
	cvplot::figure("loss").series("train").color(cvplot::Purple);
#endif
	for(int i = 0 ; ; i++) {
		predict_net->Run();
		cout<<"iter:"<<i<<endl;
		if(i % 100 == 0) {
			cout<<"saving params"<<endl;
			remove_all("MobileID_params");
			save_net->Run();
		}
	}
	return EXIT_SUCCESS;
}

void setupTrainNet(NetDef & init, NetDef & predict)
{
	ModelUtil MobileID(init,predict);
	MobileID.init.AddCreateDbOp("db","lmdb","./dataset");
	MobileID.predict.AddInput("db");
	MobileID.AddTensorProtosDbInputOp("db","data","feature",BATCHSIZE);
	//data 150x150x3
	MobileID.AddConvOps("data","conv1",3,64,1,0,4);
	MobileID.AddReluOp("conv1","conv1");
	//conv1 146x146x64
	MobileID.AddMaxPoolOp("conv1","pool1",2,0,2);
	//pool1 72x72x64
	MobileID.AddConvOps("pool1","conv2",64,64,1,0,3);
	MobileID.AddReluOp("conv2","conv2");
	//conv2 69x69x64
	MobileID.AddMaxPoolOp("conv2","pool2",2,0,2);
	//pool2 33x33x64
	MobileID.AddConvOps("pool2","conv3",64,64,1,0,3);
	MobileID.AddReluOp("conv3","conv3");
	//conv3 30x30x64
	MobileID.AddMaxPoolOp("conv3","pool3",2,0,2);
	//pool3 14x14x64
	MobileID.AddConvOps("pool3","conv4",64,10,1,0,1);
	MobileID.AddReluOp("conv4","conv4");
	//conv4 14x14x10
	MobileID.AddFcOps("conv4","ip1",2560,500);
	MobileID.AddReluOp("ip1","ip1");
	//ip1 500
	MobileID.AddFcOps("ip1","ip2",500,500);
	MobileID.AddReluOp("ip2","ip2");
	//ip2 500
	MobileID.AddFcOps("ip2","ip3",500,128);
	MobileID.AddReluOp("ip3","output");
	//output 128
	MobileID.AddSquaredL2DistanceOp({"output","feature"},"loss");	
	MobileID.AddConstantFillWithOp(1.0, "loss", "loss_grad");
	MobileID.predict.AddGradientOps();
	MobileID.AddIterOps();
#ifndef NDEBUG
	MobileID.predict.AddTimePlotOp("loss","iter","loss","train",10);
#endif
	MobileID.AddLearningRateOp("iter", "lr", -0.01,0.9,100*round(static_cast<float>(TRAINSIZE)/BATCHSIZE));
	string optimizer = "adam";
	MobileID.AddOptimizerOps(optimizer);
	//输出网络结构
	MobileID.predict.WriteText("models/MobileID_train.pbtxt");
}

void setupSaveNet(NetDef & save)
{
	NetUtil SaveNet(save);
	vector<string> params = {
		"conv1_w","conv1_b",
		"conv2_w","conv2_b",
		"conv3_w","conv3_b",
		"conv4_w","conv4_b",
		"ip1_w","ip1_b",
		"ip2_w","ip2_b",
		"ip3_w","ip3_b"
	};
	SaveNet.AddSaveOp(params,"lmdb","MobileID_params");
	//output network
	SaveNet.WriteText("models/MobileID_save.pbtxt");
}
