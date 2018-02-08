#include <cstdlib>
#include <iostream>
#include <boost/filesystem.hpp>
#include <caffe2/core/init.h>
#include <caffe2/core/context_gpu.h>
#include <caffe2/util/blob.h>
#include <caffe2/util/model.h>
#include <caffe2/util/net.h>
#include <caffe2/zoo/mobilenet.h>
#include <cvplot/cvplot.h>

#define NDEBUG
#define WITH_CUDA
#define TRAINSIZE 350418
#define BATCHSIZE 30

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
	remove_all("MobileNet_params");
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
			remove_all("MobileNet_params");
			save_net->Run();
		}
	}
	return EXIT_SUCCESS;
}

void setupTrainNet(NetDef & init, NetDef & predict)
{
	MobileNetModel MobileNet(init,predict);
	MobileNet.init.AddCreateDbOp("db","lmdb","./dataset");
	MobileNet.predict.AddInput("db");
	MobileNet.AddTensorProtosDbInputOp("db","data","feature",BATCHSIZE);
	
    MobileNet.predict.SetName("MobileNet");
    auto input = "data";
    auto n = 0;
	auto alpha = 1.0;
	bool train = true;

    std::string layer = input;

    layer = MobileNet.AddFirst("1", layer, 32, 2, alpha, train)->output(0);
    layer = MobileNet.AddFilter(tos2(n++), layer, 32, 64, 1, alpha, train)->output(0);
    layer = MobileNet.AddFilter(tos2(n++), layer, 64, 128, 2, alpha, train)->output(0);
    layer = MobileNet.AddFilter(tos2(n++), layer, 128, 128, 1, alpha, train)->output(0);
    layer = MobileNet.AddFilter(tos2(n++), layer, 128, 256, 2, alpha, train)->output(0);
    layer = MobileNet.AddFilter(tos2(n++), layer, 256, 256, 1, alpha, train)->output(0);
    layer = MobileNet.AddFilter(tos2(n++), layer, 256, 512, 2, alpha, train)->output(0);
    for (auto i = 0; i < 5; i++) {  // 6 - 10
      layer = MobileNet.AddFilter(tos2(n++), layer, 512, 512, 1, alpha, train)->output(0);
    }
    layer = MobileNet.AddFilter(tos2(n++), layer, 512, 1024, 2, alpha, train)->output(0);
    layer = MobileNet.AddFilter(tos2(n++), layer, 1024, 1024, 1, alpha, train)->output(0);
	MobileNet.AddAveragePoolOp(layer, "final_avg", 1, 0, 5);
	MobileNet.AddFcOps("final_avg", "last_out", 1024, 128, train);
	
	MobileNet.AddSquaredL2DistanceOp({"last_out","feature"},"loss");
	MobileNet.AddConstantFillWithOp(1.0, "loss", "loss_grad");
	MobileNet.predict.AddGradientOps();
	MobileNet.AddIterOps();
#ifndef NDEBUG
	MobileNet.predict.AddTimePlotOp("loss","iter","train",10);
#endif
	MobileNet.AddLearningRateOp("iter", "lr", -0.01,0.9,100*round(static_cast<float>(TRAINSIZE)/BATCHSIZE));
	string optimizer = "adam";
	MobileNet.AddOptimizerOps(optimizer);
	//输出网络结构
	MobileNet.init.WriteText("models/MobileNet_train_init.pbtxt");
	MobileNet.predict.WriteText("models/MobileNet_train_predict.pbtxt");
}

void setupSaveNet(NetDef & save)
{
	NetUtil SaveNet(save);
	vector<string> params = {
		"conv1_w","conv1_spatbn_s","conv1_spatbn_b","conv1_spatbn_rm","conv1_spatbn_riv",
		"comp_0_conv_1_w","comp_0_spatbn_1_s","comp_0_spatbn_1_b","comp_0_spatbn_1_rm","comp_0_spatbn_1_riv",
		"comp_0_conv_2_w","comp_0_spatbn_2_s","comp_0_spatbn_2_b","comp_0_spatbn_2_rm","comp_0_spatbn_2_riv",
		"comp_1_conv_1_w","comp_1_spatbn_1_s","comp_1_spatbn_1_b","comp_1_spatbn_1_rm","comp_1_spatbn_1_riv",
		"comp_1_conv_2_w","comp_1_spatbn_2_s","comp_1_spatbn_2_b","comp_1_spatbn_2_rm","comp_1_spatbn_2_riv",
		"comp_2_conv_1_w","comp_2_spatbn_1_s","comp_2_spatbn_1_b","comp_2_spatbn_1_rm","comp_2_spatbn_1_riv",
		"comp_2_conv_2_w","comp_2_spatbn_2_s","comp_2_spatbn_2_b","comp_2_spatbn_2_rm","comp_2_spatbn_2_riv",
		"comp_3_conv_1_w","comp_3_spatbn_1_s","comp_3_spatbn_1_b","comp_3_spatbn_1_rm","comp_3_spatbn_1_riv",
		"comp_3_conv_2_w","comp_3_spatbn_2_s","comp_3_spatbn_2_b","comp_3_spatbn_2_rm","comp_3_spatbn_2_riv",
		"comp_4_conv_1_w","comp_4_spatbn_1_s","comp_4_spatbn_1_b","comp_4_spatbn_1_rm","comp_4_spatbn_1_riv",
		"comp_4_conv_2_w","comp_4_spatbn_2_s","comp_4_spatbn_2_b","comp_4_spatbn_2_rm","comp_4_spatbn_2_riv",
		"comp_5_conv_1_w","comp_5_spatbn_1_s","comp_5_spatbn_1_b","comp_5_spatbn_1_rm","comp_5_spatbn_1_riv",
		"comp_5_conv_2_w","comp_5_spatbn_2_s","comp_5_spatbn_2_b","comp_5_spatbn_2_rm","comp_5_spatbn_2_riv",
		"comp_6_conv_1_w","comp_6_spatbn_1_s","comp_6_spatbn_1_b","comp_6_spatbn_1_rm","comp_6_spatbn_1_riv",
		"comp_6_conv_2_w","comp_6_spatbn_2_s","comp_6_spatbn_2_b","comp_6_spatbn_2_rm","comp_6_spatbn_2_riv",
		"comp_7_conv_1_w","comp_7_spatbn_1_s","comp_7_spatbn_1_b","comp_7_spatbn_1_rm","comp_7_spatbn_1_riv",
		"comp_7_conv_2_w","comp_7_spatbn_2_s","comp_7_spatbn_2_b","comp_7_spatbn_2_rm","comp_7_spatbn_2_riv",
		"comp_8_conv_1_w","comp_8_spatbn_1_s","comp_8_spatbn_1_b","comp_8_spatbn_1_rm","comp_8_spatbn_1_riv",
		"comp_8_conv_2_w","comp_8_spatbn_2_s","comp_8_spatbn_2_b","comp_8_spatbn_2_rm","comp_8_spatbn_2_riv",
		"comp_9_conv_1_w","comp_9_spatbn_1_s","comp_9_spatbn_1_b","comp_9_spatbn_1_rm","comp_9_spatbn_1_riv",
		"comp_9_conv_2_w","comp_9_spatbn_2_s","comp_9_spatbn_2_b","comp_9_spatbn_2_rm","comp_9_spatbn_2_riv",
		"comp_10_conv_1_w","comp_10_spatbn_1_s","comp_10_spatbn_1_b","comp_10_spatbn_1_rm","comp_10_spatbn_1_riv",
		"comp_10_conv_2_w","comp_10_spatbn_2_s","comp_10_spatbn_2_b","comp_10_spatbn_2_rm","comp_10_spatbn_2_riv",
		"comp_11_conv_1_w","comp_11_spatbn_1_s","comp_11_spatbn_1_b","comp_11_spatbn_1_rm","comp_11_spatbn_1_riv",
		"comp_11_conv_2_w","comp_11_spatbn_2_s","comp_11_spatbn_2_b","comp_11_spatbn_2_rm","comp_11_spatbn_2_riv",
		"comp_12_conv_1_w","comp_12_spatbn_1_s","comp_12_spatbn_1_b","comp_12_spatbn_1_rm","comp_12_spatbn_1_riv",
		"comp_12_conv_2_w","comp_12_spatbn_2_s","comp_12_spatbn_2_b","comp_12_spatbn_2_rm","comp_12_spatbn_2_riv",
		"last_out_w","last_out_b"
	};
	SaveNet.AddSaveOp(params,"lmdb","MobileNet_params");
	//output network
	SaveNet.WriteText("models/MobileNet_save.pbtxt");
}
