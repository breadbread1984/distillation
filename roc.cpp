#include <cstdlib>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/tuple/tuple.hpp>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <caffe2/core/init.h>
#include <caffe2/core/context_gpu.h>
#include <caffe2/util/blob.h>
#include <caffe2/util/model.h>
#include <caffe2/util/net.h>
#include <caffe2/zoo/mobilenet.h>

//#define MOBILEID

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace boost::program_options;
namespace ublas = boost::numeric::ublas;
using namespace cv;
using namespace caffe2;

void setDeployNet(NetDef & init, NetDef & predict);
vector<boost::tuple<string,string,bool> > loadList(std::ifstream & list);
TensorCPU preProcess(dlib::matrix<dlib::rgb_pixel> & face_chip);

int main(int argc,char ** argv)
{
	options_description desc;
	string inputdir;
	string listfile;
	desc.add_options()
		("help,h","打印当前使用方法")
		("input,i",value<string>(&inputdir),"LFW文件夹路径")
		("pair,p",value<string>(&listfile),"LFW验证列表文件路径");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);
	
	if(1 == argc || vm.count("help") || 1 != vm.count("input") || 1 != vm.count("pair")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}
	
	path inputroot(inputdir);
	if(false == exists(inputroot) || false == is_directory(inputroot)) {
		cout<<"LFW文件夹不存在！"<<endl;
		return EXIT_FAILURE;
	}
	std::ifstream verifpair(listfile);
	if(false == verifpair.is_open()) {
		cout<<"LFW验证列表文件无法打开！"<<endl;
		return EXIT_FAILURE;
	}
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor sp;
	dlib::deserialize("models/shape_predictor_68_face_landmarks.dat") >> sp;
	vector<boost::tuple<string,string,bool> > list = loadList(verifpair);
	//1)calculate distances
	vector<float> pos_dists,neg_dists;
	NetDef init,predict;
	setDeployNet(init,predict);
	init.mutable_device_option()->set_device_type(CUDA);
	predict.mutable_device_option()->set_device_type(CUDA);
	Workspace workspace;
	workspace.RunNetOnce(init);
	unique_ptr<NetBase> predict_net = CreateNet(predict,&workspace);
	for(vector<boost::tuple<string,string,bool> >::iterator it = list.begin() ; it != list.end() ; it++) {
		Mat img1 = imread((inputroot / get<0>(*it)).string());
		Mat img2 = imread((inputroot / get<1>(*it)).string());
		if(img1.empty()) cout<<(inputroot / get<0>(*it)).string()<<"cant be opened"<<endl;
		if(img2.empty()) cout<<(inputroot / get<1>(*it)).string()<<"cant be opened"<<endl;
		
		dlib::cv_image<dlib::rgb_pixel> cimg1(img1);
		vector<dlib::rectangle> faces = detector(cimg1);
		if(0 == faces.size()) {
			cout<<(inputroot / get<0>(*it)).string()<<" has no face"<<endl;
			continue;
		}
		dlib::rectangle & facerect1 = faces.front();
		dlib::full_object_detection shape1 = sp(cimg1,facerect1);
		dlib::matrix<dlib::rgb_pixel> face_chip;
		dlib::extract_image_chip(cimg1,dlib::get_face_chip_details(shape1,150,0.25),face_chip);
		TensorCUDA input1 = TensorCUDA(preProcess(face_chip));
		auto tensor1 = workspace.CreateBlob("data")->GetMutable<TensorCUDA>();
		tensor1->ResizeLike(input1);
		tensor1->ShareData(input1);
		predict_net->Run();
#if defined MOBILEID
		TensorCPU output1 = TensorCPU(workspace.GetBlob("output")->Get<TensorCUDA>());
#else
		TensorCPU output1 = TensorCPU(workspace.GetBlob("last_out")->Get<TensorCUDA>());
#endif
		ublas::vector<float> fv1(128);
		copy(output1.data<float>(),output1.data<float>() + 128,fv1.begin());
		
		dlib::cv_image<dlib::rgb_pixel> cimg2(img2);
		faces = detector(cimg2);
		if(0 == faces.size()) {
			cout<<(inputroot / get<1>(*it)).string()<<" has no face"<<endl;
			continue;
		}
		dlib::rectangle & facerect2 = faces.front();
		dlib::full_object_detection shape2 = sp(cimg2,facerect2);
		dlib::extract_image_chip(cimg2,dlib::get_face_chip_details(shape2,150,0.25),face_chip);
		TensorCUDA input2 = TensorCUDA(preProcess(face_chip));
		auto tensor2 = workspace.CreateBlob("data")->GetMutable<TensorCUDA>();
		tensor2->ResizeLike(input2);
		tensor2->ShareData(input2);
		predict_net->Run();
#if defined MOBILEID
		TensorCPU output2 = TensorCPU(workspace.GetBlob("output")->Get<TensorCUDA>());
#else
		TensorCPU output2 = TensorCPU(workspace.GetBlob("last_out")->Get<TensorCUDA>());
#endif
		ublas::vector<float> fv2(128);
		copy(output2.data<float>(),output2.data<float>() + 128,fv2.begin());
		
		ublas::vector<float> diff = fv1 - fv2;
		float dist = sqrt(inner_prod(diff,diff));
		if(get<2>(*it)) pos_dists.push_back(dist);
		else neg_dists.push_back(dist);
	}
	//2)calculate roc curve
	set<float> thresholds;
	thresholds.insert(pos_dists.begin(),pos_dists.end());
	thresholds.insert(neg_dists.begin(),neg_dists.end());
	std::ofstream out("roc.txt");
	map<float,boost::tuple<float,float> > roc;
	float auc = 0;
	float prev_truepos_rate, prev_falsepos_rate;
	for(set<float>::iterator it = thresholds.begin() ; it != thresholds.end() ; it++) {
		float threshold = *it;
		int truepos = 0, falsepos = 0;
		int falseneg = 0, trueneg = 0;
		for(int i = 0 ; i < pos_dists.size() ; i++) if(pos_dists[i] < threshold) truepos++; else falseneg++;
		for(int i = 0 ; i < neg_dists.size() ; i++) if(neg_dists[i] > threshold) trueneg++; else falsepos++;
		float truepos_rate = static_cast<float>(truepos) / (truepos + falseneg);
		float falsepos_rate = static_cast<float>(falsepos) / (falsepos + trueneg);
		roc.insert(make_pair(*it,boost::make_tuple(falsepos_rate,truepos_rate)));
		if(it != thresholds.begin()) auc += 0.5 * (falsepos_rate - prev_falsepos_rate) * (truepos_rate + prev_truepos_rate);
		prev_truepos_rate = truepos_rate; prev_falsepos_rate = falsepos_rate;
		out<<falsepos_rate<<" "<<truepos_rate<<" "<<*it<<endl;
	}
	cout<<"auc = "<<auc<<endl;
	
	return EXIT_SUCCESS;
}

vector<boost::tuple<string,string,bool> > loadList(std::ifstream & list)
{
	vector<boost::tuple<string,string,bool> > retVal;
	string line;
	int n_set,n_num;
	getline(list,line);
	stringstream sstr;
	sstr<<line;
	sstr>>n_set>>n_num;
	for(int i = 0 ; i < n_set ; i++) {
		for(int j = 0 ; j < n_num ; j++) {
			getline(list,line);
			stringstream sstr;
			sstr << line;
			string name; int id1,id2;
			sstr>>name >>id1>>id2;
			ostringstream ss1,ss2;
			ss1<<setw(4)<<setfill('0')<<id1;
			ss2<<setw(4)<<setfill('0')<<id2;
			string file1 = name + "/" + name + "_" + ss1.str() + ".jpg";
			string file2 = name + "/" + name + "_" + ss2.str() + ".jpg";
			retVal.push_back(boost::make_tuple(file1,file2,true));
		}
		for(int j = 0 ; j < n_num ; j++) {
			getline(list,line);
			stringstream sstr;
			sstr << line;
			string name1,name2; int id1,id2;
			sstr >>name1 >> id1>>name2>>id2;
			ostringstream ss1,ss2;
			ss1<<setw(4)<<setfill('0')<<id1;
			ss2<<setw(4)<<setfill('0')<<id2;
			string file1 = name1 + "/" + name1 + "_" + ss1.str() + ".jpg";
			string file2 = name2 + "/" + name2 + "_" + ss2.str() + ".jpg";
			retVal.push_back(boost::make_tuple(file1,file2,false));
		}
	}
	return retVal;
}

void setDeployNet(NetDef & init, NetDef & predict)
{
#if defined MOBILEID
	ModelUtil MobileID(init,predict);
	vector<string> params = {
		"conv1_w","conv1_b",
		"conv2_w","conv2_b",
		"conv3_w","conv3_b",
		"conv4_w","conv4_b",
		"ip1_w","ip1_b",
		"ip2_w","ip2_b",
		"ip3_w","ip3_b"
	};
	//1)init net
	MobileID.init.AddLoadOp(params,"lmdb","MobileID_params");
	//2)predict net
	MobileID.init.AddConstantFillOp({1},"data");
	MobileID.predict.AddInput("data");
	//data 150x150x3
	MobileID.AddConvOps("data","conv1",3,64,1,0,4,0,true);
	MobileID.AddLeakyReluOp("conv1","conv1",0.2);
	//conv1 146x146x64
	MobileID.AddMaxPoolOp("conv1","pool1",2,0,2);
	//pool1 72x72x64
	MobileID.AddConvOps("pool1","conv2",64,64,1,0,3,0,true);
	MobileID.AddLeakyReluOp("conv2","conv2",0.2);
	//conv2 69x69x64
	MobileID.AddMaxPoolOp("conv2","pool2",2,0,2);
	//pool2 33x33x64
	MobileID.AddConvOps("pool2","conv3",64,64,1,0,3,0,true);
	MobileID.AddLeakyReluOp("conv3","conv3",0.2);
	//conv3 30x30x64
	MobileID.AddMaxPoolOp("conv3","pool3",2,0,2);
	//pool3 14x14x64
	MobileID.AddConvOps("pool3","conv4",64,10,1,0,1,0,true);
	MobileID.AddLeakyReluOp("conv4","conv4",0.2);
	//conv4 14x14x10
	MobileID.AddFcOps("conv4","ip1",2560,500,true);
	MobileID.AddLeakyReluOp("ip1","ip1",0.2);
	//ip1 500
	MobileID.AddFcOps("ip1","ip2",500,500,true);
	MobileID.AddLeakyReluOp("ip2","ip2",0.2);
	//ip2 500
	MobileID.AddFcOps("ip2","ip3",500,128,true);
	MobileID.AddLeakyReluOp("ip3","output",0.2);
	//3)output to bptxt
	MobileID.init.WriteText("models/MobileID_init.pbtxt");
	MobileID.predict.WriteText("models/MobileID_deploy.pbtxt");
#else
	MobileNetModel MobileNet(init,predict);
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
	//1)init net
	MobileNet.init.AddLoadOp(params,"lmdb","MobileNet_params");
	//2)predict net
	MobileNet.init.AddConstantFillOp({1},"data");
	MobileNet.predict.AddInput("data");
    MobileNet.predict.SetName("MobileNet");
    auto input = "data";
    auto n = 0;
	auto alpha = 1.0;
	bool train = false;

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
#endif
}

TensorCPU preProcess(dlib::matrix<dlib::rgb_pixel> & face_chip)
{
	assert(150 == face_chip.nr());
	assert(150 == face_chip.nc());
	vector<TIndex> dims({1,3,face_chip.nr(),face_chip.nc()});
	vector<float> data;
	
	for(int c = 0 ; c < 3 ; c++)
		for(int h = 0 ; h < face_chip.nr() ; h++)
			for(int w = 0 ; w < face_chip.nc() ; w++) {
				switch(c) {
					case 0:
						data.push_back(face_chip(h,w).blue);
						break;
					case 1:
						data.push_back(face_chip(h,w).green);
						break;
					case 2:
						data.push_back(face_chip(h,w).red);
						break;
					default:
						assert(0);
				}
			}
	assert(data.size() == 1 * 3 * face_chip.nr() * face_chip.nc());
	return TensorCPU(dims,data,NULL);
}
