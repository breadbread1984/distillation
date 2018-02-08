#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/program_options.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <caffe2/core/common.h>
#include <caffe2/core/db.h>
#include <caffe2/core/init.h>
#include <caffe2/proto/caffe2.pb.h>

#define NUM 40

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace boost::program_options;
using namespace cv;
using namespace caffe2;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128,dlib::avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,
                            dlib::input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

void makeList(path dir,vector<string> & filelist);
void write2db(path dir,int & count,TensorProtos & protos,TensorProto * input,TensorProto * output, unique_ptr<db::Transaction> & transaction,anet_type & net,dlib::shape_predictor & sp,dlib::frontal_face_detector & detector);
vector<dlib::matrix<dlib::rgb_pixel>> jitter_image(const dlib::matrix<dlib::rgb_pixel>& img);

int main(int argc,char ** argv)
{
	string inputdir,outputdir;
	options_description desc;
	desc.add_options()
		("help,h","打印当前使用方法")
		("input,i",value<string>(&inputdir),"输入图片所在文件夹")
		("output,o",value<string>(&outputdir),"输出lmdb所在文件夹");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);

	if(1 == argc || 1 != vm.count("input") || 1 != vm.count("output")) {
		std::cout<<desc;
		return EXIT_SUCCESS;
	}

	if(false == exists(inputdir) || false == is_directory(inputdir)) {
		cout<<"输入图片所在文件夹不正确"<<endl;
		return EXIT_FAILURE;
	}
	
	if(exists(outputdir)) {
		cout<<"输出路径已经存在"<<endl;
		return EXIT_FAILURE;
	}
	
	anet_type net[NUM];
	dlib::shape_predictor sp[NUM];
	dlib::frontal_face_detector detector[NUM];
	for(int i = 0 ; i < NUM ; i++) {
		dlib::deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> net[i];
		dlib::deserialize("models/shape_predictor_68_face_landmarks.dat") >> sp[i];
		detector[i] = dlib::get_frontal_face_detector();
	}
	
	unique_ptr<db::DB> db(db::CreateDB("lmdb",outputdir,db::NEW));
	int count = 0;
	//create dataset list
	cout<<"1)making image list"<<endl;
	vector<string> filelist;
	makeList(inputdir,filelist);
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	shuffle(filelist.begin(),filelist.end(),default_random_engine(seed));
	cout<<"there are "<<filelist.size()<<" images found"<<endl;
	//write to lmdb
	cout<<"2)writing to lmdb"<<endl;
	vector<boost::shared_ptr<boost::thread> > handler;
	boost::mutex mutex;
	for(auto & file : filelist) {
		handler.push_back(boost::shared_ptr<boost::thread>(
			new boost::thread(boost::bind<void>(
				[&](string filename,int index) -> void {
					//1)detect faces
					Mat img = imread(filename);
					if(true == img.empty()) {
						cout<<filename<<" cant be opened!"<<endl;
						return;
					}
					dlib::cv_image<dlib::rgb_pixel> cimg(img);
					vector<dlib::rectangle> faces = detector[index](cimg);
					if(0 == faces.size()) {
						cout<<filename<<" has no face"<<endl;
						return;
					}
					dlib::rectangle & face = faces.front();
					//2)extract feature
					dlib::full_object_detection shape = sp[index](cimg,face);
					dlib::matrix<dlib::rgb_pixel> face_chip;
					dlib::extract_image_chip(cimg,dlib::get_face_chip_details(shape,150,0.25),face_chip);
					dlib::matrix<float,0,1> fv = mean(mat(net[index](jitter_image(face_chip))));
					//3)open lmdb (transaction can only be used in thread which created it)
					unique_ptr<db::Transaction> transaction(db->NewTransaction());
					TensorProtos protos;
					TensorProto * input = protos.add_protos();
					input->set_data_type(TensorProto::FLOAT);
					input->add_dims(3);
					input->add_dims(150);
					input->add_dims(150);
					TensorProto * output = protos.add_protos();
					output->set_data_type(TensorProto::FLOAT);
					output->add_dims(128);
					//4)write to lmdb
					input->clear_float_data();
					output->clear_float_data();
					for(int c = 0 ; c < 3 ; c++)
						for(int h = 0 ; h < face_chip.nr() ; h++)
							for(int w = 0 ; w < face_chip.nc() ; w++) {
								switch(c) {
									case 0:
										input->add_float_data(face_chip(h,w).blue);
										break;
									case 1:
										input->add_float_data(face_chip(h,w).green);
										break;
									case 2:
										input->add_float_data(face_chip(h,w).red);
										break;
									default:
										assert(0);
								}
							}
					for(auto it = fv.begin() ; it != fv.end() ; it++)
						output->add_float_data(*it);
					string value;
					protos.SerializeToString(&value);
					stringstream sstr;
					{
						boost::mutex::scoped_lock scoped_lock(mutex);
						sstr<<setw(8)<<setfill('0')<<count++;
					}
					transaction->Put(sstr.str(),value);
					transaction->Commit();
					cout<<"has written "<<count<<" samples"<<endl;
					return;
				},file,handler.size()
			))
		));
		if(NUM <= handler.size()) {
#ifndef NDEBUG
			cout<<"waiting for join"<<endl;
#endif
			for(auto & h : handler) h->join();
#ifndef NDEBUG
			cout<<"all joint"<<endl;
#endif
			handler.clear();
		}
	}
	if(handler.size()) {
		for(auto & h : handler) h->join();
		handler.clear();
	}
	cout<<"共有"<<count<<"合成样本！"<<endl;
	
	return EXIT_SUCCESS;
}

void makeList(path dir,vector<string> & filelist)
{
	regex expr(".*\\.(png|PNG|jpg|JPG|jpeg|JPEG)");
	for(directory_iterator itr(dir) ; itr != directory_iterator() ; itr++) {
		if(is_directory(itr->path())) makeList(itr->path(),filelist);
		else if(regex_match(itr->path().filename().string(),expr)) filelist.push_back(itr->path().string());
	}
}

vector<dlib::matrix<dlib::rgb_pixel>> jitter_image(const dlib::matrix<dlib::rgb_pixel>& img)
{
	//提取人脸的特征
    // All this function does is make 100 copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently.
    thread_local dlib::random_cropper cropper;
    cropper.set_chip_dims(150,150);
    cropper.set_randomly_flip(true);
    cropper.set_max_object_size(0.99999);
    cropper.set_background_crops_fraction(0);
    cropper.set_min_object_size(146,145);
    cropper.set_translate_amount(0.02);
    cropper.set_max_rotation_degrees(3);

    std::vector<dlib::mmod_rect> raw_boxes(1), ignored_crop_boxes;
    raw_boxes[0] = shrink_rect(get_rect(img),3);
    std::vector<dlib::matrix<dlib::rgb_pixel>> crops; 

    dlib::matrix<dlib::rgb_pixel> temp; 
    for (int i = 0; i < 100; ++i)
    {
        cropper(img, raw_boxes, temp, ignored_crop_boxes);
        crops.push_back(std::move(temp));
    }
    return crops;
}
