#include <cstdlib>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

using namespace std;
using namespace boost::filesystem;
using namespace boost::program_options;
using namespace cv;

void crop(path input,path output,dlib::frontal_face_detector & detector,dlib::shape_predictor & sp);

int main(int argc,char ** argv)
{
	options_description desc;
	string inputdir;
	string outputdir;
	desc.add_options()
		("help,h","print this help message")
		("input,i",value<string>(&inputdir),"input facial image directory")
		("output,o",value<string>(&outputdir),"output cropped image directory");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);
	
	if(1 == argc || 1 != vm.count("input") || 1 != vm.count("output")) {
		cout<<desc;
		return EXIT_FAILURE;
	}
	
	if(false == exists(inputdir) || false == is_directory(inputdir)) {
		cerr<<"invalid input directory"<<endl;
		return EXIT_FAILURE;
	}
	
	if(exists(outputdir)) {
		cerr<<"output directory exists!"<<endl;
		return EXIT_FAILURE;
	}
	
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor sp;
	dlib::deserialize("models/shape_predictor_68_face_landmarks.dat") >> sp;
	
	crop(path(inputdir),path(outputdir),detector,sp);
	
	return EXIT_SUCCESS;
}

void crop(path input,path output,dlib::frontal_face_detector & detector,dlib::shape_predictor & sp)
{
	if(false == exists(output)) create_directory(output);
	for(directory_iterator it(input) ; it != directory_iterator() ; it++) {
		if(is_directory(it->path())) crop(it->path(),output / it->path().filename().string(),detector,sp);
		else {
			Mat img = imread(it->path().string());
			if(img.empty()) {
				cerr<<it->path().string()<<" cant be opened!"<<endl;
				continue;
			}
			dlib::cv_image<dlib::rgb_pixel> cimg(img);
			vector<dlib::rectangle> faces = detector(cimg);
			if(0 == faces.size()) {
				cerr<<it->path().string()<<" has no face!"<<endl;
				continue;
			} else if(1 < faces.size())
				cerr<<it->path().string()<<" has multiple face, only the first detected face is cropped!"<<endl;
			dlib::rectangle & face = faces.front();
			dlib::full_object_detection shape = sp(cimg,face);
			dlib::matrix<dlib::rgb_pixel> face_chip;
			dlib::extract_image_chip(cimg,dlib::get_face_chip_details(shape,150,0.25),face_chip);
			Mat faceimg = dlib::toMat(face_chip);
			imwrite((output / it->path().filename().string()).string(),faceimg);
		}
	}
}
