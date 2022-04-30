#include "include/IP.h"

/*
blur 5 3.7 img/grenouille.jpg
laplacian img/grenouille.jpg
separable 5 3.7 img/grenouille.jpg
denoise 3 50 img/grenouille.jpg
*/

int main(int argc, char** argv){

  	Image result;
	std::string choice = argv[1];
	
	if (choice == "blur"){
		Image img(argv[4]); img.display("Image");
		result = img.Blur(atoi(argv[2]), atof(argv[3]));
		result.display("Gaussian - Blur");
	}
	else if (choice == "laplacian") {
		Image img(argv[2]); img.display("Image");
		result = img.laplacian_Filter();
		result.display("Laplacian");
	}
	else if (choice == "separable") {
		Image img(argv[4]); img.display("Image");
		result = img.gaussian_Separable(atoi(argv[2]), atof(argv[3]));
		result.display("Separable Filter");
	}
	else if (choice == "denoise") {
		Image img(argv[4]); img.display("Image");
		result = img.denoise(atoi(argv[2]), atof(argv[3]));
		result.display("Denoise");
	}
	else {
		std::cout << "Wrong Command" << std::endl;
	}
  

  cv::waitKey();
  return 0;
}
