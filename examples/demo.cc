#include <cstdio>
#include <vector>
#include <opencv2/opencv.hpp>
#include <gcnv2/gcnv2.h> // our DL detector

int main(int argc, const char* argv[])
{
    printf("Remember to export model filename path to enviroment before run this example\n");

    if ( argc != 1 )
    {
        printf("Usage: ./demo PATH_TO_IMAGE\n");
        exit(1);
    }

    // Set image input
    printf("..Read in example image\n"); fflush( stdout );
    cv::Mat image = cv::imread( argv[1], CV_LOAD_IMAGE_COLOR );

    // Create a detector descriptor instance
    cv::Feature2D GCNv2_detector_descriptor = gcnv2::GCNv2DetectorDescriptor( \
        gcnv2::GCNv2DetectorDescriptor::RUNTIME_TYPE::TORCH, 640, 480, false );
    
    // Run detection
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    GCNv2_detector_descriptor.detectAndCompute( image, cv::noArray(), keypoints, descriptors );

    // Visualize result
    // TODO: Afroz please add ur visualization here
}