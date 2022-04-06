#pragma once

#include <string>
#include <vector>
#include <memory> // unique_ptr, shared_ptr
#include <opencv2/opencv.hpp>
#include <torch/script.h>

namespace gcnv2
{

/**
 * @brief GCNv2 detector class
 * 
 * @usage: 
 *  
 */
class GCNv2DetectorDescriptor : public cv::Feature2D
{
    public:
        /**
         * @brief The GCNv2 constructor for torchscript
         * 
         * @param param-name param-content
         */
        GCNv2DetectorDescriptor( int img_height, int img_width );

        /**
         * @brief Destroy the GCNv2DetectorDescriptor object
         * 
         */
        virtual ~GCNv2DetectorDescriptor();

        /**
         * @brief Detect keypoint and compute the descriptor
         * 
         * @note detect() and compute() internally call detectAndCompute() in cv::Feature2D
         * @note when use the GCNv2DetectorDescriptor, we should call detectAndCompute() function directely
         * 
         * @param image Image
         * @param mask Mask specifying where to look for keypoints (optional). 
         *   It must be a 8-bit integer matrix with non-zero values in the region of interest.
         *   In our GCNv2DetectorDescriptor, mask is currently not supported
         * @param keypoints Detected keypoints. Should be empty container when input
         * @param descriptors Computed descriptors. Should be empty when input
         * @param useProvidedKeypoints use provided keypoints to compute descriptor
         *   In our GCNv2DetectorDescriptor, this argument should always be false since we compute keypoint
         *   inside the detectAndCompute() function
         * 
         */
        virtual void detectAndCompute( cv::InputArray image, \
                                       cv::InputArray mask, \
                                       std::vector<cv::KeyPoint>& keypoints, \
                                       cv::OutputArray& descriptors, \
                                       bool useProvidedKeypoints=false ) override;

        /**
         * @brief Getname of this extractor, used in some opencv funciton
         * 
         * @return cv::String
         */
        virtual cv::String getDefaultName() const override;
    
    protected:
        // Model input height & width
        // All three type of runtime require fixed size input
        const int img_height;
        const int img_width;

        // Device
        torch::Device torch_device = torch::kCUDA;

        // Use unique ptr to avoid potential data race issue
        std::shared_ptr<torch::jit::script::Module> torch_model;

        // Load and initialize model
        void initTorch( );

        // Run deep learning module 
        void detectAndComputeTorch( cv::Mat& _gray_image_fp32, \
                                    std::vector<cv::KeyPoint>& _keypoints, \
                                    cv::OutputArray& _descriptors );
};

}