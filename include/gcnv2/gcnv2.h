#pragma once

#include <string>
#include <vector>
#include <memory> // unique_ptr, shared_ptr
#include <opencv2/features2d.hpp>
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
        // Runtim type
        enum class RUNTIME_TYPE
        {
            TORCH,
            ONNX,
            TVM
        };

        /**
         * @brief The GCNv2 constructor for torchscript
         * 
         * @param param-name param-content
         */
        GCNv2DetectorDescriptor( RUNTIME_TYPE runtime_type, int img_height, int img_width, bool use_cuda );

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
         * @return std::string 
         */
        virtual std::string getDefaultName() const override;
    
    protected:
        // TODO add helper function & helper member varaible here

        // Which model type to use
        RUNTIME_TYPE runtime_type;

        // Model input height & width
        // All three type of runtime require fixed size input
        const int img_height;
        const int img_width;

        // Use CUDA or not
        bool use_cuda;

        // Device
        torch::Device torch_device;
        // TODO: add ONNX/TVM device

        // Use unique ptr to avoid potential data race issue
        std::unique_ptr<torch::jit::script::Module> torch_model;
        // TODO: add ONNX/TVM model

        // Load and initialize model
        void initTorch( );
        void initONNX( );
        void initTVM( );

        // Run deep learning module 
        void detectAndComputeTorch( cv::Mat& _gray_image_fp32, \
                                    std::vector<cv::KeyPoint>& _keypoints, \
                                    cv::OutputArray& _descriptors );

        void detectAndComputeONNX( cv::Mat& _gray_image_fp32, \
                                   std::vector<cv::KeyPoint>& _keypoints, \
                                   cv::OutputArray& _descriptors );

        void detectAndComputeTVM( cv::Mat& _gray_image_fp32, \
                                  std::vector<cv::KeyPoint>& _keypoints, \
                                  cv::OutputArray& _descriptors );
};

};