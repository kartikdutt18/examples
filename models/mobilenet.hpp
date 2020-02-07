/**
 * @file mobilenet.hpp
 * @author Kartik Dutt
 * 
 * Implementation of mobilenet-v1 using mlpack.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MOBILENET_HPP
#define MOBILENET_HPP

// Include all required libraries.
#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

/**
 * MobileNet v1 models for mlpack.
 * 
 * MobileNet is a general architecture and can be used for multiple use cases.
 * Depending on the use case, it can use different input layer size and different 
 * width factors. This allows different width models to reduce the number of 
 * multiply-adds and thereby reduce inference cost on mobile devices.
 * 
 * The following table describes the size and accuracy of the 100% MobileNet
 * on size 224 x 224:
 * ----------------------------------------------------------------------------
 * Width Multiplier (alpha) | ImageNet Acc |  Multiply-Adds (M) |  Params (M)
 * ----------------------------------------------------------------------------
 * |   1.0 MobileNet-224    |    70.6 %     |        529        |     4.2     |
 * |   0.75 MobileNet-224   |    68.4 %     |        325        |     2.6     |
 * |   0.50 MobileNet-224   |    63.7 %     |        149        |     1.3     |
 * |   0.25 MobileNet-224   |    50.6 %     |        41         |     0.5     |
 * ----------------------------------------------------------------------------
 * 
 * @code
 * @article{
 *   author  = {Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko,
 *              Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam},
 *   title   = {MobileNets: Efficient Convolutional Neural Networks 
 *              for Mobile Vision Applications},
 *   year    = {2017}
 * }
 * @endcode
 */

class MobileNetV1
{
 public:
  //! Create the Transposed Convolution object.
  MobileNetV1();

  /**
   * MobileNetV1 constructor intializes input shape, number of classes
   * and width multiplier.
   *  
   * @param inputWidth Width of the input image.
   * @param inputHeight Height of the input image.
   * @param inputChannels Number of input channels of the input image.
   * @param numClasses Optional number of classes to classify images into,
   *                   only to be specified if includeTop is  true.
   * @param alpha Controls the width of the network. This is known as the
   *              width multiplier in the MobileNet paper.
   *              - If `alpha` < 1.0, proportionally decreases the number
   *                of filters in each layer.
   *             - If `alpha` > 1.0, proportionally increases the number
   *                of filters in each layer.
   *            - If `alpha` = 1, default number of filters from the paper
   *              are used at each layer.
   * @param depthMultiplier Depth multiplier for depthwise convolution.
   *        This is called the resolution multiplier in the MobileNet paper.
   * @param includeTop whether to include the fully-connected layer at 
   *        the top of the network.
   * @param pooling Optional pooling mode for feature extraction when
   *        include_top is false.
   * @param weights One of 'none', 'imagenet'(pre-training on ImageNet) or path to weights.
   */
   MobileNetV1(const size_t inputWidth,
               const size_t inputHeight,
               const size_t inputChannel,
               const size_t numClasses = 1000,
               const double alpha = 1.0,
               const int depthMultiplier = 1,
               const bool includeTop = true,
               const std::string &pooling = "max",
               const std::string &weights = "None");

   /**
   * MobileNetV1 constructor intializes input shape, number of classes
   * and width multiplier.
   *  
   * @param inputShape A three-valued tuple indicating input shape.
   *                   First value is number of Channels (Channels-First).
   *                   Second value is input height.
   *                   Third value is input width.
   * @param numClasses Optional number of classes to classify images into,
   *                    only to be specified if includeTop is  true.
   * @param alpha Controls the width of the network. This is known as the
   *              width multiplier in the MobileNet paper.
   *              - If `alpha` < 1.0, proportionally decreases the number
   *                of filters in each layer.
   *              - If `alpha` > 1.0, proportionally increases the number
   *                of filters in each layer.
   *             - If `alpha` = 1, default number of filters from the paper
   *               are used at each layer.
   * @param depthMultiplier Depth multiplier for depthwise convolution.
   *                        This is called the resolution multiplier in the 
   *                        MobileNet paper.
   * @param includeTop Whether to include the fully-connected layer at 
   *                   the top of the network.
   * @param pooling Optional pooling mode for feature extraction when
   *                include_top is false.
   * @param weights One of 'none', 'imagenet'(pre-training on ImageNet) or path to weights.
   */
   MobileNetV1(
               const std::tuple<size_t, size_t, size_t> inputShape,
               const size_t numClasses = 1000,
               const double alpha = 1.0,
               const int depthMultiplier = 1,
               const bool includeTop = true,
               const std::string &pooling = "max",
               const std::string &weights = "None");
   /**
    *
    * Represents Depth-Wise Convolution Block as defined in MobileNet paper.
    * 
    * A depthwise convolution block consists of a depthwise conv,
    * batch normalization, relu, pointwise convolution,
    * batch normalization and relu activation.
    * 
    * @param inChannels The number of input maps.
    * @param outChannels The number of output maps.
    * @param kernelWidth Width of the filter/kernel.
    * @param kernelHeight Height of the filter/kernel.
    * @param strideWidth Stride of filter application in the x direction.
    * @param strideHeight Stride of filter application in the y direction.
    * @param padW Padding width of the input.
    * @param padH Padding height of the input.
    * @param inputWidth The width of the input data.
    * @param inputHeight The height of the input data.
    */ 
    Sequential<> * DepthWiseConvolutionBlock(
                                            const size_t inChannels,
                                            const size_t outChannels,
                                            const size_t kernelWidth,
                                            const size_t kernelHeight,
                                            const size_t strideWidth,
                                            const size_t strideHeight,
                                            const size_t padW,
                                            const size_t padH,
                                            const size_t inputWidth,
                                            const size_t inputHeight);

    /**
    *
    * Represents Depth-Wise Convolution Block as defined in MobileNet paper.
    * 
    * A depthwise convolution block consists of a depthwise conv,
    * batch normalization, relu6, pointwise convolution,
    * batch normalization and relu6 activation.
    * 
    * @param inChannels The number of input maps.
    * @param outChannels The number of output maps.
    * @param kernelWidth Width of the filter/kernel.
    * @param kernelHeight Height of the filter/kernel.
    * @param strideWidth Stride of filter application in the x direction.
    * @param strideHeight Stride of filter application in the y direction.
    * @param padW Padding width of the input.
    * @param padH Padding height of the input.
    * @param inputWidth The width of the input data.
    * @param inputHeight The height of the input data.
    */ 
    Sequential<> * ConvolutionBlock(
                                    const size_t inChannels,
                                    const size_t outChannels,
                                    const size_t kernelWidth,
                                    const size_t kernelHeight,
                                    const size_t strideWidth,
                                    const size_t strideHeight,                                            const size_t padW,
                                    const size_t padH,
                                    const size_t inputWidth,
                                    const size_t inputHeight);
}