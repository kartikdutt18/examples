/**
 * @file alexnet.hpp
 * @author Kartik Dutt
 * 
 * Implementation of alex-net using mlpack.
 * 
 * For more information, see the following paper.
 * 
 * @code
 * @misc{
 *   author = {Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton},
 *   title = {ImageNet Classification with Deep Convolutional Neural Networks},
 *   year = {2012}
 * }
 * @endcode
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_ALEXNET_HPP
#define MODELS_ALEXNET_HPP

// Include all required libraries.
#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

using namespace mlpack;
using namespace mlpack::ann;

class AlexNet
{
 public:
  //! Create the AlexNet object.
  AlexNet();

  /**
   * AlexNet constructor intializes input shape, number of classes
   * and width multiplier.
   *
   * @param inputChannels Number of input channels of the input image.
   * @param inputWidth Width of the input image.
   * @param inputHeight Height of the input image.
   * @param numClasses Optional number of classes to classify images into,
   *                   only to be specified if includeTop is  true.
   * @param includeTop whether to include the fully-connected layer at 
   *        the top of the network.
   * @param weights One of 'none', 'imagenet'(pre-training on ImageNet) or path to weights.
   */
  AlexNet(const size_t inputChannel,
          const size_t inputWidth,
          const size_t inputHeight,
          const size_t numClasses = 1000,
          const bool includeTop = true,
          const std::string &weights = "None");

  /**
   * AlexNet constructor intializes input shape, number of classes
   * and width multiplier.
   *  
   * @param inputShape A three-valued tuple indicating input shape.
   *                   First value is number of Channels (Channels-First).
   *                   Second value is input height.
   *                   Third value is input width..
   * @param numClasses Optional number of classes to classify images into,
   *                   only to be specified if includeTop is  true.
   * @param includeTop whether to include the fully-connected layer at 
   *        the top of the network.
   * @param weights One of 'none', 'imagenet'(pre-training on ImageNet) or path to weights.
   */
  AlexNet(const std::tuple<size_t, size_t, size_t> inputShape,
          const size_t numClasses = 1000,
          const bool includeTop = true,
          const std::string &weights = "None");

  // Custom Destructor.
  ~AlexNet();

  /** 
   * Defines Model Architecture.
   * 
   * @return Sequential Pointer to the sequential AlexNet model.
   */
  Sequential<>* CompileModel();

  /**
   * Load model from a path.
   * 
   *
   * @param filePath Path to load the model from.
   * @return Sequential Pointer to a sequential model.
   */
  Sequential<>* LoadModel(const std::string &filePath);

  /**
   * Save model to a location.
   *
   * @param filePath Path to save the model to.
   */
  void SaveModel(const std::string &filePath);

  /**
   * Return output shape of model.
   * @returns outputShape of size_t type.
   */
  size_t OutputShape() { return outputShape; };

  /**
   * Returns compiled version of model.
   * If called without compiling would result in empty Sequetial
   * Pointer.
   * 
   * @return Sequential Pointer to a sequential model.
   */
  Sequential<>* GetModel() { return alexNet; };

 private:
  /**
   * Returns AdaptivePooling Block.
   * 
   * @param outputlWidth Width of the output.
   * @param outputHeight Height of the output.
   */
  Sequential<>* AdaptivePoolingBlock(const size_t outputWidth,
                                     const size_t outputHeight)
  {
    Sequential<>* poolingBlock;
    const size_t strideWidth = std::floor(inputWidth / outputWidth);
    const size_t strideHeight = std::floor(inputHeight / outputHeight);

    const size_t kernelWidth = inputWidth - (outputWidth - 1) * strideWidth;
    const size_t kernelHeight = inputHeight - (outputHeight - 1) * strideHeight;
    poolingBlock->Add<MaxPooling<>>(kernelWidth, kernelHeight,
        strideWidth, strideHeight);
    // Update inputWidth and inputHeight.
    inputWidth = outputWidth;
    inputHeight = outputHeight;
    return poolingBlock;
  }

  /**
   * Returns Convolution Block.
   * 
   * @param inSize Number of input maps.
   * @param outSize Number of output maps.
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param padW Padding width of the input.
   * @param padH Padding height of the input.
   */
  Sequential<>* ConvolutionBlock(const size_t inSize,
                                 const size_t outSize,
                                 const size_t kernelWidth,
                                 const size_t kernelHeight,
                                 const size_t strideWidth = 1,
                                 const size_t strideHeight = 1,
                                 const size_t padW = 0,
                                 const size_t padH = 0)
  {
    Sequential<>* convolutionBlock;
    convolutionBlock->Add<Convolution<> >(inSize, outSize, kernelWidth,
        kernelHeight, strideWidth, strideHeight, padW, padH, inputWidth,
        inputHeight);
      convolutionBlock->Add<ReLULayer<> >();
    

    // Update inputWidth and input Height.
    inputWidth = ConvOutSize(inputWidth, kernelWidth, strideWidth, padW);
    inputHeight = ConvOutSize(inputHeight, kernelHeight, strideHeight, padH);
    return convolutionBlock;
  }

  /**
   * Returns Pooling Block.
   * 
   * @param inSize Number of input maps.
   * @param outSize Number of output maps.
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   */
  Sequential<>* PoolingBlock(const size_t kernelWidth,
                             const size_t kernelHeight,
                             const size_t strideWidth = 1,
                             const size_t strideHeight = 1)
  {
    Sequential<>* poolingBlock;
    poolingBlock->Add<MaxPooling<>>(kernelWidth, kernelHeight,
        strideWidth, strideHeight);
    // Update inputWidth and inputHeight.
    inputWidth = PoolOutSize(inputWidth, kernelWidth, strideWidth);
    inputHeight = PoolOutSize(inputHeight, kernelHeight, strideHeight);
    return poolingBlock;
  }

  /**
   * Return the convolution output size.
   *
   * @param size The size of the input (row or column).
   * @param k The size of the filter (width or height).
   * @param s The stride size (x or y direction).
   * @param pSideOne The size of the padding (width or height) on one side.
   * @param pSideTwo The size of the padding (width or height) on another side.
   * @return The convolution output size.
   */
      size_t ConvOutSize(const size_t size,
                         const size_t k,
                         const size_t s,
                         const size_t padding)
  {
    return std::floor(size + 2 * padding - k) / s + 1;
  }

  /*
   * Return the convolution output size.
   *
   * @param size The size of the input (row or column).
   * @param k The size of the filter (width or height).
   * @param s The stride size (x or y direction).
   * @return The convolution output size.
   */
  size_t PoolOutSize(const size_t size,
                     const size_t k,
                     const size_t s)
  {
    return std::floor(size - 1) / s + 1;
  }
  //! Locally stored AlexNet Model.
  Sequential<>* alexNet;

  //! Locally stored width of the image.
  size_t inputWidth;

  //! Locally stored height of the image.
  size_t inputHeight;

  //! Locally stored number of channels in the image.
  size_t inputChannel;

  //! Locally stored number of output classes.
  size_t numClasses;

  //! Locally stored include the final dense layer.
   bool includeTop;

  //! Locally stored type of pre-trained weights.
  std::string weights;

  //! Locally stored output shape of the AlexNet
  size_t outputShape;
};

#include "alexnet_impl.hpp"

#endif