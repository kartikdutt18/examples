/**
 * @file alexnet_impl.hpp
 * @author Kartik Dutt
 * 
 * Implementation of alex-net using mlpack.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_ALEXNET_IMPL_HPP
#define MODELS_ALEXNET_IMPL_HPP

#include "alexnet.hpp"

AlexNet::AlexNet(const size_t inputChannel,
                 const size_t inputWidth,
                 const size_t inputHeight,
                 const size_t numClasses,
                 const bool includeTop,
                 const std::string &weights):
                 AlexNet(
                  std::tuple<size_t, size_t, size_t>(inputChannel,
                      inputWidth,
                      inputHeight),
                  numClasses,
                  includeTop,
                  weights)
{
  // Nothing to do Here.
}

AlexNet::AlexNet(const std::tuple<size_t, size_t, size_t> inputShape,
                 const size_t numClasses,
                 const bool includeTop,
                 const std::string &weights):
                 inputWidth(std::get<1>(inputShape)),
                 inputHeight(std::get<2>(inputShape)),
                 inputChannel(std::get<0>(inputShape)),
                 numClasses(numClasses),
                 includeTop(includeTop),
                 weights(weights),
                 outputShape(512)
{
  alexNet = new Sequential<>();
  // Add Convlution Block with inputChannels as input maps,
  // output maps = 64, kernel_size = (11, 11) stride = (4, 4)
  // and padding = (2, 2).
  ConvolutionBlock(inputChannel, 64, 11, 11, 4, 4, 2, 2);

  // Add Max-Pooling Layer with kernel size = (3, 3) and stride = (2, 2).
  PoolingBlock(3, 3, 2, 2);

  // Add Convlution Block with inputChannels = 64,
  // output maps = 192, kernel_size = (5, 5) stride = (1, 1)
  // and padding = (2, 2).
  ConvolutionBlock(64, 192, 5, 5, 1, 1, 2, 2);

  // Add Max-Pooling Layer with kernel size = (3, 3) and stride = (2, 2).
  PoolingBlock(3, 3, 2, 2);

  // Add Convlution Block with input maps = 192,
  // output maps = 384, kernel_size = (3, 3) stride = (1, 1)
  // and padding = (1, 1).
  ConvolutionBlock(192, 384, 3, 3, 1, 1, 1, 1);

  // Add Convlution Block with input maps = 384,
  // output maps = 256, kernel_size = (3, 3) stride = (1, 1)
  // and padding = (1, 1).
  ConvolutionBlock(384, 256, 3, 3, 1, 1, 1, 1);

  // Add Convlution Block with input maps = 256,
  // output maps = 256, kernel_size = (3, 3) stride = (1, 1)
  // and padding = (1, 1).
  ConvolutionBlock(256, 256, 3, 3, 1, 1, 1, 1);

  // Add Max-Pooling Layer with kernel size = (3, 3) and stride = (2, 2).
  PoolingBlock(3, 3, 2, 2);

  if (includeTop)
  {
    AdaptivePoolingBlock(6, 6);
    alexNet->Add<Dropout<>>(0.2);
    alexNet->Add<Linear<>>(256 * 6 * 6, 4096);
    alexNet->Add<ReLULayer<>>();
    alexNet->Add<Dropout<>>(0.2);
    alexNet->Add<Linear<>>(4096, 4096);
    alexNet->Add<ReLULayer<>>();
    alexNet->Add<Linear<>>(4096, numClasses);
    alexNet->Add<LogSoftMax<>>();
  }
  else
  {
    alexNet->Add<MaxPooling<>>(inputWidth, inputHeight, 1, 1, true);
    outputShape = 512;
  }
}

#endif
