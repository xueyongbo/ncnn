// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "roiavgpooling.h"

#include "layer_type.h"

#include <float.h>

namespace ncnn {

RoiAvgPooling::RoiAvgPooling()
{
}

int RoiAvgPooling::load_param(const ParamDict& pd)
{
    isintegral = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    return 0;
}

static Mat _integral(const Mat& src)
{
    Mat integral;
    copy_make_border(src, integral, 1, 0, 1, 0, ncnn::BORDER_CONSTANT, 0.0);
    float* integralPtr = integral;

    // 计算积分图
    int csetp = integral.w * src.elempack;
    for (int i = 0; i < src.h; ++i)
    {
        float* lt = integralPtr + i * csetp;
        float* lb = lt + csetp;
        for (int j = 0; j < src.w; ++j)
        {
            float* rt = lt + 1;
            float* rb = lb + 1;

            *rb = *rt + *lb + *rb - *lt;

            lt = rt;
            lb = rb;
        }
    }

    return integral;
}

int RoiAvgPooling::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.size() == 2)
    {
        Mat pos = bottom_blobs[0];
        Mat in = bottom_blobs[1];

        int w = in.w;
        int h = in.h;
        int channels = in.c;
        size_t in_elemsize = in.elemsize;
        size_t in_elempack = in.elempack;

        Mat out;
        int outw = pos.h;
        int outh = 1;
        out.create(outw, outh, channels, in_elemsize, in_elempack);
        float divisor = 1.0 / (kernel_w * kernel_h);

        if (isintegral)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int c = 0; c < channels; ++c)
            {
                Mat integral = _integral(in.channel(c));

                // 计算avg
                float* integralPtr = integral;
                float* outPtr = out.channel(c);
                float* xyPtr = pos;
                for (int i = 0; i < pos.h; ++i)
                {
                    int x = *xyPtr++;
                    int y = *xyPtr++;

                    float* left_top_Ptr = integralPtr + y * integral.w + x;
                    float* right_top_Ptr = left_top_Ptr + kernel_w;
                    float* left_bottom_Ptr = integralPtr + (kernel_h + y) * integral.w + x;
                    float* right_bottom_Ptr = left_bottom_Ptr + kernel_w;
                    *outPtr = divisor * ((*right_bottom_Ptr) + (*left_top_Ptr) - ((*right_top_Ptr) + (*left_bottom_Ptr)));
                    outPtr++;
                }
            }
        }
        else
        {
            const int maxk = kernel_w * kernel_h;

            // kernel offsets
            std::vector<int> _space_ofs(maxk);
            int* space_ofs = &_space_ofs[0];
            {
                int p1 = 0;
                int p2 = 0;
                for (int i = 0; i < kernel_h; i++)
                {
                    for (int j = 0; j < kernel_w; j++)
                    {
                        space_ofs[p1] = p2 + j;
                        p1++;
                    }
                    p2 += w;
                }
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int c = 0; c < channels; ++c)
            {
                float* inPtr = in.channel(c);
                float* outPtr = out.channel(c);
                float* xyPtr = pos;
                for (int i = 0; i < pos.h; ++i)
                {
                    int x = *xyPtr++;
                    int y = *xyPtr++;
                    float sum = 0;
                    float* startPtr = inPtr + y * w + x;
                    for (int k = 0; k < maxk; ++k)
                    {
                        sum += startPtr[_space_ofs[k]];
                    }
                    *outPtr = sum * divisor;
                    outPtr++;
                }
            }
        }
        top_blobs[0] = out;
    }
    else
    {
        Mat in = bottom_blobs[0];

        int w = in.w;
        int h = in.h;
        int channels = in.c;
        size_t in_elemsize = in.elemsize;
        size_t in_elempack = in.elempack;

        Mat out;
        int outw = w - kernel_w + 1;
        int outh = h - kernel_h + 1;
        out.create(outw, outh, channels, in_elemsize, in_elempack);
        float divisor = 1.0 / (kernel_w * kernel_h);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int c = 0; c < channels; ++c)
        {
            Mat integral = _integral(in.channel(c));

            // 计算avg
            float* integralPtr = integral;
            float* outPtr = out.channel(c);
            float* left_top_Ptr = integralPtr;
            float* right_top_Ptr = left_top_Ptr + kernel_w;
            float* left_bottom_Ptr = integralPtr + kernel_h * integral.w;
            float* right_bottom_Ptr = left_bottom_Ptr + kernel_w;
            for (int i = 0; i < outh; ++i)
            {
                for (int j = 0; j < outw; ++j)
                {
                    *outPtr = divisor * ((*right_bottom_Ptr) + (*left_top_Ptr) - ((*right_top_Ptr) + (*left_bottom_Ptr)));
                    left_top_Ptr++;
                    right_top_Ptr++;
                    left_bottom_Ptr++;
                    right_bottom_Ptr++;
                    outPtr++;
                }
                left_top_Ptr += kernel_w;
                right_top_Ptr += kernel_w;
                left_bottom_Ptr += kernel_w;
                right_bottom_Ptr += kernel_w;
            }
        }
        top_blobs[0] = out;
    }
    return 0;
}

} // namespace ncnn
