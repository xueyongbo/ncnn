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

#if __AVX__
#include "avx_activation.h"
#include "avx_usability.h"
#endif

#if __SSE2__
#include <emmintrin.h>
#endif
#if __AVX__
#include <immintrin.h>
#endif

#include "roiavgpooling_x86.h"

#include "layer_type.h"

#include <float.h>

namespace ncnn {

RoiAvgPooling_x86::RoiAvgPooling_x86()
{
    support_packing = true;
}

#if __AVX__
static Mat _integralAVX(const Mat& src)
{
    Mat integral;
    copy_make_border(src, integral, 1, 0, 1, 0, ncnn::BORDER_CONSTANT, 0.0);

    float* integralPtr = integral;

    // 计算积分图
    int csetp = integral.w * 8;

    for (int i = 0; i < src.h; ++i)
    {
        float* lt = integralPtr + i * csetp;
        float* lb = lt + csetp;
        __m256 ltPtr = _mm256_loadu_ps(lt);
        __m256 lbPtr = _mm256_loadu_ps(lb);
        for (int j = 0; j < src.w; ++j)
        {
            lt += 8;
            lb += 8;
            __m256 rtPtr = _mm256_loadu_ps(lt);
            __m256 rbPtr = _mm256_loadu_ps(lb);
            rbPtr = _mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(rtPtr, lbPtr), rbPtr), ltPtr);
            _mm256_storeu_ps(lb, rbPtr);
            ltPtr = rtPtr;
            lbPtr = rbPtr;
        }
    }
    return integral;
}
#endif

int RoiAvgPooling_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if __AVX__
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

        if (in.elempack == 8 && in.elemsize == 32)
        {
            __m256 divisor256 = _mm256_set1_ps(divisor);
            if (isintegral)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int c = 0; c < channels; ++c)
                {
                    ncnn::Mat integral = _integralAVX(in.channel(c));

                    float* integralPtr = integral;
                    float* outPtr = out.channel(c);
                    float* xyPtr = pos;
                    int csetp = integral.w * 8;
                    for (int i = 0; i < pos.h; ++i)
                    {
                        int x = *xyPtr++;
                        int y = *xyPtr++;

                        float* left_top_Ptr = integralPtr + y * csetp + x * 8;
                        float* right_top_Ptr = left_top_Ptr + kernel_w * 8;
                        float* left_bottom_Ptr = integralPtr + (kernel_h + y) * csetp + x * 8;
                        float* right_bottom_Ptr = left_bottom_Ptr + kernel_w * 8;

                        __m256 ltPtr = _mm256_loadu_ps(left_top_Ptr);
                        __m256 lbPtr = _mm256_loadu_ps(left_bottom_Ptr);
                        __m256 rtPtr = _mm256_loadu_ps(right_top_Ptr);
                        __m256 rbPtr = _mm256_loadu_ps(right_bottom_Ptr);

                        __m256 oPtr = _mm256_mul_ps(divisor256, _mm256_sub_ps(_mm256_add_ps(rbPtr, ltPtr), _mm256_add_ps(lbPtr, rtPtr)));
                        _mm256_storeu_ps(outPtr, oPtr);

                        outPtr += 8;
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
                            space_ofs[p1] = p2 + j * 8;
                            p1++;
                        }
                        p2 += w * 8;
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
                        __m256 sum = _mm256_set1_ps(0.0f);
                        float* startPtr = inPtr + (y * w + x) * 8;
                        for (int k = 0; k < maxk; ++k)
                        {
                            __m256 v = _mm256_loadu_ps(startPtr + _space_ofs[k]);
                            sum = _mm256_add_ps(sum, v);
                        }
                        sum = _mm256_mul_ps(divisor256, sum);
                        _mm256_storeu_ps(outPtr, sum);
                        outPtr += 8;
                    }
                }
            }
            top_blobs[0] = out;
            return 0;
        }
        else
        {
            return RoiAvgPooling::forward(bottom_blobs, top_blobs, opt);
        }
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

        if (in.elempack == 8 && in.elemsize == 32)
        {
            __m256 divisor256 = _mm256_set1_ps(divisor);
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int c = 0; c < channels; ++c)
            {
                Mat integral = _integralAVX(in.channel(c));

                // 计算avg
                float* integralPtr = integral;
                float* outPtr = out.channel(c);

                int kstep = kernel_w * 8;
                float* left_top_Ptr = integralPtr;
                float* right_top_Ptr = left_top_Ptr + kstep;
                float* left_bottom_Ptr = integralPtr + kernel_h * integral.w * 8;
                float* right_bottom_Ptr = left_bottom_Ptr + kstep;
                for (int i = 0; i < outh; ++i)
                {
                    for (int j = 0; j < outw; ++j)
                    {
                        __m256 ltPtr = _mm256_loadu_ps(left_top_Ptr);
                        __m256 lbPtr = _mm256_loadu_ps(left_bottom_Ptr);
                        __m256 rtPtr = _mm256_loadu_ps(right_top_Ptr);
                        __m256 rbPtr = _mm256_loadu_ps(right_bottom_Ptr);

                        __m256 oPtr = _mm256_mul_ps(divisor256, _mm256_sub_ps(_mm256_add_ps(rbPtr, ltPtr), _mm256_add_ps(lbPtr, rtPtr)));
                        _mm256_storeu_ps(outPtr, oPtr);

                        left_top_Ptr += 8;
                        right_top_Ptr += 8;
                        left_bottom_Ptr += 8;
                        right_bottom_Ptr += 8;
                        outPtr += 8;
                    }
                    left_top_Ptr += kstep;
                    right_top_Ptr += kstep;
                    left_bottom_Ptr += kstep;
                    right_bottom_Ptr += kstep;
                }
            }
            top_blobs[0] = out;
            return 0;
        }
        else
        {
            return RoiAvgPooling::forward(bottom_blobs, top_blobs, opt);
        }
    }
    return 0;
#else
    return RoiAvgPooling::forward(bottom_blobs, top_blobs, opt);
#endif __AVX__
}

} // namespace ncnn
