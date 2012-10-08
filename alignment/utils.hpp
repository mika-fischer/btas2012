// Copyright (c) 2012, Mika Fischer <mika.fischer@kit.edu>
//                     Computer Vision for HCI Lab
//                     Karlsruhe Institute of Technology
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice, this list
//       of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright notice, this
//       list of conditions and the following disclaimer in the documentation and/or other
//       materials provided with the distribution.
//     * Neither the name of the Karlsruhe Institute of Technology nor the names of its
//       contributors may be used to endorse or promote products derived from this
//       software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
// THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <opencv2/core/core.hpp>
#include <stdexcept>

#define ASSERT_THROW(cond) \
{ \
    if (!(cond)) \
        throw std::runtime_error("Assertion '" #cond "' failed."); \
}

inline double get(const cv::Mat& m, int r, int c)
{
    ASSERT_THROW(!m.empty());
    ASSERT_THROW(m.channels() == 1);
    ASSERT_THROW(r >= 0);
    ASSERT_THROW(r < m.rows);
    ASSERT_THROW(c >= 0);
    ASSERT_THROW(c < m.cols);

    if (!m.empty())
    {
        switch (m.type())
        {
            case CV_8U:
                return m.at<uchar>(r, c);
            case CV_8S:
                return m.at<schar>(r, c);
            case CV_16U:
                return m.at<ushort>(r, c);
            case CV_16S:
                return m.at<short>(r, c);
            case CV_32S:
                return m.at<int>(r, c);
            case CV_32F:
                return m.at<float>(r, c);
            case CV_64F:
                return m.at<double>(r, c);
        }
    }

    throw std::runtime_error("Invalid matrix");
}

