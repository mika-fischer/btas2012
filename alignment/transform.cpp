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

#include "transform.hpp"
#include "utils.hpp"

using cv::Mat;
using cv::Mat_;
using cv::Point2d;
using namespace std;

Mat estimateSimilarityTransform(const vector<Point2d>& src_points,
        const vector<Point2d>& dst_points)
{
    ASSERT_THROW(src_points.size() >= 2 && dst_points.size() >= 2);
    ASSERT_THROW(src_points.size() == dst_points.size());

    // Solve for T = (cos(phi), sin(phi), t_x, t_y) using least squares

    double A[16], B[4], T[4];
    Mat_<double> AA(4, 4, A);
    Mat_<double> BB(4, 1, B);
    Mat_<double> TT(4, 1, T);

    memset(A, 0, sizeof(A));
    memset(B, 0, sizeof(B));

    for(size_t i=0; i<src_points.size(); ++i)
    {
        const cv::Point2d& a = src_points[i];
        const cv::Point2d& b = dst_points[i];

        A[ 0] += a.x * a.x + a.y * a.y;
        A[ 1] += 0;
        A[ 2] += a.x;
        A[ 3] += a.y;

        A[ 4] += 0;
        A[ 5] += a.x * a.x + a.y * a.y;
        A[ 6] += -a.y;
        A[ 7] += a.x;

        A[ 8] += a.x;
        A[ 9] += -a.y;
        A[10] += 1;
        A[11] += 0;

        A[12] += a.y;
        A[13] += a.x;
        A[14] += 0;
        A[15] += 1;

        B[ 0] += a.x * b.x + a.y * b.y;
        B[ 1] += a.x * b.y - a.y * b.x;
        B[ 2] += b.x;
        B[ 3] += b.y;
    }

    cv::solve(AA, BB, TT, cv::DECOMP_SVD);

    return (Mat_<double>(2, 3) <<
            T[0], -T[1], T[2],
            T[1],  T[0], T[3]);
}

cv::Point2d transformPoint(const cv::Point2d& point, const cv::Mat& transform)
{
    double m11 = get(transform, 0, 0);
    double m12 = get(transform, 0, 1);
    double m21 = get(transform, 1, 0);
    double m22 = get(transform, 1, 1);
    double t1  = get(transform, 0, 2);
    double t2  = get(transform, 1, 2);

    return cv::Point2d(
            m11 * point.x + m12 * point.y + t1,
            m21 * point.x + m22 * point.y + t2);
}
