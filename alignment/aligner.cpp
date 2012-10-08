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

#include "aligner.hpp"
#include "transform.hpp"
#include "utils.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <boost/foreach.hpp>

using cv::Point2d;
using boost::optional;
using namespace std;

FourPointAlignerParameters::~FourPointAlignerParameters()
{}

SimpleContinuousAlignerParameters::SimpleContinuousAlignerParameters(
        const cv::Size& size, size_t padding,
        double eye_row, double eye_dist, double mouth_row, double mouth_offset)
: size_(size), padding_(padding), eye_row_(eye_row), eye_dist_(eye_dist),
    mouth_row_(mouth_row), mouth_offset_(mouth_offset)
{}

cv::Size SimpleContinuousAlignerParameters::size(double pan, double tilt)
{
    return cv::Size(size_.width + 2*padding_, size_.height + 2*padding_);
}

void SimpleContinuousAlignerParameters::points(double pan, double tilt,
        const optional<Point2d>& left_eye,
        const optional<Point2d>& right_eye,
        const optional<Point2d>& nose_tip,
        const optional<Point2d>& mouth_center,
        vector<Correspondence>& correspondences)
{
    double rpan         = pan /360.0 * 2*CV_PI;
    double rtilt        = tilt/360.0 * 2*CV_PI;
    double eyedist      = cos(rpan) * eye_dist_;
    double tilt_eyedist = sin(rpan) * eye_dist_;
    double sign         = pan >= 0 ? 1 : -1;

    if (pan >= 45)
    {
        ASSERT_THROW(left_eye);
        Point2d eye((size_.width - 1)/2.0 - sign * 0.5 * eyedist, eye_row_);
        correspondences.push_back(make_pair(eye, *left_eye));
    }
    else if (pan <= -45)
    {
        ASSERT_THROW(right_eye);
        Point2d eye((size_.width - 1)/2.0 - sign * 0.5 * eyedist, eye_row_);
        correspondences.push_back(make_pair(eye, *right_eye));
    }
    else
    {
        ASSERT_THROW(left_eye && right_eye);
        Point2d center    ((size_.width - 1)/2.0, eye_row_);
        Point2d center_img((*right_eye + *left_eye) * 0.5);
        correspondences.push_back(make_pair(center, center_img));
    }

    Point2d mouth((size_.width - 1)/2.0 + sin(rpan) * mouth_offset_, mouth_row_);
    correspondences.push_back(make_pair(mouth, *mouth_center));

    // Account for shortening of eye-mouth distance when face is tilted
    if (tilt != 0)
    {
        // Wanted distance in y direction
        double d_eye_mouth = mouth_row_ - eye_row_;

        // Difference in distance eyerow mouthrow
        // double diff = d_eye_mouth - d_eye_mouth_img_tilted;
        double diff = (1-cos(rtilt)) * d_eye_mouth;

        // Apply the offset to the target points
        correspondences[0].first.y += diff / 2.0;
        correspondences[1].first.y -= diff / 2.0;
    }

    // Add padding
    BOOST_FOREACH (Correspondence& c, correspondences)
    {
        c.first.x += padding_;
        c.first.y += padding_;
    }
}

FourPointAligner::FourPointAligner(FourPointAlignerParameters::Ptr parameters)
: parameters_(parameters)
{}

cv::Mat FourPointAligner::alignFace(const cv::Mat& img,
        const optional<Point2d>& left_eye, const optional<Point2d>& right_eye,
        const optional<Point2d>& nose_tip, const optional<Point2d>& mouth_center,
        double pan, double tilt) const
{
    cv::Size size = parameters_->size(pan, tilt);
    vector<FourPointAlignerParameters::Correspondence> points;
    parameters_->points(pan, tilt, left_eye, right_eye, nose_tip, mouth_center, points);

    vector<Point2d> ref, lm;
    BOOST_FOREACH (const FourPointAlignerParameters::Correspondence& c, points)
    {
        ref.push_back(c.first);
        lm.push_back(c.second);
    }

    cv::Mat transform = estimateSimilarityTransform(ref, lm);

    cv::Mat result(size, img.type());

    cv::warpAffine(img, result, transform, result.size(), cv::INTER_LINEAR|cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);

    return result;
}

cv::Mat FourPointAligner::alignFacePoints(const cv::Mat& img,
        optional<Point2d>& left_eye, optional<Point2d>& right_eye,
        optional<Point2d>& nose_tip, optional<Point2d>& mouth_center,
        double pan, double tilt) const
{
    cv::Size size = parameters_->size(pan, tilt);
    vector<FourPointAlignerParameters::Correspondence> points;
    parameters_->points(pan, tilt, left_eye, right_eye, nose_tip, mouth_center, points);

    vector<cv::Point2d> ref, lm;
    BOOST_FOREACH (const FourPointAlignerParameters::Correspondence& c, points)
    {
        ref.push_back(c.first);
        lm.push_back(c.second);
    }

    cv::Mat transform = estimateSimilarityTransform(ref, lm);

    cv::Mat inv_transform;
    cv::invertAffineTransform(transform, inv_transform);

    if (left_eye)
        left_eye = transformPoint(*left_eye, inv_transform);
    if (right_eye)
        right_eye = transformPoint(*right_eye, inv_transform);
    if (nose_tip)
        nose_tip = transformPoint(*nose_tip, inv_transform);
    if (mouth_center)
        mouth_center = transformPoint(*mouth_center, inv_transform);

    cv::Mat result(size, img.type());

    cv::warpAffine(img, result, transform, result.size(), cv::INTER_LINEAR|cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);

    return result;
}
