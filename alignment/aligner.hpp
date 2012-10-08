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
#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>
#include <utility>

class FourPointAlignerParameters
{
    public:
        typedef boost::shared_ptr<FourPointAlignerParameters> Ptr;
        virtual ~FourPointAlignerParameters();

        typedef std::pair<cv::Point2d, cv::Point2d> Correspondence;

        virtual cv::Size size(double pan, double tilt) = 0;
        virtual void points(double pan, double tilt,
                const boost::optional<cv::Point2d>& left_eye,
                const boost::optional<cv::Point2d>& right_eye,
                const boost::optional<cv::Point2d>& nose_tip,
                const boost::optional<cv::Point2d>& mouth_center,
                std::vector<Correspondence>& correspondences) = 0;
};

class SimpleContinuousAlignerParameters : public FourPointAlignerParameters
{
    public:
        typedef boost::shared_ptr<SimpleContinuousAlignerParameters> Ptr;

        SimpleContinuousAlignerParameters(const cv::Size& size, size_t padding,
                double eye_row, double eye_dist, double mouth_row, double mouth_offset);

        virtual cv::Size size(double pan, double tilt);
        virtual void points(double pan, double tilt,
                const boost::optional<cv::Point2d>& left_eye,
                const boost::optional<cv::Point2d>& right_eye,
                const boost::optional<cv::Point2d>& nose_tip,
                const boost::optional<cv::Point2d>& mouth_center,
                std::vector<Correspondence>& correspondences);

    private:
        cv::Size size_;
        size_t padding_;
        double eye_row_;
        double eye_dist_;
        double mouth_row_;
        double mouth_offset_;
};

class FourPointAligner
{
    public:
        FourPointAligner(FourPointAlignerParameters::Ptr parameters);

        cv::Mat alignFace(const cv::Mat& img,
                const boost::optional<cv::Point2d>& left_eye, const boost::optional<cv::Point2d>& right_eye,
                const boost::optional<cv::Point2d>& nose_tip, const boost::optional<cv::Point2d>& mouth_center,
                double pan, double tilt) const;

        cv::Mat alignFacePoints(const cv::Mat& img,
                boost::optional<cv::Point2d>& left_eye, boost::optional<cv::Point2d>& right_eye,
                boost::optional<cv::Point2d>& nose_tip, boost::optional<cv::Point2d>& mouth_center,
                double pan, double tilt) const;

    private:
        FourPointAlignerParameters::Ptr parameters_;
};
