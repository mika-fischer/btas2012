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
#include "utils.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <iostream>

using boost::optional;
using cv::Point2d;
namespace fs = boost::filesystem;
namespace po = boost::program_options;
namespace pt = boost::property_tree;
using namespace std;

void usage(const char* argv0, const po::options_description& options)
{
    cout << "Usage: " << fs::basename(argv0) << " [options]\n\n" << options << endl;
}

void align_images(const fs::path& label_file, const fs::path& image_base, const fs::path& aligned_dir,
        const FourPointAligner& aligner)
{
    pt::ptree labels;
    fs::ifstream in(label_file, ios::binary);
    if (!in)
        throw runtime_error("Error opening label file " + label_file.string());
    pt::json_parser::read_json(in, labels);

    fs::path new_labels = aligned_dir / "labels.txt";
    fs::ofstream out(new_labels, ios::binary);
    if (!out)
        throw runtime_error("Error writing label file " + new_labels.string());

    out << "# filename lecx lecy recx recy ntx nty mcx mcy" << endl;

    BOOST_FOREACH (const pt::ptree::value_type& node, labels)
    {
        fs::path image_filename = image_base / node.second.get<fs::path>("filename");
        fs::path aligned_filename = aligned_dir / image_filename.filename();
        cv::Mat img = cv::imread(image_filename.string(), cv::IMREAD_GRAYSCALE);
        const pt::ptree& annotations = node.second.get_child("annotations");
        ASSERT_THROW(annotations.size() == 1);
        BOOST_FOREACH (const pt::ptree::value_type& ann_node, annotations)
        {
            const pt::ptree& face = ann_node.second;
            ASSERT_THROW(face.get<string>("class") == "Face");

            Point2d lec, rec, nt, mc;
            double pan, tilt;
            lec.x = face.get<double>("lecx");
            lec.y = face.get<double>("lecy");
            rec.x = face.get<double>("recx");
            rec.y = face.get<double>("recy");
            nt.x  = face.get<double>("ntx");
            nt.y  = face.get<double>("nty");
            mc.x  = face.get<double>("mcx");
            mc.y  = face.get<double>("mcy");
            pan   = face.get<double>("pan");
            tilt  = face.get<double>("tilt");

            optional<Point2d> olec = lec;
            optional<Point2d> orec = rec;
            optional<Point2d> ont  = nt;
            optional<Point2d> omc  = mc;

            cv::Mat aligned = aligner.alignFacePoints(img, olec, orec, ont, omc, pan, tilt);
            cv::imwrite(aligned_filename.string(), aligned);

            out << aligned_filename.filename().string()
                    << " " << olec->x << " " << olec->y
                    << " " << orec->x << " " << orec->y
                    << " " << ont->x  << " " << ont->y
                    << " " << omc->x  << " " << omc->y
                    << endl;
        }
    }
}

int main(int argc, char* argv[])
{
    cv::Size size;
    size_t padding;
    double eye_row, eye_dist, mouth_row, mouth_offset;

    string label_file, image_base, aligned_dir;

    po::options_description options("Options");
    options.add_options()
        ("help,h",
                "Show usage information")

        ("label-file,l", po::value<string>(&label_file)->required(),
                "Location of the label file (in JSON format).")
        ("image-base,i", po::value<string>(&image_base)->required(),
                "Directory to use as base dir for relative paths in label file.")
        ("aligned-dir,a", po::value<string>(&aligned_dir)->required(),
                "Directory in which to store the aligned images. Existing files will be overwritten!")

        ("width", po::value<int>(&size.width)->default_value(104),
                "Width of aligned images in pixels (without padding).")
        ("height", po::value<int>(&size.height)->default_value(128),
                "Height of aligned images in pixels (without padding).")
        ("padding", po::value<size_t>(&padding)->default_value(0),
                "Padding of aligned images in pixels. Padding is applied on all sides of the images.")
        ("eye-row", po::value<double>(&eye_row)->default_value(42.0),
                "Y-coordinate in aligned image where the center between the eyes should lie.")
        ("eye-dist", po::value<double>(&eye_dist)->default_value(62.0),
                "Average eye distance in pixels in the aligned images. This is used to compute the "
                "location of the center between the eyes, when only one eye is visible.")
        ("mouth-row", po::value<double>(&mouth_row)->default_value(106.0),
                "Y-coordinate in aligned image where the mouth center should lie.")
        ("mouth-offset", po::value<double>(&mouth_offset)->default_value(20.0),
                "Average offset of the mouth center in x-direction from the position below the center "
                "between the eyes. This is used to keep aligned images upright in non-frontal poses.")
        ;

    try
    {
        // Parse command line options
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(options).run(), vm);

        if (vm.count("help"))
        {
            cout << "BTAS2012 Aligner\n\n"
                    "Author: Mika Fischer <mika.fischer@kit.edu>\n"
                    "License: GPL\n\n";
            usage(argv[0], options);
            return EXIT_SUCCESS;
        }
        po::notify(vm);

        // Create/check aligned image directory
        if (!fs::exists(aligned_dir))
            fs::create_directories(aligned_dir);
        if (!fs::is_directory(aligned_dir))
            throw runtime_error("Aligned image directory " + aligned_dir + " is not a directory");

        // Set up aligner
        SimpleContinuousAlignerParameters::Ptr aligner_parameters =
                boost::make_shared<SimpleContinuousAlignerParameters>(
                        size, padding, eye_row, eye_dist, mouth_row, mouth_offset);
        FourPointAligner aligner(aligner_parameters);

        // Align images
        align_images(label_file, image_base, aligned_dir, aligner);
    }
    catch (po::error& e)
    {
        cout << "ERROR: " << e.what() << "\n\n";
        usage(argv[0], options);
        return EXIT_FAILURE;
    }
    catch (std::exception& e)
    {
        cout << "ERROR: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    catch (...)
    {
        cout << "ERROR: Unknown error" << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
