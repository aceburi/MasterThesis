#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11_json/pybind11_json.hpp>

#include <gurobi_c++.h>

#include "Encoder.hpp"
#include "quant/SequentialCreator.hpp"
#include "codegen2/test.hpp"
#include "analytics/Analyze.hpp"
#include "StorageAdaptor.hpp"

PYBIND11_MODULE(nn2logic, m) {
    m.def("QTreeAnalyze", &analytics::analyze, py::arg("filename"), R"pbdocs(
        Provides statistics about the occurence of possible classes of leaves.
    
        :param filename: input file path. Type is parsed by :class:`nn2logic.JsonLoader`.
        :return: A JSON dictionary with the following keys:
            
            | **nopLeaf**: number of possible classes counted by leaf. The values are the occurences of each class.
            | **nopVisit**: number of possible classes counted by samples. The values are the occurences of each class.
            | **totalLeaves**: the total number of leaves.
            | **totalVisits**: the total number of samples.

    )pbdocs");

    pybind11::class_<QTree::FixedPoint>(m, "FixedPoint", R"pbdocs(
        Fixed Point number representation.
        )pbdocs")
        .def(pybind11::init<size_t,size_t>(), py::arg("value"), py::arg("shift"), R"pbdocs(
            :param value: raw integer value of the fixed point number
            :param shift: amount to shift in fixed point multiplication.
        )pbdocs");

    m.def("QHybridCreator", &codegen2::loadFromJson, py::arg("filename"), R"pbdocs(
        Create hybrid code. Uses information stored in JSON file.

        :param filename: input file path. Type is parsed by :class:`nn2logic.JsonLoader`.
    )pbdocs");

    m.def("JsonConverter", &StorageAdaptor::convert, 
        py::arg("src"), py::arg("dst"), R"pbdocs(
        Convert between arbitrary JSON (\*.json) and UBJSON (\*.ubj) files.
        File formats are deduced by extension.

        :param src: input file path
        :param dst: output file path
    )pbdocs");

    // Encoder
    pybind11::class_<Encoder>(m, "InputEncoder", R"pbdocs(
            Handles encoding of inputs. Makes sure the properties of inputs are correctly modelled as MIP for later use in tree exploration.
        )pbdocs")
        .def(pybind11::init())
        .def("registerScalar", &Encoder::registerScalar, py::arg("name"), py::arg("upperLimit"), R"pbdocs(
            Register a scalar Value.

            :param name: unique name for the scalar
            :param upperLimit: maximum value of this scalar
        )pbdocs")
        .def("registerBinary", &Encoder::registerBinary, py::arg("name"), py::arg("upperLimit"), R"pbdocs(
            Register a Binary Value. if upperLimit is unequal to 1, the value will be inserted with a scaling factor

            :param name: unique name for the binary
            :param upperLimit: scaling factor
        )pbdocs")
        .def("registerInt", &Encoder::registerInt, py::arg("name"), py::arg("upperLimit"), R"pbdocs(
            Register an integer Value.

            :param name: unique name for the integer
            :param upperLimit: maximum value of this integer
        )pbdocs")
        .def("update", &Encoder::update, R"pbdocs(
            Submit changes to MIP model and tune it accordingly.
        )pbdocs")
        .def("markBinariesOneHot", &Encoder::markBinariesOneHot, py::arg("names"), py::arg("strict"), R"pbdocs(
            Mark a group of binary variables as one-hot encoded. Meaning no more than one of them can be true simultaneously.

            :param names: names of the binary values
            :param strict: if true at least one has to be true, otherwise all can be false at the same time as well. Defaults to false.
        )pbdocs");

    // Scaler
    pybind11::class_<QTree::Scaler<int>>(m, "QScales", "Scaling factors for the channels of a layer")
        .def(pybind11::init<std::vector<QTree::FixedPoint>&,int>(), py::arg("scales"), py::arg("upperLimit"))
        .def_readwrite("scales", &QTree::Scaler<int>::scales, "Fixed point scaling values per channel/neuron")
        .def_readwrite("upperLimit", &QTree::Scaler<int>::upperLimit, "upper limit for saturation")
        .def_readwrite("lowerLimit", &QTree::Scaler<int>::lowerLimit, "lower limit for saturation");

    // Layer Descriptor
    pybind11::class_<QTree::Layer<int>>(m, "QLayer")
        .def(pybind11::init<const Eigen::Ref<const Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic>>,const Eigen::Ref<const Eigen::Matrix<int,Eigen::Dynamic,1>>,bool,const QTree::Scaler<int>&>(),
            py::arg("weight"), py::arg("bias"), py::arg("relu"), py::arg("requant"), R"pbdocs(
                Representation of a quantized layer.
            )pbdocs")
        .def_readwrite("weight", &QTree::Layer<int>::weight, "weight matrix")
        .def_readwrite("bias", &QTree::Layer<int>::bias, "bias vector")
        .def_readwrite("relu", &QTree::Layer<int>::relu, "whether the layer uses a ReLU activation function")
        .def_readwrite("requant", &QTree::Layer<int>::requant, "scaling factors of the channels");

    // Storage Adapter
    pybind11::class_<StorageAdaptor>(m, "StorageAdaptor", R"pbdocs(
            Storage adapter to save quantized networks, their tree representation and dataset.
            Currently supports JSON and UBJSON.
        )pbdocs")
        .def(pybind11::init<const std::string&>())
        .def(pybind11::init<const std::vector<QTree::Layer<int>>&,const std::vector<StorageAdaptor::Samples_t>&>())
        .def("setDataset", &StorageAdaptor::setDataset, py::arg("samples"), R"pbdocs(
            Set data and targets.

            :param samples: quantized samples and targets
        )pbdocs")
        .def("save", &StorageAdaptor::save, py::arg("dst"), R"pbdocs(
            Save to disk. File format is determined by extension of `dst`.

            :param dst: file to save into.
        )pbdocs");

    // Function to write dataset into header
    m.def("datasetHeader", &StorageAdaptor::createHeaderFromDataset);

    // Quantized Sequential Tree Creator Class
    pybind11::class_<QTree::SequentialCreator<int,int>>(m, "QTreeBuilder", R"pbdocs(
        Explore equivalent decision tree of provided quantized networks in combination with a dataset.
        )pbdocs")
        .def(pybind11::init<const StorageAdaptor&, Encoder*>(),
            py::arg("storage"), py::arg("encoder"), R"pbdocs(
            :param storage: quantized network and dataset for tree exploration
            :param encoder: :class:`InputEncoder` that contains the MIP model of the input data
            )pbdocs")
        .def("getPaths", &QTree::SequentialCreator<int,int>::getPaths, R"pbdocs(
            :return: list of paths of the equivalent decision tree
            )pbdocs")
        .def("store", &QTree::SequentialCreator<int,int>::store, R"pbdocs(
            :return: a :class:`JsonLoader` representation of the network and equivalent decision tree. does not contain dataset at the moment.
            )pbdocs");


    // handle Gurobi exceptions
    pybind11::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const GRBException &e) {
            throw std::runtime_error("Gurobi Error code = " + std::to_string(e.getErrorCode()) + "\n" + e.getMessage());
        }
    });
}