/*
 * delays_2d.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 04/08/2020
 *
 * Information: Testing STDP on CUBA_LIF
 */

#include <iostream>
#include "blaze/Blaze.h"
#include "tbb/parallel_for.h"

#include "../source/core.hpp"
#include "../source/neurons/parrot.hpp"
#include "../source/neurons/cuba_lif.hpp"
#include "../source/neurons/decision_making.hpp"
#include "../source/neurons/regression.hpp"
#include "../source/addons/analysis.hpp"
#include "../source/addons/weight_maps.hpp"
#include "../source/learning_rules/stdp.hpp"

int main(int argc, char** argv) {
    int trials = 1;

    // nmnist parameters
    std::string training_path        = "/Users/omaroubari/Datasets/es_N-MNIST/Train";
    std::string test_path            = "/Users/omaroubari/Datasets/es_N-MNIST/Test";
    std::string tensor_base_name     = "nmnist";
    std::vector<std::string> classes = {"5","6","9"};
    int percentage_data              = 10;
    int width                        = 28;
    int height                       = 28;
    int origin                       = 0;
    int number_of_sublayers          = 4;
    int kernel_size                  = 7;
    int stride                       = 1;
    int regression_size              = 1000;
    uint64_t t_max                   = 100000;
    int polarities                   = 1;
    bool logistic_regression         = true;
    bool seed                        = false;
    
    // learning parameters
    float A_plus = 1;
    float A_minus = 0.4;
    float Tau_plus = 20;
    float Tau_minus = 40;
    ;
    if (trials == 1) {
        // initialisation
        hummus::Network network(seed);
        hummus::DataParser parser(seed);

        // generating training database
        auto training_dataset = parser.load_data(training_path, percentage_data, classes);
        int logistic_start = static_cast<int>(training_dataset.files.size()) - regression_size;

        // generating test database
        auto test_dataset = parser.load_data(test_path, percentage_data, classes);

        // learning rule
        auto& stdp = network.make_addon<hummus::STDP>(A_plus, A_minus, Tau_plus, Tau_minus);
        
        // creating layers
        auto pixel_grid = network.make_grid<hummus::Parrot>(width, height, 1, {}, 0, 20);
        auto conv1 = network.make_grid<hummus::CUBA_LIF>(pixel_grid, number_of_sublayers, kernel_size, stride, {&stdp},
                                                         3, // refractory period
                                                         200, // capacitance
                                                         10, // G leak
                                                         true, // WTA
                                                         false, // threshold homeostasis
                                                         false, // burst
                                                         20, // trace tau
                                                         20, // homeostasis tau
                                                         0.1); // homeostasis beta
        
//        auto pool1 = network.make_subsampled_grid<hummus::Parrot>(conv1, {}, 0);
        
//        std::cout << conv1.sublayers[0].neurons.size() << " " << pool1.sublayers[0].neurons.size() << std::endl;
        
        // creating classifier
        hummus::layer classifier;
        if (logistic_regression) {
            classifier = network.make_logistic_regression<hummus::Regression>(training_dataset, test_dataset, 0.1, 0, 0, 70, 128, 10, logistic_start, hummus::optimiser::SGD, tensor_base_name, 0, {});
        } else {
            classifier = network.make_decision<hummus::Decision_Making>(training_dataset, test_dataset, 10, 60, 0, {});
        }

        // connecting the input and output layer with memristive synapses. conductances initialised with a uniform distribution between G_min and G_max
        network.convolution<hummus::Square>(pixel_grid, conv1, 1, hummus::Uniform(0, 1, 0, 0, false), 100);
//        network.pooling<hummus::Square>(conv1, pool1, 1, hummus::Normal(0.25), 100);
        
        std::cout << "number of neurons: " << conv1.neurons.size() << std::endl;
        std::cout << "number of synapses per neuron: " << network.get_neurons()[conv1.neurons[0]]->get_dendritic_tree().size() << std::endl;
        
        // verbose level
        network.verbosity(0);

        // initialise add-ons
        auto& results = network.make_addon<hummus::Analysis>(test_dataset.labels, tensor_base_name+"labels.txt");

        // run the network
        network.run_es_database(training_dataset.files, test_dataset.files, t_max, 0, polarities, width-1+origin, origin, height-1+origin, origin);

        // measuring classification accuracy
        results.accuracy();
        
    } else if (trials > 1) {
        blaze::DynamicVector<double> mean_accuracy(trials);
        tbb::parallel_for(static_cast<size_t>(0), static_cast<size_t>(trials), [&](size_t i) {
            // initialisation
            hummus::Network network(seed);
            hummus::DataParser parser(seed);

            // generating training database
            auto training_dataset = parser.load_data(training_path, percentage_data, classes);
            int logistic_start = static_cast<int>(training_dataset.files.size()) - regression_size;

            // generating test database
            auto test_dataset = parser.load_data(test_path, percentage_data, classes);

            // learning rule
            auto& stdp = network.make_addon<hummus::STDP>(A_plus, A_minus, Tau_plus, Tau_minus);

            // creating layers
            auto pixel_grid = network.make_grid<hummus::Parrot>(width, height, 1, {});
            auto output = network.make_grid<hummus::CUBA_LIF>(pixel_grid, number_of_sublayers, kernel_size, stride, {&stdp});

            // creating classifier
            hummus::layer classifier;
            if (logistic_regression) {
                classifier = network.make_logistic_regression<hummus::Regression>(training_dataset, test_dataset, 0.1, 0, 0, 70, 128, 10, logistic_start, hummus::optimiser::SGD, tensor_base_name, 0, {});
            } else {
                classifier = network.make_decision<hummus::Decision_Making>(training_dataset, test_dataset, 10, 60, 0, {});
            }

            // connecting the input and output layer with memristive synapses. conductances initialised with a uniform distribution between G_min and G_max
            network.convolution<hummus::Square>(pixel_grid, output, 1, hummus::Uniform(0, 1, 0, 0, false), 100);

            if (i == 0) {
                std::cout << "number of neurons: " << output.neurons.size() << std::endl;
                std::cout << "number of synapses per neuron: " << network.get_neurons()[output.neurons[0]]->get_dendritic_tree().size() << std::endl;
            }
            
            // verbose level
            network.verbosity(0);
            
            // initialise add-ons
            auto& results = network.make_addon<hummus::Analysis>(test_dataset.labels, tensor_base_name+"labels.txt");

            // run the network
            network.run_es_database(training_dataset.files, test_dataset.files, t_max, 0, polarities, width-1+origin, origin, height-1+origin, origin);

            // measuring classification accuracy
            mean_accuracy[i] = results.accuracy(0);
        });
        std::cout << blaze::mean(mean_accuracy) << "\u00b1" << blaze::stddev(mean_accuracy) << std::endl;
    }
    
    // exiting application
    return 0;
}
