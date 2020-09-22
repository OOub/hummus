/*
 * ulpec.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 04/08/2020
 *
 * Information: ULPEC demonstrator simulation test
 */

#include <iostream>
#include "blaze/Blaze.h"
#include "tbb/parallel_for.h"

#include "../source/core.hpp"
#include "../source/neurons/ulpec_input.hpp"
#include "../source/neurons/ulpec_lif.hpp"
#include "../source/neurons/decision_making.hpp"
#include "../source/neurons/regression.hpp"
#include "../source/addons/analysis.hpp"
#include "../source/addons/weight_maps.hpp"
#include "../source/learning_rules/ulpec_stdp.hpp"

int main(int argc, char** argv) {
    int trials = 5;

//    std::string training_path        = "/Users/omaroubari/Datasets/es_N-CARS/Train";
//    std::string test_path            = "/Users/omaroubari/Datasets/es_N-CARS/Test";
//    std::string tensor_base_name     = "ncars";
//    std::vector<std::string> classes = {};
//    int percentage_data              = 100;
//    int width                        = 64;
//    int height                       = 56;
//    int origin                       = 0;
//    int number_of_sublayers          = 5;
//    int kernel_size                  = 13;
//    int stride                       = 11;
//    int regression_size              = 1000;
//    uint64_t t_max                   = UINT64_MAX;
//    int polarities                   = 1;
//    bool multiple_epochs             = false;
//    bool logistic_regression         = true;
//    bool seed                        = false;

    // nmnist parameters
    std::string training_path        = "/Users/omaroubari/Datasets/es_N-MNIST/Train";
    std::string test_path            = "/Users/omaroubari/Datasets/es_N-MNIST/Test";
    std::string tensor_base_name     = "nmnist";
    std::vector<std:: string> classes = {};
    int percentage_data              = 100;
    int width                        = 28;
    int height                       = 28;
    int origin                       = 0;
    int number_of_sublayers          = 4;
    int kernel_size                  = 7;
    int stride                       = 5;
    int regression_size              = 5000;
    uint64_t t_max                   = 100000;//UINT64_MAX;
    int polarities                   = 1;//2;
    bool multiple_epochs             = false;
    bool logistic_regression         = true;
    bool seed                        = false;
    
    // neuron parameters
    float scaling_factor = 12.5;
    float capacitance = 1e-12;
    float threshold = 0.8;
    float i_discharge = 100e-12;
    float delta_v = 1.4;
    float skip = false;
    
    // learning parameters
    float learning_rate = 0.001;
    float gmax = 1e-8;
    float gmin = 1e-6;

    if (trials == 1) {
        // initialisation
        hummus::Network network(seed);
        hummus::DataParser parser(seed);
        
        // verbose level
        network.verbosity(0);
        
        // generating training database
        auto training_dataset = parser.load_data(training_path, percentage_data, classes);
        int logistic_start = static_cast<int>(training_dataset.files.size()) - regression_size;
        if (regression_size == 0) {
            logistic_start = 0;
        }
        
        // generating test database
        auto test_dataset = parser.load_data(test_path, percentage_data, classes);

        // learning rule
        auto& ulpec_stdp = network.make_addon<hummus::ULPEC_STDP>(learning_rate, -learning_rate, -1.6, 1.6, gmin, gmax, 1);
        
        // creating layers
        auto pixel_grid = network.make_grid<hummus::ULPEC_Input>(width, height, 1, {}, 25, 1.2, 1.1, 10, -1);
        auto output = network.make_grid<hummus::ULPEC_LIF>(pixel_grid, number_of_sublayers, kernel_size, stride, {&ulpec_stdp}, 10, capacitance, threshold, 0, i_discharge, 0, scaling_factor, true, 0.5, 10, 1.5, delta_v, skip);
        
        // creating classifier
        hummus::layer classifier;
        if (logistic_regression) {
            classifier = network.make_logistic_regression<hummus::Regression>(training_dataset, test_dataset, 0.1, 0, 0.01, 70, 128, 10, logistic_start, hummus::optimiser::SGD, tensor_base_name, 0, {});
        } else {
            classifier = network.make_decision<hummus::Decision_Making>(training_dataset, test_dataset, 10, 60, 0, {});
        }

        // connecting the input and output layer with memristive synapses. conductances initialised with a uniform distribution between G_min and G_max
        network.convolution<hummus::Memristor>(pixel_grid, output, 1, hummus::Uniform(gmax, gmin, 0, 0, false), 100, -1);

        std::cout << "number of neurons: " << output.neurons.size() << std::endl;
        std::cout << "number of synapses per neuron: " << network.get_neurons()[output.neurons[0]]->get_dendritic_tree().size() << std::endl;

        if (multiple_epochs) {
            // disabling propagation to the regression layer
            network.deactivate_layer(classifier.id);

            // training the STDP
            network.run_es_database(training_dataset.files, {}, t_max, 0, polarities, width-1+origin, origin, height-1+origin, origin);

            // reset the network
            network.reset_network();

            // enabling propagation to the regression layer
            network.activate_layer(classifier.id);

            // initialise add-ons
            auto& results = network.make_addon<hummus::Analysis>(test_dataset.labels, tensor_base_name+"labels.txt");
            auto& gmaps = network.make_addon<hummus::WeightMaps>(tensor_base_name+"gmaps.bin", 5000);
            gmaps.activate_for(output.neurons);

            // separate epoch to train the Logistic regression
            network.run_es_database(training_dataset.files, test_dataset.files, t_max, 0, polarities, width-1+origin, origin, height-1+origin, origin);

            // measuring classification accuracy
            results.accuracy();

        } else {
            // initialise add-ons
            auto& results = network.make_addon<hummus::Analysis>(test_dataset.labels, tensor_base_name+"labels.txt");
            auto& g_maps = network.make_addon<hummus::WeightMaps>(tensor_base_name+"gmaps.bin", 5000);
            g_maps.activate_for(output.neurons);

            // run the network
            network.run_es_database(training_dataset.files, test_dataset.files, t_max, 0, polarities, width-1+origin, origin, height-1+origin, origin);

            // measuring classification accuracy
            results.accuracy();
        }
    } else if (trials > 1) {
        blaze::DynamicVector<double> mean_accuracy(trials);
        tbb::parallel_for(static_cast<size_t>(0), static_cast<size_t>(trials), [&](size_t i) {
            // initialisation
            hummus::Network network(seed);
            hummus::DataParser parser(seed);
            
            // verbose level
            network.verbosity(0);
            
            // generating training database
            auto training_dataset = parser.load_data(training_path, percentage_data, classes);
            int logistic_start = static_cast<int>(training_dataset.files.size()) - regression_size;
            if (regression_size == 0) {
                logistic_start = 0;
            }
            
            // generating test database
            auto test_dataset = parser.load_data(test_path, percentage_data, classes);

            // learning rule
            auto& ulpec_stdp = network.make_addon<hummus::ULPEC_STDP>(learning_rate, -learning_rate, -1.6, 1.6, gmin, gmax, 1);

            // creating layers
            auto pixel_grid = network.make_grid<hummus::ULPEC_Input>(width, height, 1, {}, 25, 1.2, 1.1, 10, -1);
            auto output = network.make_grid<hummus::ULPEC_LIF>(pixel_grid, number_of_sublayers, kernel_size, stride, {&ulpec_stdp}, 10, capacitance, threshold, 0, i_discharge, 0, scaling_factor, true, 0.5, 10, 1.5, delta_v, skip);

            // creating classifier
            hummus::layer classifier;
            if (logistic_regression) {
                classifier = network.make_logistic_regression<hummus::Regression>(training_dataset, test_dataset, 0.1, 0, 0.01, 70, 128, 10, logistic_start, hummus::optimiser::SGD, tensor_base_name+std::to_string(i), 0, {});
            } else {
                classifier = network.make_decision<hummus::Decision_Making>(training_dataset, test_dataset, 10, 60, 0, {});
            }

            // connecting the input and output layer with memristive synapses. conductances initialised with a uniform distribution between G_min and G_max
            network.convolution<hummus::Memristor>(pixel_grid, output, 1, hummus::Uniform(gmax, gmin, 0, 0, false), 100, -1);

            if (i == 0) {
                std::cout << "number of neurons: " << output.neurons.size() << std::endl;
                std::cout << "number of synapses per neuron: " << network.get_neurons()[output.neurons[0]]->get_dendritic_tree().size() << std::endl;
            }

            if (multiple_epochs) {
                // disabling propagation to the regression layer
                network.deactivate_layer(classifier.id);

                // training the STDP
                network.run_es_database(training_dataset.files, {}, t_max, 0, polarities, width-1+origin, origin, height-1+origin, origin);

                // reset the network
                network.reset_network();

                // enabling propagation to the regression layer
                network.activate_layer(classifier.id);

                // initialise add-ons
                auto& results = network.make_addon<hummus::Analysis>(test_dataset.labels, tensor_base_name+std::to_string(i)+"labels.txt");
                auto& gmaps = network.make_addon<hummus::WeightMaps>(tensor_base_name+std::to_string(i)+"gmaps.bin", 5000);
                gmaps.activate_for(output.neurons);

                // separate epoch to train the Logistic regression
                network.run_es_database(training_dataset.files, test_dataset.files, t_max, 0, polarities, width-1+origin, origin, height-1+origin, origin);

                // measuring classification accuracy
                mean_accuracy[i] = results.accuracy(0);

            } else {
                // initialise add-ons
                auto& results = network.make_addon<hummus::Analysis>(test_dataset.labels, tensor_base_name+std::to_string(i)+"labels.txt");
                auto& g_maps = network.make_addon<hummus::WeightMaps>(tensor_base_name+std::to_string(i)+"gmaps.bin", 5000);
                g_maps.activate_for(output.neurons);

                // run the network
                network.run_es_database(training_dataset.files, test_dataset.files, t_max, 0, polarities, width-1+origin, origin, height-1+origin, origin);

                // measuring classification accuracy
                mean_accuracy[i] = results.accuracy(0);
            }
        });
        std::cout << blaze::mean(mean_accuracy) << "\u00b1" << blaze::stddev(mean_accuracy) << std::endl;
    }
    
    // exiting application
    return 0;
}
