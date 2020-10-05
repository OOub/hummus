/*
 * ulpec_complex.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/08/2020
 *
 * Information: ULPEC demonstrator simulation test with multiple layers
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/neurons/ulpec_input.hpp"
#include "../source/neurons/ulpec_lif.hpp"
#include "../source/neurons/decision_making.hpp"
#include "../source/neurons/regression.hpp"
#include "../source/addons/analysis.hpp"
#include "../source/addons/weight_maps.hpp"
#include "../source/learning_rules/ulpec_stdp.hpp"

int main(int argc, char** argv) {
    // parameters
    std::string training_path        = "/Users/omaroubari/Datasets/es_N-CARS/Train";
    std::string test_path            = "/Users/omaroubari/Datasets/es_N-CARS/Test";
    std::string tensor_base_name     = "ncars";
    std::vector<std::string> classes = {};
    int percentage_data              = 100;
    int width                        = 28;
    int height                       = 28;
    int origin                       = 0;
    int number_of_sublayers1         = 4;
    int kernel_size1                 = 5;
    int stride1                      = 5;
    int number_of_sublayers2         = 4;
    int kernel_size2                 = 5;
    int stride2                      = 5;
    int regression_size              = 5000;
    uint64_t t_max                   = UINT64_MAX;
    int polarities                   = 2;
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
    float learning_rate = 0.01;
    float gmin = 1e-8;
    float gmax = 1e-6;
    
    // logistic regression parameters
    int ref_period = 10;
    int epochs = 100;
    int batch_size = 32;
    float lr = 0.01;
    float momentum = 0.9;
    float weight_Decay = 0.01;
    bool lr_decay = true;
    
    // initialisation
    hummus::Network network(seed);
    hummus::DataParser parser(seed);

    // generating training database
    auto training_dataset = parser.load_data(training_path, percentage_data, classes);
    int logistic_start = static_cast<int>(training_dataset.files.size()) - regression_size;

    // generating test database
    auto test_dataset = parser.load_data(test_path, percentage_data, classes);

    // learning rule
    auto& ulpec_stdp = network.make_addon<hummus::ULPEC_STDP>(learning_rate, -learning_rate, -1.6, 1.6, gmax, gmin);

    // creating layers
    auto pixel_grid = network.make_grid<hummus::ULPEC_Input>(width, height, 25, 1.2, 1.1, 10, -1); /// 28 x 28 grid of ULPEC_Input neurons
    auto output_one = network.make_grid<hummus::ULPEC_LIF>(pixel_grid, number_of_sublayers1, kernel_size1, stride1, {&ulpec_stdp}, ref_period, capacitance, threshold, 0, i_discharge, 0, scaling_factor, true, 0.5, 10, 1.5, delta_v, skip);
    auto output_two = network.make_grid<hummus::ULPEC_LIF>(output_one, number_of_sublayers2, kernel_size2, stride2, {&ulpec_stdp}, ref_period, capacitance, threshold, 0, i_discharge, 0, scaling_factor, true, 0.5, 10, 1.5, delta_v, skip);
    
    // creating classifier
    hummus::layer classifier;
    if (logistic_regression) {
        classifier = network.make_logistic_regression<hummus::Regression>(training_dataset, test_dataset, lr, momentum, weight_Decay, lr_decay, epochs, batch_size, 10, logistic_start, hummus::optimiser::SGD, tensor_base_name, 0, {});
    } else {
        classifier = network.make_decision<hummus::Decision_Making>(training_dataset, test_dataset, 1000, 60, 0, {});
    }

    // connecting the input and output layer with memristive synapses. conductances initialised with a uniform distribution between G_min and G_max
    network.convolution<hummus::Memristor>(pixel_grid, output_one, 1, hummus::Uniform(gmin, gmax, 0, 0, false), 100, -1);
    network.convolution<hummus::Memristor>(output_one, output_two, 1, hummus::Uniform(gmin, gmax, 0, 0, false), 100, -1);
    
    // verbose level
    network.verbosity(0);

    // initialise add-ons
    auto& results = network.make_addon<hummus::Analysis>(test_dataset.labels, tensor_base_name+"labels.txt");

    // run the network
    network.run_es_database(training_dataset.files, test_dataset.files, t_max, 0, polarities, width-1+origin, origin, height-1+origin, origin);

    // measuring classification accuracy
    results.accuracy();

    // exiting application
    return 0;
}
