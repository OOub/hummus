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
    std::string training_path        = "/Users/omaroubari/Datasets/es_POKER-DVS/Train";
    std::string test_path            = "/Users/omaroubari/Datasets/es_POKER-DVS/Test";
    std::string tensor_base_name     = "pokerdvs";
    std::vector<std::string> classes = {};
    int percentage_data              = 5;
    int width                        = 28;
    int height                       = 28;
    int origin                       = 0;
    int number_of_neurons            = 100;
    int regression_size              = 1000;
    uint64_t t_max                   = 100000;
    int polarities                   = 1;
    bool logistic_regression         = true;
    bool seed                        = true;
    
    // initialisation
    hummus::Network network(seed);
    hummus::DataParser parser(seed);

    // generating training database
    auto training_dataset = parser.load_data(training_path, percentage_data, classes);
    int logistic_start = static_cast<int>(training_dataset.files.size()) - regression_size;

    // generating test database
    auto test_dataset = parser.load_data(test_path, percentage_data, classes);

    // learning rule
    auto& ulpec_stdp = network.make_addon<hummus::ULPEC_STDP>(0.01, -0.01, -1.6, 1.6, 1e-7, 1e-9);

    // creating layers
    auto pixel_grid = network.make_grid<hummus::ULPEC_Input>(width, height, 25, 1.2, 1.1, 10, -1); /// 28 x 28 grid of ULPEC_Input neurons
    auto output_one = network.make_layer<hummus::ULPEC_LIF>(number_of_neurons, {&ulpec_stdp}, 10, 1e-12, 1, 0, 100e-12, 0, 12.5, true, 0.5, 10, 1.5, 1.4, false); /// 100 ULPEC_LIF neurons
    auto output_two = network.make_layer<hummus::ULPEC_LIF>(number_of_neurons, {&ulpec_stdp}, 10, 100e-12, 1, 0, 100e-12, 0, 12.5, true, 0.5, 10, 1.5, 1.4, false);
    
    // creating classifier
    hummus::layer classifier;
    if (logistic_regression) {
        classifier = network.make_logistic_regression<hummus::Regression>(training_dataset, test_dataset, 0.1, 0, 5e-4, true, 70, 128, 10, logistic_start, hummus::optimiser::Adam, tensor_base_name, 0, {});
    } else {
        classifier = network.make_decision<hummus::Decision_Making>(training_dataset, test_dataset, 1000, 60, 0, {});
    }

    // connecting the input and output layer with memristive synapses. conductances initialised with a uniform distribution between G_min and G_max
    network.all_to_all<hummus::Memristor>(pixel_grid, output_one, 1, hummus::Uniform(1e-9, 1e-7, 0, 0, false), 100, -1);
    network.all_to_all<hummus::Memristor>(output_one, output_two, 1, hummus::Uniform(1e-9, 1e-7, 0, 0, false), 100, -1);
    
    // verbose level
    network.verbosity(1);

    // initialise add-ons
    auto& results = network.make_addon<hummus::Analysis>(test_dataset.labels, tensor_base_name+"labels.txt");

    // run the network
    network.run_es_database(training_dataset.files, test_dataset.files, t_max, 0, polarities, width-1+origin, origin, height-1+origin, origin);

    // measuring classification accuracy
    results.accuracy();

    // exiting application
    return 0;
}
