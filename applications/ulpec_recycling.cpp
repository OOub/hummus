/*
 * ulpec_recycling.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 22/12/2019
 *
 * Information: ULPEC demonstrator simulation test - can we recycle the unused neurons
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/GUI/display.hpp"
#include "../source/neurons/ulpec_input.hpp"
#include "../source/neurons/ulpec_lif.hpp"
#include "../source/neurons/decision_making.hpp"
#include "../source/neurons/regression.hpp"
#include "../source/addons/potential_logger.hpp"
#include "../source/addons/analysis.hpp"
#include "../source/addons/weight_maps.hpp"
#include "../source/learning_rules/ulpec_stdp.hpp"

int main(int argc, char** argv) {
    // parameters
    bool seed = true;

    // initialisation
    hummus::Network network(seed);
    hummus::DataParser parser(seed);

    // generating training database
    auto training_database = parser.generate_database("/Users/omaroubari/Datasets/es_N-MNIST/Train", 10, 0, {"5", "6", "9"});
    
    // generating test database
    auto test_database = parser.generate_database("/Users/omaroubari/Datasets/es_N-MNIST/Test", 10, 0, {"5", "6", "9"});

    // learning rule is a combination of STDP + recycling metric (start with randomization from uniform distribution)
    auto& ulpec_stdp = network.make_addon<hummus::ULPEC_STDP>(0.01, -0.01, -1.6, 1.6, 1e-7, 1e-9);

    // creating layers
    auto pixel_grid = network.make_grid<hummus::ULPEC_Input>(28, 28, 1, {}, 25, 1.2, 1.1, 10, -1); /// 28 x 28 grid of ULPEC_Input neurons
    auto output = network.make_layer<hummus::ULPEC_LIF>(100, {&ulpec_stdp}, 10, 1e-12, 1, 0, 100e-12, 0, 12.5, true, 0.5, 10, 1.5, 1.4, false); /// 100 ULPEC_LIF neurons
    auto classifier = network.make_logistic_regression<hummus::Regression>(training_database.second, test_database.second, 0.1, 0, 5e-4, 70, 128, 10, 0, hummus::optimiser::Adam, "nmnist_recycling", 0, {});

    // connecting the input and output layer with memristive synapses. conductances initialised with a uniform distribution between G_min and G_max
    network.all_to_all<hummus::Memristor>(pixel_grid, output, 1, hummus::Uniform(1e-9, 1e-7, 0, 0, false), 100, -1);

    // running network asynchronously with spatial cropping down to 28x28 input and taking only the first N-MNIST saccade
    network.verbosity(1);

    // initialise add-ons
    auto& results = network.make_addon<hummus::Analysis>(test_database.second, "nmnist_recycling_labels.txt");

    // run the network
    network.run_es_database(training_database.first, test_database.first, 100000, 0, 2, 27, 0, 27, 0);

    // measuring classification accuracy
    results.accuracy();

    // exiting application
    return 0;
}
