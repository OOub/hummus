/*
 * ulpec_test.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 02/10/2019
 *
 * Information: ULPEC demonstrator simulation test
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
    bool cadence = false;
    bool use_gui = false;
    bool plot_currents = false;
    bool logistic_regression = true;
    bool seed = false;

    // 3 class NMNIST
//    std::string training_path        = "/Users/omaroubari/Datasets/es_N-MNIST/Train";
//    std::string test_path            = "/Users/omaroubari/Datasets/es_N-MNIST/Test";
//    std::string gmap_filename        = "nmnist_3_g_maps.bin";
//    std::string label_filename       = "nmnist_3_labels.txt";
//    std::vector<std::string> classes = {"5", "6", "9"};
//    int percentage_data              = 100;
//    int logistic_start               = 0;
//    std::string tensor_base_name     = "nmnist_3";
//    bool multiple_epochs             = false;
//    int width                        = 28;
//    int height                       = 28;
//    int origin                       = 0;
//    int repetitions                  = 0;

    // 10 class NMNIST
//    std::string training_path        = "/Users/omaroubari/Datasets/es_N-MNIST/Train";
//    std::string test_path            = "/Users/omaroubari/Datasets/es_N-MNIST/Test";
//    std::string gmap_filename        = "nmnist_10_g_maps.bin";
//    std::string label_filename       = "nmnist_10_labels.txt";
//    std::vector<std::string> classes = {};
//    int percentage_data              = 100;
//    int logistic_start               = 0;
//    std::string tensor_base_name     = "nmnist_10";
//    bool multiple_epochs             = false;
//    int width                        = 28;
//    int height                       = 28;
//    int origin                       = 0;
//    int repetitions                  = 0;

    // 10 class NMNIST - 2 epochs
    std::string training_path        = "/Users/omaroubari/Datasets/es_N-MNIST/Train";
    std::string test_path            = "/Users/omaroubari/Datasets/es_N-MNIST/Test";
    std::string gmap_filename        = "nmnist_10_2e_g_maps.bin";
    std::string label_filename       = "nmnist_10_2e_labels.txt";
    std::vector<std::string> classes = {};
    int percentage_data              = 1;
    int logistic_start               = 0;
    std::string tensor_base_name     = "nmnist_10_2e";
    bool multiple_epochs             = true;
    int width                        = 28;
    int height                       = 28;
    int origin                       = 0;
    int repetitions                  = 0;

    // 4 class POKER-DVS 28x28 cropped
//    std::string training_path        = "/Users/omaroubari/Datasets/es_POKER-DVS/Train";
//    std::string test_path            = "/Users/omaroubari/Datasets/es_POKER-DVS/Test";
//    std::string gmap_filename        = "poker_g_maps.bin";
//    std::string label_filename       = "poker_labels.txt";
//    std::vector<std::string> classes = {};
//    int percentage_data              = 100;
//    int logistic_start               = 0;
//    std::string tensor_base_name     = "poker";
//    bool multiple_epochs             = false;
//    int width                        = 28;
//    int height                       = 28;
//    int origin                       = 0;
//    int repetitions                  = 20;

    // 2 class N-CARS
//    std::string training_path        = "/Users/omaroubari/Datasets/es_denoised_N-CARS/Train";
//    std::string test_path            = "/Users/omaroubari/Datasets/es_denoised_N-CARS/Test";
//    std::string gmap_filename        = "ncars_scaled_g_maps.bin";
//    std::string label_filename       = "ncars_scaled_labels.txt";
//    std::vector<std::string> classes = {};
//    int percentage_data              = 10;
//    int logistic_start               = static_cast<int>(training_path.size()) - 300;
//    std::string tensor_base_name     = "ncars_scaled";
//    bool multiple_epochs             = false;
//    int width                        = 28;
//    int height                       = 28;
//    int origin                       = 10;
//    int repetitions                  = 0;

    // experiment to validate the neuron model in comparison to cadence recordings
    if (cadence) {
        double runtime = 500; /// microseconds

        // initialisation
        hummus::Network network;
        auto& potential_logger = network.make_addon<hummus::PotentialLogger>("ulpec_v_log.bin");
        potential_logger.activate_for(2);

        if (use_gui) {
            auto& display = network.make_gui<hummus::Display>();
            display.set_time_window(runtime); /// microseconds
            display.set_potential_limits(0, 1.5);
            display.track_neuron(2);
            if (plot_currents) {
                display.plot_currents();
                display.set_current_limits(0, 5e-8);
            }
        }

        // creating the layers
        auto input = network.make_layer<hummus::ULPEC_Input>(2, {}, 0, 1.2, 0, 10, 1.2);
        auto output = network.make_layer<hummus::ULPEC_LIF>(1, {}, 0, 5e-12, 0, 0, 12e-9, 0, 650, true, 0.5, 10, 1.5, 1.4);

        // changing the time_constant of the second input neuron from 10 us to 15us
        network.get_neurons()[1]->set_membrane_time_constant(15);

        // connecting the input and output layer with memristive synapses
        network.all_to_all<hummus::Memristor>(input, output, 1, hummus::Normal(1e-5), 100, 1);

        // injecting artificial spikes
        std::vector<hummus::event> pre_one; /// 25 spikes over 500 microseconds separated by 20 us for neuron 0
        for (int i=0; i<25; i++) {
            pre_one.emplace_back(hummus::event{static_cast<double>(i*20+10), 0});
        }
        network.inject_input(pre_one);

        std::vector<hummus::event> pre_two; /// 20 spikes over 500 microseconds separated by 25 us for neuron 1
        for (auto i=0; i<20; i++) {
            pre_two.emplace_back(hummus::event{static_cast<double>(i*25+10), 1});
        }
        network.inject_input(pre_two);

        // running network
        network.verbosity(2);
        network.run(runtime);
    } else {
        // initialisation
        hummus::Network network(seed);
        hummus::DataParser parser(seed);

        if (use_gui) {
            auto& display = network.make_gui<hummus::Display>();
            display.set_time_window(100000);
            display.set_potential_limits(-2.1, 2.1);
            display.track_neuron(2);
            display.hardware_acceleration(false);
            if (plot_currents) {
                display.plot_currents();
                display.set_current_limits(0, 5e-8);
            }
        }

        // generating training database
        auto training_database = parser.generate_database(training_path, percentage_data, repetitions, classes);

        // generating test database
        auto test_database = parser.generate_database(test_path, percentage_data, 0, classes);

        auto& ulpec_stdp = network.make_addon<hummus::ULPEC_STDP>(0.01, -0.01, -1.6, 1.6, 1e-7, 1e-9);

        // creating layers
        auto pixel_grid = network.make_grid<hummus::ULPEC_Input>(width, height, 1, {}, 25, 1.2, 1.1, 10, -1); /// 28 x 28 grid of ULPEC_Input neurons
        auto output = network.make_layer<hummus::ULPEC_LIF>(100, {&ulpec_stdp}, 10, 1e-12, 1, 0, 100e-12, 0, 12.5, true, 0.5, 10, 1.5, 1.4, false); /// 100 ULPEC_LIF neurons

        hummus::layer classifier;
        if (logistic_regression) {
            classifier = network.make_logistic_regression<hummus::Regression>(training_database.second, test_database.second, 0.1, 0, 5e-4, 70, 128, 10, logistic_start, hummus::optimiser::Adam, tensor_base_name, 0, {});
        } else {
            classifier = network.make_decision<hummus::Decision_Making>(training_database.second, test_database.second, 1000, 60, 0, {});
        }

        // connecting the input and output layer with memristive synapses. conductances initialised with a uniform distribution between G_min and G_max
        network.all_to_all<hummus::Memristor>(pixel_grid, output, 1, hummus::Uniform(1e-9, 1e-7, 0, 0, false), 100, -1);

        // running network asynchronously with spatial cropping down to 28x28 input and taking only the first N-MNIST saccade
        network.verbosity(1);

        if (multiple_epochs) {
            // disabling propagation to the regression layer
            network.deactivate_layer(classifier.id);

            // training the STDP
            network.run_es_database(training_database.first, {}, 100000, 0, 1, width-1+origin, origin, height-1+origin, origin);

            // reset the network
            network.reset_network();

            // enabling propagation to the regression layer
            network.activate_layer(classifier.id);

            // initialise add-ons
            auto& results = network.make_addon<hummus::Analysis>(test_database.second, label_filename);
            auto& g_maps = network.make_addon<hummus::WeightMaps>(gmap_filename, 5000);
            g_maps.activate_for(output.neurons);

            // separate epoch to train the Logistic regression
            network.run_es_database(training_database.first, test_database.first, 100000, 0, 1, width-1+origin, origin, height-1+origin, origin);

            // measuring classification accuracy
            results.accuracy();

        } else {
            // initialise add-ons
            auto& results = network.make_addon<hummus::Analysis>(test_database.second, label_filename);
            auto& g_maps = network.make_addon<hummus::WeightMaps>(gmap_filename, 5000);
            g_maps.activate_for(output.neurons);

            // run the network
            network.run_es_database(training_database.first, test_database.first, 100000, 0, 2, width-1+origin, origin, height-1+origin, origin);

            // measuring classification accuracy
            results.accuracy();
        }
    }

    // exiting application
    return 0;
}
