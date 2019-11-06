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
#include "../source/addons/potential_logger.hpp"
#include "../source/addons/analysis.hpp"
#include "../source/addons/weight_maps.hpp"
#include "../source/learning_rules/ulpec_stdp.hpp"

int main(int argc, char** argv) {
    // parameters
    bool cadence = false;
    bool use_gui = false;
    bool plot_currents = false;

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
        hummus::Network network;
        hummus::DataParser parser;
        
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
        
        // generating N-MNIST training database
        auto training_database = parser.generate_nmnist_database("/Users/omaroubari/Datasets/es_N-MNIST/Train", 1, {"5", "6", "9"});
        
        // generating N-MNIST test database
        auto test_database = parser.generate_nmnist_database("/Users/omaroubari/Datasets/es_N-MNIST/Test", 1, {"5", "6", "9"});
        
        auto& ulpec_stdp = network.make_addon<hummus::ULPEC_STDP>(0.1, -0.1, -1.6, 1.6, 1e-7, 1e-9);
        auto& results = network.make_addon<hummus::Analysis>(test_database.second, "labels.txt");
        
        // creating layers
        auto pixel_grid = network.make_grid<hummus::ULPEC_Input>(28, 28, 1, {}, 25, 1.2, 1.1, 10, -1); /// 28 x 28 grid of ULPEC_Input neurons
        auto output = network.make_layer<hummus::ULPEC_LIF>(100, {&ulpec_stdp}, 10, 1e-12, 1.2, 0, 100e-12, 0, 15, true, 0.5, 10, 1.5, 1.4); /// 100 ULPEC_LIF neurons
        auto decision_layer = network.make_decision<hummus::Decision_Making>(training_database.second, 10, 50, 0, {});

        auto& g_maps = network.make_addon<hummus::WeightMaps>("ulpec_g_maps.bin", 1000);
        g_maps.activate_for(output.neurons);
        
        // connecting the input and output layer with memristive synapses. conductances initialised with a uniform distribution between G_min and G_max
        network.all_to_all<hummus::Memristor>(pixel_grid, output, 1, hummus::Uniform(1e-9, 1e-7, 0, 0, false), 100, -1);
        
        // running network asynchronously with spatial cropping down to 28x28 input and taking only the first N-MNIST saccade
        network.verbosity(1);
        network.run_database(training_database.first, test_database.first, 100000, 0, 1, 27, 0, 27, 0);
                              
        // measuring Classification Accuracy
        results.accuracy();
    }
    
    // exiting application
    return 0;
}
