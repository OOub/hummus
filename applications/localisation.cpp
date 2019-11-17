/*
 * localisation.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 03/05/2019
 *
 * Information: Network for wave triangulation from an array of 8 piezoelectric sensors using delay learning
 * grid of 2D neurons, each associated with a specific temporal pattern. the final output neuron give us the position of the touch
 * a calibration step is necessary to find the delays for each grid point
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/neurons/cuba_lif.hpp"
#include "../source/neurons/parrot.hpp"
#include "../source/neurons/decision_making.hpp"
#include "../source/GUI/display.hpp"
#include "../source/learning_rules/myelin_plasticity_v1.hpp"
#include "../source/addons/analysis.hpp"
#include "../source/addons/spike_logger.hpp"
#include "../source/addons/myelin_plasticity_logger.hpp"

int main(int argc, char** argv) {
    // parameters
    bool use_gui = false;
    int  time_scaling_factor = 1e4;
    float timestep = 0.1;
    
    // initialisation
    hummus::Network network;
    hummus::DataParser parser;
    
    if (use_gui) {
        auto& display = network.make_gui<hummus::Display>();
        display.set_time_window(5000);
        display.track_neuron(8);
    }
    
    // generating sense8 training and testing databases
    auto training_database = parser.generate_database("/Users/omaroubari/Datasets/sense8_no_distance/Train", 100, 10, {"90", "180"});
    auto test_database = parser.generate_database("/Users/omaroubari/Datasets/sense8_no_distance/Test", 100, 0, {"90", "180"});
    
    // initialising addons
    auto& mp = network.make_addon<hummus::MP_1>();
    auto& results = network.make_addon<hummus::Analysis>(test_database.second, "labels.txt");
    network.make_addon<hummus::SpikeLogger>("sense8_spikelog.bin");
    network.make_addon<hummus::MyelinPlasticityLogger>("sense8_mplog.bin");
    
    // creating layers
    auto input = network.make_circle<hummus::Parrot>(8, {0.3}, {}); // input layer with 8 neurons
    auto output = network.make_layer<hummus::CUBA_LIF>(100, {&mp}, 0, 200, 10, true, false, false, 20); // 100 ouptput neurons
    auto decision = network.make_decision<hummus::Decision_Making>(training_database.second, 10, 50, 0, {});
    
    // connecting layers
    network.all_to_all<hummus::Square>(input, output, 1, hummus::Normal(0.125, 0, 5, 3), 100, 10, 80);
    
    // running network
    network.verbosity(1);
    network.run_npy_database(training_database.first, timestep, test_database.first, time_scaling_factor);
    
    // measuring classification accuracy
    results.accuracy();
    
    // exit application
    return 0;
}
