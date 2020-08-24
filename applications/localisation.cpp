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
#include "../source/learning_rules/myelin_plasticity_v2.hpp"
#include "../source/addons/analysis.hpp"
#include "../source/addons/spike_logger.hpp"
#include "../source/addons/potential_logger.hpp"
#include "../source/addons/myelin_plasticity_logger.hpp"

int main(int argc, char** argv) {
    // general parameters
    bool synthetic_data = false;
    bool use_gui = false;
    bool random_connectivity = true;
    
    // network parameters
    float timestep = 1;
    bool wta = true;
    bool homeostasis = true;
    
    // initialisation
    hummus::Network network;
    hummus::DataParser parser;
    
    if (use_gui) {
        auto& display = network.make_gui<hummus::Display>();
        display.set_time_window(50000);
        display.track_neuron(8);
    }
    
    // generating sense8 training data
    hummus::dataset training_data;
    if (synthetic_data) {
        training_data = parser.load_data("/Users/omaroubari/Datasets/sense8/sense8_data_syn.npy", "/Users/omaroubari/Datasets/sense8/sense8_labels_syn.txt");
    } else {
        training_data = parser.load_data("/Users/omaroubari/Datasets/sense8/sense8_data.npy", "/Users/omaroubari/Datasets/sense8/sense8_labels.txt");
    }
    
    // initialising addons
    std::string spike_log;
    std::string mp_log;
    if (synthetic_data) {
        spike_log = "sense8_spikelog_syn.bin";
        mp_log    = "sense8_mplog_syn.bin";
    } else {
        spike_log = "sense8_spikelog_1tp.bin";
        mp_log    = "sense8_mplog_1tp.bin";
    }
    auto& mp = network.make_addon<hummus::MP_1>(100, 0.1);
    network.make_addon<hummus::SpikeLogger>(spike_log);
    network.make_addon<hummus::MyelinPlasticityLogger>(mp_log);
    
    // creating layers
    auto input     = network.make_circle<hummus::Parrot>(8, {0.3}, {}); // input layer with 8 neurons
    auto direction = network.make_layer<hummus::CUBA_LIF>(50, {&mp}, 100, 250, 10, wta, homeostasis, false);
    
    // connecting layers
    if (random_connectivity) {
        network.random_to_all<hummus::Exponential>(input, direction, 4, hummus::Normal(0, 0, 5, 3));
    } else {
        network.all_to_all<hummus::Exponential>(input, direction, 1, hummus::Normal(0.125, 0, 5, 3), 100, 3, 200);
    }
    
    // running network
    network.verbosity(1);
    network.run_data(training_data.spikes, timestep);

    // exit application
    return 0;
}
