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
    bool sequential_run = true;
    bool synthetic_data = false;
    bool use_gui = true;
    
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
    
    if (sequential_run) {
        // generating sense8 training data
        std::vector<hummus::event> training_data;
        std::deque<hummus::label> training_labels;
        if (synthetic_data) {
            training_data = parser.read_txt_data("/Users/omaroubari/Datasets/sense8_data_syn.txt");
            training_labels = parser.read_txt_labels("/Users/omaroubari/Datasets/sense8_labels_syn.txt");
        } else {
            training_data = parser.read_txt_data("/Users/omaroubari/Datasets/sense8_data.txt");
            training_labels = parser.read_txt_labels("/Users/omaroubari/Datasets/sense8_labels.txt");
        }
        
        // initialising addons
        std::string spike_log;
        std::string mp_log;
        if (synthetic_data) {
            spike_log = "sense8_spikelog_syn2.bin";
            mp_log    = "sense8_mplog_syn2.bin";
        } else {
            spike_log = "sense8_spikelog.bin";
            mp_log    = "sense8_mplog.bin";
        }
        auto& mp = network.make_addon<hummus::MP_1>(100, 1);
        network.make_addon<hummus::SpikeLogger>(spike_log);
        network.make_addon<hummus::MyelinPlasticityLogger>(mp_log);
        
        // creating layers
        auto input     = network.make_circle<hummus::Parrot>(8, {0.3}, {}); // input layer with 8 neurons
        auto direction = network.make_layer<hummus::CUBA_LIF>(50, {&mp}, 100, 250, 10, wta, homeostasis, false);
        
        // connecting layers
        network.all_to_all<hummus::Square>(input, direction, 1, hummus::Normal(0.125, 0, 5, 3), 100, 3, 180);
        
        // running network
        network.verbosity(1);
        network.run_data(training_data, timestep);
        
    } else {
        // parameter for npy run
        int  time_scaling_factor = 1e4;
        
        // generating sense8 training and testing databases
        std::pair<std::vector<std::string>, std::deque<hummus::label>> training_database;
        std::pair<std::vector<std::string>, std::deque<hummus::label>> test_database;
        if (synthetic_data) {
            training_database = parser.generate_database("/Users/omaroubari/Datasets/sense8_synthetic/Train", 100, 100, {});
            test_database = parser.generate_database("/Users/omaroubari/Datasets/sense8_synthetic/Test", 100, 0, {});
        } else {
            training_database = parser.generate_database("/Users/omaroubari/Datasets/sense8_no_distance/Train", 100, 100, {});
            test_database = parser.generate_database("/Users/omaroubari/Datasets/sense8_no_distance/Test", 100, 0, {});
        }
        
        // initialising addons
        auto& mp = network.make_addon<hummus::MP_1>();
        auto& potentials = network.make_addon<hummus::PotentialLogger>("sense8_potentiallog.bin");
        
        network.make_addon<hummus::SpikeLogger>("sense8_spikelog.bin");
        network.make_addon<hummus::MyelinPlasticityLogger>("sense8_mplog.bin");
        
        // creating layers
        auto input = network.make_circle<hummus::Parrot>(8, {0.3}, {}); // input layer with 8 neurons
        auto output = network.make_layer<hummus::CUBA_LIF>(8, {&mp}, 0, 250, 10, wta, homeostasis, false); // 100 output neurons
        
        // add mask on potential logger
        potentials.activate_for(output.neurons);
        
        // connecting layers
        network.all_to_all<hummus::Square>(input, output, 1, hummus::Normal(0.125, 0, 5, 3), 100, 3, 270);
        
        // running network
        network.verbosity(1);
        network.run_npy_database(training_database.first, timestep, test_database.first, time_scaling_factor);
    }
    // exit application
    return 0;
}
