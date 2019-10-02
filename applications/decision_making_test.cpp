/*
 * decision_making_test.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/07/2019
 *
 * Information: Example of the decision-making at work.
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/GUI/display.hpp"
#include "../source/addons/analysis.hpp"
#include "../source/learningRules/stdp.hpp"
#include "../source/neurons/parrot.hpp"
#include "../source/neurons/cuba_lif.hpp"
#include "../source/neurons/decisionMaking.hpp"
#include "../source/addons/spike_logger.hpp"

int main(int argc, char** argv) {
    /// parameters
    bool use_gui = false;
    
    /// initialisation
    hummus::Network network;
    network.make_addon<hummus::SpikeLogger>("spike_log.bin");
    hummus::DataParser parser;
    
    if (use_gui) {
        auto& display = network.make_gui<hummus::Display>();
        display.set_time_window(100000);
        display.track_neuron(1228);
        display.plot_currents();
    }
    
    /// generating N-MNIST training database
    auto training_database = parser.generate_nmnist_database("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/es_N-MNIST/small_Train", 100, {"0", "1"});
    
    /// generating N-MNIST test database
    auto test_database = parser.generate_nmnist_database("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/es_N-MNIST/Test", 1, {"0", "1"});
    
    auto& stdp = network.make_addon<hummus::STDP>(1, 0.4, 20000, 40000);
    auto& results = network.make_addon<hummus::Analysis>(test_database.second);

    /// creating the layers
    auto pixel_grid = network.make_grid<hummus::Parrot>(35, 35, 1, {});
    auto hidden_layer = network.make_layer<hummus::CUBA_LIF>(10, {&stdp}, 10000, 20000, 1, false, false, false, 20000);
    auto decision_layer = network.make_decision<hummus::DecisionMaking>(training_database.second, 10, 60, 0, {});

    network.all_to_all<hummus::Square>(pixel_grid, hidden_layer, 1, hummus::Normal(0.08, 0.02, 5000, 300), 80, 10000);
    network.lateral_inhibition<hummus::Square>(hidden_layer, 1, hummus::Normal(-1, 0, 0, 1), 20, 10000);
    
    /// running network
    network.verbosity(0);
    network.run_database(training_database.first, test_database.first, 100000);
    
    /// Measuring Classification Accuracy
    results.accuracy();
    
    /// Exiting Application
    return 0;
}
