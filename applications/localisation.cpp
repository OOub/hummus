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
#include "../source/GUI/display.hpp"
#include "../source/addons/potential_logger.hpp"
#include "../source/addons/myelin_plasticity_logger.hpp"
#include "../source/learningRules/myelin_plasticity_v1.hpp"

int main(int argc, char** argv) {

    /// parameters
    bool use_gui = false;
    
    /// initialisation
    hummus::Network network;
    hummus::DataParser parser;
    
    auto training_data = parser.read_txt_data("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/localisation/direction_only_100.txt");
    auto training_labels = parser.read_txt_labels("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/localisation/direction_only_100_labels.txt");
     
    if (use_gui) {
        auto& display = network.make_gui<hummus::Display>();
        display.set_time_window(10000);
        display.plot_currents();
    }
    
    // delay learning rule
    auto& mp = network.make_addon<hummus::MP_1>();

    //  ----- CREATING THE NETWORK -----
    auto input     = network.make_circle<hummus::Parrot>(8, {0.3}, {}); // input layer with 8 neurons
    auto direction = network.make_layer<hummus::CUBA_LIF>(100, {&mp}, 0, 200, 10, false, false, false, 20); // 100 direction neurons
    
    //  ----- CONNECTING THE NETWORK -----
    network.all_to_all<hummus::Square>(input, direction, 1, hummus::Normal(0.125, 0, 5, 3), 100);
    network.lateral_inhibition<hummus::Square>(direction, 1, hummus::Normal(-1, 0, 0, 1), 100);

    // running network
    network.verbosity(0);
    network.run_data(training_data, 0.1);

    return 0;
}
