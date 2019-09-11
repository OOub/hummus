/*
 * mp_1D.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: Example of a spiking neural network that can learn one dimensional patterns.
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/GUI/display.hpp"
#include "../source/addons/spikeLogger.hpp"
#include "../source/learningRules/myelinPlasticity.hpp"
#include "../source/neurons/parrot.hpp"
#include "../source/neurons/LIF.hpp"
#include "../source/neurons/decisionMaking.hpp"

int main(int argc, char** argv) {
    //  ----- READING TRAINING DATA FROM FILE -----
	hummus::DataParser dataParser;

    auto trainingData = dataParser.read_txt_data("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/1D_patterns/oneD_10neurons_4patterns_.txt", true, 0);

    //  ----- INITIALISING THE NETWORK -----
    hummus::Network network;

    auto& display = network.make_gui<hummus::Display>();
    network.make_addon<hummus::SpikeLogger>("1D_spikeLog.bin");
    network.make_addon<hummus::MyelinPlasticityLogger>("1D_mpLog.bin");

    //  ----- NETWORK PARAMETERS -----
	float conductance = 200;
    float leakageConductance = 10;
    int   inputNeurons = 10;
    int   layer1Neurons = 4;

	bool burst = false;
	bool homeostasis = true;

	//  ----- INITIALISING THE LEARNING RULE -----
	auto& mp = network.make_addon<hummus::MyelinPlasticity>();

    //  ----- CREATING THE NETWORK -----
    auto input = network.make_layer<hummus::Parrot>(inputNeurons, {});
    auto output = network.make_layer<hummus::LIF>(layer1Neurons, {&mp}, 3, conductance, leakageConductance, homeostasis, burst, 20);

	//  ----- CONNECTING THE NETWORK -----
    network.all_to_all<hummus::Exponential>(input, output, 1, hummus::Normal(0.1, 0, 5, 3), 100);
    network.lateral_inhibition<hummus::Exponential>(output, 1, hummus::Normal(-1, 0, 0, 1), 100);

    //  ----- DISPLAY SETTINGS -----
	display.set_time_window(5000);
	display.track_neuron(11);

    network.turn_off_learning(80000);
    network.verbosity(0);

    //  ----- RUNNING THE NETWORK -----
    network.run_data(trainingData, 0.1);

    //  ----- EXITING APPLICATION -----
    return 0;
}
