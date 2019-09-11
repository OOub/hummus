/*
 * stdp_test.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: Example of stdp working. 10 neurons are connected to an output neuron. In the beginning, all 10 neurons are needed
 * to fire (disable the learning rule to see that). With STDP, postsynaptic firing slowly shifts and the neurons that fire after the
 * output neuron get depressed (use the debug option to see the weight progression as the network is running)
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/GUI/display.hpp"
#include "../source/learningRules/stdp.hpp"
#include "../source/neurons/parrot.hpp"
#include "../source/neurons/LIF.hpp"

int main(int argc, char** argv) {
    //  ----- READING TRAINING DATA FROM FILE -----
	hummus::DataParser dataParser;

	auto trainingData = dataParser.read_txt_data("../../data/stdpTest.txt");

    //  ----- INITIALISING THE NETWORK -----
    hummus::Network network;
    auto& display = network.make_gui<hummus::Display>();

    //  ----- NETWORK PARAMETERS -----
    float conductance = 200;
    float leakageConductance = 10;
	float refractoryPeriod = 30;
    int inputNeurons = 10;
    int layer1Neurons = 1;
    float weight = 1./10;

	//  ----- INITIALISING THE LEARNING RULE -----
    auto& stdp = network.make_addon<hummus::STDP>();

	//  ----- CREATING THE NETWORK -----
    auto input = network.make_layer<hummus::Parrot>(inputNeurons, {});
    auto output = network.make_layer<hummus::LIF>(layer1Neurons, {&stdp}, refractoryPeriod, conductance, leakageConductance, false, true);

    //  ----- CONNECTING THE NETWORK -----
    network.all_to_all<hummus::Exponential>(input, output, 1, hummus::Normal(weight, 0, 0, 0), 100);

    //  ----- DISPLAY SETTINGS -----
  	display.set_time_window(100);
  	display.track_neuron(10);
  	display.track_layer(1);
    display.plot_currents(true);

    //  ----- RUNNING THE NETWORK -----
    network.run_data(trainingData, 0.1);

    //  ----- EXITING APPLICATION -----
    return 0;
}
