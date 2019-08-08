/*
 * stdpPotentiation.cpp
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
#include "../source/randomDistributions/normal.hpp"
#include "../source/GUI/qt/qtDisplay.hpp"
#include "../source/learningRules/stdp.hpp"
#include "../source/learningRules/timeInvariantSTDP.hpp"
#include "../source/learningRules/myelinPlasticity.hpp"
#include "../source/learningRules/rewardModulatedSTDP.hpp"
#include "../source/neurons/parrot.hpp"
#include "../source/neurons/LIF.hpp"
#include "../source/neurons/decisionMaking.hpp"
#include "../source/synapses/exponential.hpp"

int main(int argc, char** argv) {
    //  ----- READING TRAINING DATA FROM FILE -----
	hummus::DataParser dataParser;

	auto trainingData = dataParser.readData("../../data/stdpTest.txt");

    //  ----- INITIALISING THE NETWORK -----
    hummus::Network network;
    auto& display = network.makeGUI<hummus::QtDisplay>();

    //  ----- NETWORK PARAMETERS -----
    float conductance = 200;
    float leakageConductance = 10;
	float refractoryPeriod = 30;
    int inputNeurons = 10;
    int layer1Neurons = 1;
    float weight = 1./10;

	//  ----- INITIALISING THE LEARNING RULE -----
    auto& stdp = network.makeAddon<hummus::STDP>();

	//  ----- CREATING THE NETWORK -----
    auto input = network.makeLayer<hummus::Parrot>(inputNeurons, {});
    auto output = network.makeLayer<hummus::LIF>(layer1Neurons, {&stdp}, refractoryPeriod, conductance, leakageConductance, false, true);

    //  ----- CONNECTING THE NETWORK -----
    network.allToAll<hummus::Exponential>(input, output, 1, hummus::Normal(weight, 0, 0, 0), 100, hummus::synapseType::excitatory);

    //  ----- DISPLAY SETTINGS -----
  	display.setTimeWindow(100);
  	display.trackNeuron(10);
  	display.trackLayer(1);
    display.plotCurrents(true);

    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, 0.1);

    //  ----- EXITING APPLICATION -----
    return 0;
}
