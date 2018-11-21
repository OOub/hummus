/*
 * stdpPotentiation.cpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/05/2018
 *
 * Information: Example of stdp working. 10 neurons are connected to an output neuron. In the beginning, all 10 neurons are needed to fire (disable the learning rule to see that). With
 * STDP, postsynaptic firing slowly shifts and the neurons that fire after the output neuron get depressed (use the debug option to see the weight progression as the network is
 * running)
 */

#include <iostream>

#include "../source/network.hpp"
#include "../source/qtDisplay.hpp"
#include "../source/spikeLogger.hpp"
#include "../source/stdp.hpp"
#include "../source/supervisedWTA.hpp"

int main(int argc, char** argv)
{

    //  ----- READING TRAINING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
	
	auto trainingData = dataParser.readTrainingData("../../data/stdpTest.txt");
	
    //  ----- INITIALISING THE NETWORK -----
	adonis_c::QtDisplay qtDisplay;
	adonis_c::SpikeLogger spikeLogger("stdpSpikeLog");
	adonis_c::Network network({&spikeLogger}, &qtDisplay);

    //  ----- NETWORK PARAMETERS -----
	auto runtime = trainingData.back().timestamp+1;
	float timestep = 0.1;
	
	float decayCurrent = 10;
	float potentialDecay = 20;
	float refractoryPeriod = 3;
	
    int inputNeurons = 10;
    int layer1Neurons = 2;
	
    float weight = 1./10;
	
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis_c::Stdp stdp(0,1);
	adonis_c::SupervisedWTA wta;
	
	//  ----- CREATING THE NETWORK -----
	network.addLayer(0, {&stdp}, inputNeurons, 1, 1, decayCurrent, potentialDecay, refractoryPeriod);
	network.addLayer(1, {&stdp, &wta}, layer1Neurons, 1, 1, decayCurrent, potentialDecay, refractoryPeriod);

    //  ----- CONNECTING THE NETWORK -----
	network.allToAll(network.getLayers()[0], network.getLayers()[1], false, weight, false, 0);

	//  ----- INJECTING SPIKES -----
	network.injectSpikeFromData(&trainingData);
	
    //  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(100);
  	qtDisplay.trackNeuron(10);
  	qtDisplay.trackLayer(1);
	
    //  ----- RUNNING THE NETWORK -----
    network.run(runtime, timestep);

    //  ----- EXITING APPLICATION -----
    return 0;
}
