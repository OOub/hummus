/*
 * testNetwork.cpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/05/2018
 *
 * Information: Example of a basic spiking neural network.
 */

#include <iostream>

#include "../source/network.hpp"
#include "../source/qtDisplay.hpp"
#include "../source/spikeLogger.hpp"
#include "../source/stdp.hpp"

int main(int argc, char** argv)
{

    //  ----- INITIALISING THE NETWORK -----
	adonis_c::QtDisplay qtDisplay;
	adonis_c::SpikeLogger spikeLogger("spikeLog");
	adonis_c::Network network({&spikeLogger}, &qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	float runtime = 100;
	float timestep = 0.1;

	float decayCurrent = 10;
	float potentialDecay = 20;
	float refractoryPeriod = 3;

    int inputNeurons = 2;
    int layer1Neurons = 1;

    float weight = 1./2;

	//  ----- CREATING THE NETWORK -----
	// input neurons
	network.addLayer(0, nullptr, inputNeurons, 1, 2, decayCurrent, potentialDecay, refractoryPeriod);

	// layer 1 neurons
	network.addLayer(1, nullptr, layer1Neurons, 1, 1, decayCurrent, potentialDecay, refractoryPeriod);

    //  ----- CONNECTING THE NETWORK -----
	// input layer -> layer 1
	network.allToAll(network.getLayers()[0], network.getLayers()[1], false, weight, false, 0);

    //  ----- INJECTING SPIKES -----
	network.injectSpike(network.getNeurons()[0].prepareInitialSpike(10));
	network.injectSpike(network.getNeurons()[0].prepareInitialSpike(15));
	network.injectSpike(network.getNeurons()[0].prepareInitialSpike(40));

    //  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(runtime);
  	qtDisplay.trackNeuron(2);

    //  ----- RUNNING THE NETWORK -----
    network.run(runtime, timestep);

    //  ----- EXITING APPLICATION -----
    return 0;
}
