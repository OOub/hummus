/*
 * testNetwork.cpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 11/12/2018
 *
 * Information: Example of a basic spiking neural network.
 */

#include <iostream>

#include "../source/network.hpp"
#include "../source/qtDisplay.hpp"
#include "../source/spikeLogger.hpp"
#include "../source/rewardModulatedSTDP.hpp"


int main(int argc, char** argv)
{

    //  ----- INITIALISING THE NETWORK -----
	adonis_c::QtDisplay qtDisplay;
	adonis_c::Network network(&qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	float runtime = 100;
	float timestep = 0.1;

	//  ----- CREATING THE NETWORK -----

	// input neurons
	network.addLayer(0, {}, 1, 1, 1, false);

	// layer 1 neurons
	network.addLayer(1, {}, 2, 1, 1, false);

    //  ----- CONNECTING THE NETWORK -----
	network.allToAll(network.getLayers()[0], network.getLayers()[1], 1, 0, 10, 2);
	network.lateralInhibition(network.getLayers()[1], -1);
	
    //  ----- INJECTING SPIKES -----
	network.injectSpike(network.getNeurons()[0].prepareInitialSpike(10));
	network.injectSpike(network.getNeurons()[0].prepareInitialSpike(30));
	
    //  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(runtime);
  	qtDisplay.trackNeuron(2);

    //  ----- RUNNING THE NETWORK -----
    network.run(timestep, runtime);

    //  ----- EXITING APPLICATION -----
    return 0;
}
