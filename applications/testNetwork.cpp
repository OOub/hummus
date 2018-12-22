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

int main(int argc, char** argv)
{

    //  ----- INITIALISING THE NETWORK -----
	adonis_c::QtDisplay qtDisplay;
	adonis_c::Network network(&qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	float runtime = 100;
	float timestep = 1;

	//  ----- CREATING THE NETWORK -----
	
	// creating layers of neurons
	network.addLayer({}, 5, 1, 1, false);
	network.addLayer({}, 1, 1, 1, false);

    //  ----- CONNECTING THE NETWORK -----
	network.allToAll(network.getLayers()[0], network.getLayers()[1], 1./5, 0, 0, 0);
	network.lateralInhibition(network.getLayers()[1], -1);
	
    //  ----- INJECTING SPIKES -----
	network.injectSpike(network.getNeurons()[0].prepareInitialSpike(10));
	network.injectSpike(network.getNeurons()[1].prepareInitialSpike(10));
	network.injectSpike(network.getNeurons()[2].prepareInitialSpike(10));
	network.injectSpike(network.getNeurons()[3].prepareInitialSpike(10));
	network.injectSpike(network.getNeurons()[4].prepareInitialSpike(10));

	
    //  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(runtime);
  	qtDisplay.trackNeuron(5);

    //  ----- RUNNING THE NETWORK -----
    network.run(runtime, timestep);

    //  ----- EXITING APPLICATION -----
    return 0;
}
