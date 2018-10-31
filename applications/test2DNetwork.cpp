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
	adonis_c::Network network({}, &qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	float runtime = 100;
	float timestep = 0.1;

	//  ----- CREATING THE NETWORK -----
	network.add2dLayer(0, 4, 8, 8, nullptr, 1);
	network.add2dLayer(1, 4, 8, 8, nullptr, 1, 1);
	network.convolution(network.getLayers()[0], network.getLayers()[1], false, 1./2, false, 0);

    //  ----- INJECTING SPIKES -----
	network.injectSpike(network.getNeurons()[0].prepareInitialSpike(10));
	network.injectSpike(network.getNeurons()[0].prepareInitialSpike(15));
	network.injectSpike(network.getNeurons()[0].prepareInitialSpike(40));

    //  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(runtime);
  	qtDisplay.trackNeuron(64);

    //  ----- RUNNING THE NETWORK -----
    network.run(runtime, timestep);

    //  ----- EXITING APPLICATION -----
    return 0;
}
