/*
 * testNetwork.cpp
 * Adonis_t - clock-driven spiking neural network simulator
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
#include "../source/learningLogger.hpp"

int main(int argc, char** argv)
{

    //  ----- INITIALISING THE NETWORK -----
	adonis_t::QtDisplay qtDisplay;
	adonis_t::SpikeLogger spikeLogger("spikeLog");
	adonis_t::LearningLogger learningLogger("learningLog");
	adonis_t::Network network({&spikeLogger, &learningLogger}, &qtDisplay);

    //  ----- NETWORK PARAMETERS -----
	float runtime = 100;
	float timestep = 0.1;
	
	float decayCurrent = 10;
	float potentialDecay = 20;
	float refractoryPeriod = 3;
	
    int inputNeurons = 3;
    int layer1Neurons = 1;
	
    float weight = 19e-10/2;
	
	//  ----- CREATING THE NETWORK -----
	// input neurons
	network.addNeurons(0, adonis_t::learningMode::noLearning, inputNeurons, decayCurrent, potentialDecay, refractoryPeriod);

	// layer 1 neurons
	network.addNeurons(1, adonis_t::learningMode::myelinPlasticityNoReinforcement, layer1Neurons, decayCurrent, potentialDecay, refractoryPeriod);


    //  ----- CONNECTING THE NETWORK -----
	// input layer -> layer 1
	network.allToAllConnectivity(&network.getNeuronPopulations()[0].rfNeurons, &network.getNeuronPopulations()[1].rfNeurons, false, weight, false, 0);

    //  ----- INJECTING SPIKES -----
	network.injectSpike(network.getNeuronPopulations()[0].rfNeurons[0].prepareInitialSpike(10));
	network.injectSpike(network.getNeuronPopulations()[0].rfNeurons[1].prepareInitialSpike(13));
	network.injectSpike(network.getNeuronPopulations()[0].rfNeurons[2].prepareInitialSpike(30));

	// method to turn off learning. this is useful in case we want to cross-validate or test our network
//	network.turnOffLearning(20);
	
    //  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(runtime);
  	qtDisplay.trackNeuron(3);
	
    //  ----- RUNNING THE NETWORK -----
    network.run(runtime, timestep);

    //  ----- EXITING APPLICATION -----
    return 0;
}
