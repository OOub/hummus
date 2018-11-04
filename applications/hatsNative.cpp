/*
 * hatsNetwork.cpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 09/10/2018
 *
 * Information: spiking neural network running the n-Cars database
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
	adonis_c::Network network(&qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	
    // IDs for each layer (order is important)
    int layer0 = 0;
    int layer1 = 1;
    int layer2 = 2;
    int layer3 = 3;
	
	int gridWidth = 64;
	int gridHeight = 56;
	int rfSize = 4;
	
	float decayCurrent = 10;
	float decayPotential = 20;
	float refractoryPeriod = 3;
	bool burstingActivity = false;
	float eligibilityDecay = 20;
	
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis_c::Stdp stdp(layer0, layer1);
	
	//  ----- CREATING THE NETWORK -----
	network.add2dLayer(layer0, rfSize, gridWidth, gridHeight, &stdp, 1, -1, false, decayCurrent, decayPotential, refractoryPeriod, burstingActivity, eligibilityDecay);
	network.add2dLayer(layer1, rfSize, gridWidth, gridHeight, &stdp, 1, 1, false, decayCurrent, decayPotential, refractoryPeriod, burstingActivity, eligibilityDecay);
	network.add2dLayer(layer2, rfSize, gridWidth/2, gridHeight/2, nullptr, 1, 1, false, decayCurrent, decayPotential, refractoryPeriod, burstingActivity, eligibilityDecay);
	network.addLayer(layer3, nullptr, 1, 1, 1, decayCurrent, decayPotential, refractoryPeriod, burstingActivity, eligibilityDecay);
	
	network.convolution(network.getLayers()[layer0], network.getLayers()[layer1], false, 1./8, false, 0);
	network.pooling(network.getLayers()[layer1], network.getLayers()[layer2], false, 1., false, 0);
	network.allToAll(network.getLayers()[layer2], network.getLayers()[layer3], false, 1./28, false, 0);
	
	//  ----- READING TRAINING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
    auto trainingData = dataParser.readTrainingData("../../data/hats/native/nCars_native_train.txt");
	
    //  ----- INJECTING TRAINING SPIKES -----
	network.injectSpikeFromData(&trainingData);
	
	//  ----- READING TEST DATA FROM FILE -----
	auto testingData = dataParser.readTestData(&network, "../../data/hats/native/nCars_native_test_train.txt");
//	auto testingData = dataParser.readTestData(&network, "../../data/hats/native/nCars_native_test_test.txt");
	
//	//  ----- INJECTING TEST SPIKES -----
	network.injectSpikeFromData(&testingData);
	
	// ----- ADDING LABELS
	auto labels = dataParser.readLabels("" , "../../data/hats/native/nCars_native_test_train_Label.txt");
//	auto labels = dataParser.readLabels("" , "../../data/hats/native/nCars_native_test_test_Label.txt");
	network.addLabels(&labels);
	
    //  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(1000);
  	qtDisplay.trackLayer(2);
	qtDisplay.trackNeuron(network.getNeurons().back().getNeuronID());
	
    //  ----- RUNNING THE NETWORK -----
    float runtime = testingData.back().timestamp+1;
	float timestep = 0.1;
	
    network.run(runtime, timestep);

    //  ----- EXITING APPLICATION -----
    return 0;
}
