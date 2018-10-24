/*
 * hatsNetwork.cpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 09/10/2018
 *
 * Information: Spiking neural network running with histograms of averaged time surfaces converted into spikes.
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
	
	int gridWidth = 42;
	int gridHeight = 35;
	int rfSize = 7;
	
	float decayCurrent = 10;
	float potentialDecay = 20;
	float refractoryPeriod = 3;

	float eligibilityDecay = 20;
	
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis_c::Stdp stdp(layer0, layer1);
	
	//  ----- CREATING THE NETWORK -----
	// Input layer (2D neurons)
	network.addReceptiveFields(rfSize, gridWidth, gridHeight, layer0,  nullptr, -1, decayCurrent, potentialDecay, refractoryPeriod, false, eligibilityDecay);
	
	// convolution layer
	network.addReceptiveFields(rfSize, gridWidth, gridHeight, layer1, &stdp, 1, decayCurrent, potentialDecay, refractoryPeriod, false, eligibilityDecay);
	
	// flattening layer
	network.addNeurons(layer2, &stdp, 30, decayCurrent, potentialDecay, refractoryPeriod, false, eligibilityDecay);
	
    //  ----- CONNECTING THE NETWORK -----
	// connecting input layer to convolution layer
	network.rfConnectivity(layer0, layer1, false, 1./30, false, 0);
	
	// connecting convolution layer to output layer
	network.rfConnectivity(layer1, layer2, true, 1./5, true, 20, true);
	
	// connecting output and inhibition layer
	
	//  ----- READING TRAINING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
    auto trainingData = dataParser.readTrainingData("../../data/hats/poisson/one/nCars_train_100samplePerc_100rep.txt");
	
    //  ----- INJECTING TRAINING SPIKES -----
	network.injectSpikeFromData(&trainingData);
	
	//  ----- READING TEST DATA FROM FILE -----
	auto testingData = dataParser.readTestData(&network, "../../data/hats/poisson/one/nCars_test_100samplePerc_1rep.txt");
	
	//  ----- INJECTING TEST SPIKES -----
	network.injectSpikeFromData(&testingData);
	
	// ----- ADDING LABELS
	auto labels = dataParser.readLabels("../../data/hats/poisson/one/nCars_train_100samplePerc_100repLabel.txt" , "../../data/hats/poisson/one/nCars_test_100samplePerc_1repLabel.txt");
	network.addLabels(&labels);
	
    //  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(2000);
  	qtDisplay.trackLayer(1);
  	qtDisplay.trackNeuron(1531);
	
    //  ----- RUNNING THE NETWORK -----
    float runtime = testingData.back().timestamp+1;
    std::cout << runtime << std::endl;
	float timestep = 0.1;
	
    network.run(runtime, timestep);

    //  ----- EXITING APPLICATION -----
    return 0;
}
