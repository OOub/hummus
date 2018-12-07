/*
 * cardsClassification.cpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 05/12/2018
 *
 * Information: Spiking neural network classifying the poker-DVS dataset
 */

#include <iostream>

#include "../source/network.hpp"
#include "../source/analysis.hpp"
#include "../source/qtDisplay.hpp"
#include "../source/testOutputLogger.hpp"
#include "../source/supervisedReinforcement.hpp"
#include "../source/stdp.hpp"

int main(int argc, char** argv)
{
    //  ----- INITIALISING THE NETWORK -----
	adonis_c::QtDisplay qtDisplay;
	adonis_c::Analysis analysis("../../data/cards/test_pip4_rep10_jitter0Label.txt");
	adonis_c::TestOutputLogger testOutputLogger("cardsClassification.bin");
	adonis_c::Network network({&testOutputLogger, &analysis}, &qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	
    // IDs for each layer (order is important)
    int layer0 = 0;
    int layer1 = 1;
    int layer2 = 2;
	
	int gridWidth = 24;
	int gridHeight = 24;
	int rfSize = 4;
	
	float decayCurrent = 10;
	float decayPotential = 20;
	float refractoryPeriod = 3;
	bool  burstingActivity = false;
	float eligibilityDecay = 20;
	
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis_c::Stdp stdp(layer0, layer1);
	adonis_c::SupervisedReinforcement supervisedReinforcement;
	
	//  ----- CREATING THE NETWORK -----
	network.add2dLayer(layer0, rfSize, gridWidth, gridHeight, {}, 1, -1, false, decayCurrent, decayPotential, refractoryPeriod, burstingActivity, eligibilityDecay);
	network.add2dLayer(layer1, rfSize, gridWidth, gridHeight, {}, 1, 1, false, decayCurrent, decayPotential, refractoryPeriod, burstingActivity, eligibilityDecay);
	network.addLayer(layer2, {}, 2, 1, 1, decayCurrent, decayPotential, 1000, burstingActivity, eligibilityDecay);

	network.convolution(network.getLayers()[layer0], network.getLayers()[layer1], true, 1., true, 20);
	network.allToAll(network.getLayers()[layer1], network.getLayers()[layer2], true, 1., true, 20);
	
	//  ----- READING TRAINING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
    auto trainingData = dataParser.readTrainingData("../../data/cards/train_pip4_rep10_jitter0.txt");
	
    //  ----- INJECTING TRAINING SPIKES -----
	network.injectSpikeFromData(&trainingData);
	
	//  ----- READING TEST DATA FROM FILE -----
	auto testingData = dataParser.readTestData(&network, "../../data/cards/test_pip4_rep10_jitter0.txt");
	
	//  ----- INJECTING TEST SPIKES -----
	network.injectSpikeFromData(&testingData);
	
	// ----- ADDING LABELS
	auto labels = dataParser.readLabels("../../data/cards/train_pip4_rep10_jitter0Label.txt");
	network.addLabels(&labels);
	
	//  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(5000);
  	qtDisplay.trackLayer(2);
	qtDisplay.trackNeuron(network.getNeurons().back().getNeuronID());
	
    //  ----- RUNNING THE NETWORK -----
    float runtime = testingData.back().timestamp+1000;
	float timestep = 0.1;
	
    network.run(runtime, timestep);
	analysis.accuracy();
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
