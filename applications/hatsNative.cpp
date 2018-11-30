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
#include "../source/testOutputLogger.hpp"
#include "../source/supervisedReinforcement.hpp"
#include "../source/stdp.hpp"
#include "../source/myelinPlasticity.hpp"

int main(int argc, char** argv)
{
    //  ----- INITIALISING THE NETWORK -----
	adonis_c::TestOutputLogger testOutputLogger("hatsNative.bin");
	adonis_c::Network network({&testOutputLogger});
	
    //  ----- NETWORK PARAMETERS -----
	
    // IDs for each layer (order is important)
    int layer0 = 0;
    int layer1 = 1;
    int layer2 = 2;
	
	int gridWidth = 60;
	int gridHeight = 50;
	int rfSize = 10;
	
	float decayCurrent = 10;
	float decayPotential = 20;
	float refractoryPeriod = 3;
	bool burstingActivity = false;
	float eligibilityDecay = 20;
	
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis_c::Stdp stdp(layer1, layer2);
	adonis_c::MyelinPlasticity myelinPlasticity(1, 1);
	adonis_c::SupervisedReinforcement supervisedReinforcement;
	
	//  ----- CREATING THE NETWORK -----
	network.add2dLayer(layer0, rfSize, gridWidth, gridHeight, {&stdp}, 1, -1, false, decayCurrent, decayPotential, refractoryPeriod, burstingActivity, eligibilityDecay);
	network.add2dLayer(layer1, rfSize, gridWidth, gridHeight, {&stdp}, 1, 1, false, decayCurrent+10, decayPotential+10, refractoryPeriod, burstingActivity, eligibilityDecay);
	network.addLayer(layer2, {}, 2, 1, 1, decayCurrent+20, decayPotential+20, 1200, burstingActivity, eligibilityDecay);
	
	network.allToAll(network.getLayers()[layer0], network.getLayers()[layer1], true, 1., true, 10);
	network.allToAll(network.getLayers()[layer1], network.getLayers()[layer2], true, 1., true, 20);
	
	//  ----- READING TRAINING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
    auto trainingData = dataParser.readTrainingData("../../data/hats/native/nCars_100samplePerc_10rep.txt");
	
    //  ----- INJECTING TRAINING SPIKES -----
	network.injectSpikeFromData(&trainingData);
	
	//  ----- READING TEST DATA FROM FILE -----
	auto testingData = dataParser.readTestData(&network, "../../data/hats/native/nCars_100samplePerc_1rep.txt");
	
//	//  ----- INJECTING TEST SPIKES -----
	network.injectSpikeFromData(&testingData);
	
	// ----- ADDING LABELS
	auto labels = dataParser.readLabels("../../data/hats/native/nCars_100samplePerc_10repLabel.txt");
	network.addLabels(&labels);
	
    //  ----- RUNNING THE NETWORK -----
    float runtime = testingData.back().timestamp+1;
	float timestep = 0.1;
	
    network.run(runtime, timestep);

    //  ----- EXITING APPLICATION -----
    return 0;
}
