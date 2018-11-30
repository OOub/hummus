/*
 * hatsNetwork.cpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 04/11/2018
 *
 * Information: spiking neural network running the n-Cars database with HATS encoded with the Intentisty-to-latency method;
 */

#include <iostream> 

#include "../source/network.hpp"
#include "../source/testOutputLogger.hpp"
#include "../source/analysis.hpp"
#include "../source/stdp.hpp"
#include "../source/supervisedReinforcement.hpp"

int main(int argc, char** argv)
{
    //  ----- INITIALISING THE NETWORK -----
	adonis_c::TestOutputLogger testOutputLogger("hatsLatency.bin");
	adonis_c::Analysis analysis("../../data/hats/latency/test_nCars_10samplePerc_1repLabel.txt");
	adonis_c::Network network({&testOutputLogger, &analysis});
	
    //  ----- NETWORK PARAMETERS -----
	
    // IDs for each layer (order is important)
    int layer0 = 0;
    int layer1 = 1;
    int layer2 = 2;
	
	float decayCurrent = 10;
	float decayPotential = 20;
	float refractoryPeriod = 3;
	bool burstingActivity = false;
	float eligibilityDecay = 20;
	
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis_c::Stdp stdp(layer0, layer1);
	adonis_c::SupervisedReinforcement supervisedReinforcement;
	
	//  ----- CREATING THE NETWORK -----
	network.addLayer(layer0, {&stdp}, 4116, 1, 1, decayCurrent, decayPotential, refractoryPeriod, burstingActivity, eligibilityDecay);
	network.addLayer(layer1, {&stdp}, 100, 1, 1, decayCurrent, decayPotential, refractoryPeriod, burstingActivity, eligibilityDecay);
	network.addLayer(layer2, {&supervisedReinforcement}, 2, 1, 1, decayCurrent, decayPotential, refractoryPeriod, burstingActivity, eligibilityDecay);
	
	network.allToAll(network.getLayers()[layer0], network.getLayers()[layer1], true, 1., true, 5);
	network.allToAll(network.getLayers()[layer1], network.getLayers()[layer2], true, 1., true, 5);
	
	//  ----- READING TRAINING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
    auto trainingData = dataParser.readTrainingData("../../data/hats/latency/train_nCars_10samplePerc_1rep.txt");
	
    //  ----- INJECTING TRAINING SPIKES -----
	network.injectSpikeFromData(&trainingData);
	
	//  ----- READING TEST DATA FROM FILE -----
	auto testingData = dataParser.readTestData(&network, "../../data/hats/latency/test_nCars_10samplePerc_1rep.txt");
	
//	//  ----- INJECTING TEST SPIKES -----
	network.injectSpikeFromData(&testingData);
	
	// ----- ADDING LABELS
	auto labels = dataParser.readLabels("../../data/hats/latency/train_nCars_10samplePerc_1repLabel.txt");
	network.addLabels(&labels);
	
    //  ----- RUNNING THE NETWORK -----
    float runtime = testingData.back().timestamp+1;
	float timestep = 1;
	
    network.run(runtime, timestep);
	analysis.accuracy();
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
