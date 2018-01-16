/*
 * ATISNetwork.cpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 16/01/2018
 *
 * Information: Example of a basic spiking neural network.
 */

#include <iostream>

#include "../source/dataParser.hpp"
#include "../source/network.hpp"
#include "../source/display.hpp"
#include "../source/logger.hpp"

int main(int argc, char** argv)
{
//  ----- READING DATA FROM FILE -----
	int repeatsInTeacher = 8000;
	baal::DataParser dataParser;
	
	auto data = dataParser.read1D("../../data/pip/1rec_4pips/4pips_1type_2000reps.txt");
	
	auto teacher = dataParser.read1D("../../data/pip/1rec_4pips/teacher4pips_1type_2000reps.txt");

	for (auto idx=0; idx<teacher.size(); idx++)
	{
		teacher[idx].resize(repeatsInTeacher);
	}
	
//  ----- NETWORK PARAMETERS -----
	baal::Display network;

//  ----- INITIALISING THE NETWORK -----
	float runtime = data[0].back()+100;
	float timestep = 0.1;

	float decayCurrent = 10;
	float potentialDecay = 20;
	float refractoryPeriod = 3;
    float efficacyDecay = 1000;
    float efficacy = 1;

    int inputNeurons = 788;
    int layer1Neurons = 788;

    float weight = 19e-10/800;

	network.addNeurons(inputNeurons, decayCurrent, potentialDecay, refractoryPeriod, efficacyDecay, efficacy);
	network.addNeurons(layer1Neurons, decayCurrent, potentialDecay, refractoryPeriod, efficacyDecay, efficacy);

	network.allToallConnectivity(&network.getNeuronPopulations()[0], &network.getNeuronPopulations()[1], false, weight, true, 20);

	// injecting spikes in the input layer
	for (auto idx=0; idx<data[0].size(); idx++)
	{
		network.injectSpike(network.getNeuronPopulations()[0][data[1][idx]].prepareInitialSpike(data[0][idx]));
    }

	// injecting the teacher signal for supervised threshold learning
  	network.injectTeacher(&teacher);

//  ----- DISPLAY SETTINGS -----
	network.useHardwareAcceleration(true);
	network.setTimeWindow(5000);
	network.setOutputMinY(layer1Neurons);
	network.trackNeuron(787);

//  ----- RUNNING THE NETWORK -----
    int errorCode = network.run(runtime, timestep);

//  ----- EXITING APPLICATION -----
    return errorCode;
    return 0;
}