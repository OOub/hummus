/*
 * ATISNetwork.cpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/05/2018
 *
 * Information: Example of a spiking neural network using receptive fields for the pip card task.
 */

#include <iostream>

#include "../source/dataParser.hpp"
#include "../source/network.hpp"
#include "../source/display.hpp"
#include "../source/logger.hpp"

int main(int argc, char** argv)
{
	//  ----- READING DATA FROM FILE -----
	baal::DataParser dataParser;
	auto data = dataParser.readData("../../data/pip/1rec_1pip/1pip_1type_200reps.txt");
	
	//  ----- INITIALISING THE NETWORK -----
	baal::Network network;
	
	//  ----- NETWORK PARAMETERS -----
	float runtime = data.back().timestamp+1;
	float timestep = 0.1;
	int neuronsPerReceptiveField = 5;
	int imageSize = 24;
	
	//  ----- RECEPTIVE FIELDS CONNECTIVITY -----
	// input layer with 36 receptive fields
	int x = 0;
	int y = 0;
	for (auto i=0; i<std::pow(imageSize/4,2); i++)
	{
		if (i % (imageSize/4) == 0 && i != 0)
		{
			x++;
			y = 0;
		}
		network.addNeurons(neuronsPerReceptiveField, 0, x, y);
		y++;
	}
	
	// intermediate layer with 4 receptive fields
	x = 0;
	y = 0;
	for (auto i=0; i<4; i++)
	{
		if (i % 2 == 0 && i != 0)
		{
			x++;
			y = 0;
		}
		network.addNeurons(neuronsPerReceptiveField, 1, x, y);
		y++;
	}
	
	// output layer with 1 receptive field
	network.addNeurons(neuronsPerReceptiveField, 2);
	
	// connecting the input layer to the first layer
	for (auto i=0; i<std::pow(imageSize/4,2); i++)
	{
		if (network.getNeuronPopulations()[i][0].getX() < 12 && network.getNeuronPopulations()[i][0].getY() < 12)
		{
			network.allToallConnectivity(&network.getNeuronPopulations()[i], &network.getNeuronPopulations()[36], false, 1, false);
		}
		else if (network.getNeuronPopulations()[i][0].getX() >= 12 && network.getNeuronPopulations()[i][0].getX() <= 24 && network.getNeuronPopulations()[i][0].getY() < 12)
		{
			network.allToallConnectivity(&network.getNeuronPopulations()[i], &network.getNeuronPopulations()[37], false, 1, false);
		}
		else if (network.getNeuronPopulations()[i][0].getX() < 12 && network.getNeuronPopulations()[i][0].getY() >= 12 && network.getNeuronPopulations()[i][0].getY() <= 24)
		{
			network.allToallConnectivity(&network.getNeuronPopulations()[i], &network.getNeuronPopulations()[38], false, 1, false);
		}
		else if (network.getNeuronPopulations()[i][0].getX() >= 12 && network.getNeuronPopulations()[i][0].getX() <= 24 && network.getNeuronPopulations()[i][0].getY() >= 12 && network.getNeuronPopulations()[i][0].getY() <= 24)
		{
			network.allToallConnectivity(&network.getNeuronPopulations()[i], &network.getNeuronPopulations()[39], false, 1, false);
		}
	}
	
	// connecting the first layer to the output layer
	for (auto i=std::pow(imageSize/4,2); i<network.getNeuronPopulations().size()-1; i++)
	{
		network.allToallConnectivity(&network.getNeuronPopulations()[i], &network.getNeuronPopulations()[40], false, 1, false);
	}
	
	//  ----- INJECTING SPIKES -----
	for (auto i=0; i<data.size(); i++)
	{
		
	}
	
	//  ----- RUNNING THE NETWORK -----
	network.run(runtime, timestep);
	
	//  ----- EXITING APPLICATION -----
	return 0;
}
