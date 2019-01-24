/*
 * predictionLogger.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 20/11/2018
 *
 * Information: Add-on used to log the spikes from the output layer when the learning is off
 */

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <stdexcept>

#include "../core.hpp"
#include "spikeLogger.hpp"
#include "../dataParser.hpp"

namespace adonis
{
    class PredictionLogger : public AddOn
    {
    public:
    	// ----- CONSTRUCTOR -----
        PredictionLogger(std::string filename)
        {
            saveFile.open(filename, std::ios::out | std::ios::binary);
            if (!saveFile.good())
            {
                throw std::runtime_error("the file could not be opened");
            }
        }
		
		// ----- PUBLIC LOGGER METHODS -----
		void neuronFired(double timestamp, axon* a, Network* network) override
		{
			// logging only after learning is stopped
			if (!network->getLearningStatus())
			{
				// restrict only to the output layer
				if (a->postNeuron->getLayerID() == network->getLayers().back().ID)
				{	
					std::array<char, 12> bytes;
					SpikeLogger::copy_to(bytes.data() + 0, timestamp);
					SpikeLogger::copy_to(bytes.data() + 8, a->preNeuron ? a->preNeuron->getNeuronID() : -1);
					SpikeLogger::copy_to(bytes.data() + 10, a->postNeuron->getNeuronID());
					saveFile.write(bytes.data(), bytes.size());
				}
			}
		}

	protected:
		// ----- IMPLEMENTATION VARIABLES -----
        std::ofstream saveFile;
	};
}
