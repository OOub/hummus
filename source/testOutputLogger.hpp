/*
 * testOutputLogger.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 20/11/2018
 *
 * Information: Add-on to the Network class, used to log the spikes from the output layer during the testing phase
 */

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <stdexcept>

#include "network.hpp"
#include "spikeLogger.hpp"

namespace adonis_c
{
    class TestOutputLogger : public StandardNetworkDelegate
    {
    public:
    	// ----- CONSTRUCTOR -----
        TestOutputLogger(std::string filename)
        {
            saveFile.open(filename, std::ios::out | std::ios::binary);
            if (!saveFile.good())
            {
                throw std::runtime_error("the file could not be opened");
            }
        }
		
		void learningTurnedOff(double timestamp) override
		{
			std::array<char, 8> bytes;
			SpikeLogger::copy_to(bytes.data() + 0, timestamp);
			saveFile.write(bytes.data(), bytes.size());
		}
		
		// ----- PUBLIC LOGGER METHODS -----
		void neuronFired(double timestamp, projection* p, Network* network) override
		{
			// logging only after learning is stopped
			if (!network->getLearningStatus())
			{
				// restrict only to the output layer
				if (p->postNeuron->getLayerID() == network->getLayers().back().ID)
				{
					std::array<char, 34> bytes;
					SpikeLogger::copy_to(bytes.data() + 0, timestamp);
					SpikeLogger::copy_to(bytes.data() + 8, p->delay);
					SpikeLogger::copy_to(bytes.data() + 12, p->weight);
					SpikeLogger::copy_to(bytes.data() + 16, p->postNeuron->getPotential());
					SpikeLogger::copy_to(bytes.data() + 20, p->preNeuron ? p->preNeuron->getNeuronID() : -1);
					SpikeLogger::copy_to(bytes.data() + 22, p->postNeuron->getNeuronID());
					SpikeLogger::copy_to(bytes.data() + 24, p->postNeuron->getLayerID());
					SpikeLogger::copy_to(bytes.data() + 26, p->postNeuron->getRfRow());
					SpikeLogger::copy_to(bytes.data() + 28, p->postNeuron->getRfCol());
					SpikeLogger::copy_to(bytes.data() + 30, p->postNeuron->getX());
					SpikeLogger::copy_to(bytes.data() + 32, p->postNeuron->getY());
					saveFile.write(bytes.data(), bytes.size());
				}
			}
		}

	protected:
		// ----- IMPLEMENTATION VARIABLES -----
        std::ofstream saveFile;
	};
}
