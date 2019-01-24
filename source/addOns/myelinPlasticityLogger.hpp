/*
 * myelinPlasticityLogger.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/06/2018
 *
 * Information: Add-on used to write the learning rule's output into a log binary file; In other words, which neurons are being modified at each learning epoch. This can be read using the snnReader.m matlab function
 */

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <stdexcept>

#include "../core.hpp"
#include "spikeLogger.hpp"

namespace adonis
{
    class MyelinPlasticityLogger : public AddOn
    {
    public:
    	// ----- CONSTRUCTOR -----
        MyelinPlasticityLogger(std::string filename)
        {
            saveFile.open(filename, std::ios::out | std::ios::binary);
            if (!saveFile.good())
            {
                throw std::runtime_error("the file could not be opened");
            }
        }

		// ----- PUBLIC LOGGER METHODS -----
		void myelinPlasticityEvent(double timestamp, Network* network, Neuron* postNeuron, const std::vector<double>& timeDifferences, const std::vector<std::vector<int16_t>>& plasticNeurons)
		{
			const int64_t bitSize = 24+8*timeDifferences.size()+8*plasticNeurons[0].size();
			std::vector<char> bytes(bitSize);
			SpikeLogger::copy_to(bytes.data() + 0, bitSize);
			SpikeLogger::copy_to(bytes.data() + 8, timestamp);
			SpikeLogger::copy_to(bytes.data() + 16, postNeuron->getNeuronID());
			SpikeLogger::copy_to(bytes.data() + 18, postNeuron->getLayerID());
			SpikeLogger::copy_to(bytes.data() + 20, postNeuron->getRfRow());
			SpikeLogger::copy_to(bytes.data() + 22, postNeuron->getRfCol());
			
			int count = 24;
			for (auto i=0; i<timeDifferences.size(); i++)
			{
				SpikeLogger::copy_to(bytes.data() + count, timeDifferences[i]);
				SpikeLogger::copy_to(bytes.data() + count+8, plasticNeurons[0][i]);
				SpikeLogger::copy_to(bytes.data() + count+10, plasticNeurons[1][i]);
				SpikeLogger::copy_to(bytes.data() + count+12, plasticNeurons[2][i]);
				SpikeLogger::copy_to(bytes.data() + count+14, plasticNeurons[3][i]);
				count += 16;
			}
			saveFile.write(bytes.data(), bytes.size());
		}

	protected:
		// ----- IMPLEMENTATION VARIABLES -----
        std::ofstream saveFile;
	};
}
