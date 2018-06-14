/*
 * learningLogger.hpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/06/2018
 *
 * Information: Add-on to the Network class, used to write the learning rule's output into a log binary file.
 */

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <stdexcept>

#include "network.hpp"
#include "logger.hpp"

namespace baal
{
    class LearningLogger : public NetworkDelegate
    {
    public:
    	// ----- CONSTRUCTOR -----
        LearningLogger(std::string filename)
        {
            saveFile.open(filename, std::ios::out | std::ios::binary);
            if (!saveFile.good())
            {
                throw std::runtime_error("the file could not be opened");
            }
        }

	// ----- PUBLIC LOGGER METHODS -----
	Mode getMode() const override
	{
		return NetworkDelegate::Mode::learningLogger;
	}

	void getArrivingSpike(double timestamp, projection* p, bool spiked, bool empty, Network* network, Neuron* postNeuron, const std::vector<double>& timeDifferences, const std::vector<std::vector<int16_t>>& plasticNeurons) override
	{
		const int64_t bitSize = 22+8*timeDifferences.size()+4*plasticNeurons[0].size();
		std::vector<char> bytes(bitSize);
		Logger::copy_to(bytes.data() + 0, bitSize);
		Logger::copy_to(bytes.data() + 8, timestamp);
		Logger::copy_to(bytes.data() + 16, postNeuron->getNeuronID());
		Logger::copy_to(bytes.data() + 18, postNeuron->getLayerID());
		Logger::copy_to(bytes.data() + 20, postNeuron->getRFID());
		
		int count = 22;
		for (auto i=0; i<timeDifferences.size(); i++)
		{
			Logger::copy_to(bytes.data() + count, timeDifferences[i]);
			Logger::copy_to(bytes.data() + count+8, plasticNeurons[0][i]);
			Logger::copy_to(bytes.data() + count+10, plasticNeurons[1][i]);
			count += 12;
		}
		saveFile.write(bytes.data(), bytes.size());
	}

	protected:
		// ----- IMPLEMENTATION VARIABLES -----
        std::ofstream saveFile;
	};
}

