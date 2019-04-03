/*
 * myelinPlasticityLogger.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/06/2018
 *
 * Information: Add-on used to write the learning rule's output into a binary file; In other words, which neurons are being modified at each learning epoch. The logger is constrained to reduce file size
 */

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <stdexcept>

#include "../core.hpp"
#include "spikeLogger.hpp"

namespace hummus {
    class MyelinPlasticityLogger : public AddOn {
        
    public:
    	// ----- CONSTRUCTOR -----
        MyelinPlasticityLogger(std::string filename) :
                previousTimestamp(0) {
                    
            saveFile.open(filename, std::ios::out | std::ios::binary);
            if (!saveFile.good()) {
                throw std::runtime_error("the file could not be opened");
            }
        }

		// ----- PUBLIC LOGGER METHODS -----
        
		void myelinPlasticityEvent(double timestamp, Network* network, Neuron* postNeuron, const std::vector<double>& timeDifferences, const std::vector<std::vector<int>>& plasticNeurons) {
            
            // defining what to save and constraining it so that file size doesn't blow up
			const int16_t bitSize = 11+4*timeDifferences.size()+4*plasticNeurons[0].size();
			std::vector<char> bytes(bitSize);
			SpikeLogger::copy_to(bytes.data() + 0, static_cast<int16_t>(bitSize));
			SpikeLogger::copy_to(bytes.data() + 2, static_cast<int32_t>((timestamp - previousTimestamp) * 100));
			SpikeLogger::copy_to(bytes.data() + 6, static_cast<int16_t>(postNeuron->getNeuronID()));
			SpikeLogger::copy_to(bytes.data() + 8, static_cast<int8_t>(postNeuron->getLayerID()));
			SpikeLogger::copy_to(bytes.data() + 9, static_cast<int8_t>(postNeuron->getRfCoordinates().first));
			SpikeLogger::copy_to(bytes.data() + 10, static_cast<int8_t>(postNeuron->getRfCoordinates().second));
			
			int count = 11;
			for (auto i=0; i<timeDifferences.size(); i++) {
				SpikeLogger::copy_to(bytes.data() + count,   static_cast<int32_t>(timeDifferences[i] * 100));
				SpikeLogger::copy_to(bytes.data() + count+4, static_cast<int8_t>(plasticNeurons[0][i]));
				SpikeLogger::copy_to(bytes.data() + count+5, static_cast<int8_t>(plasticNeurons[1][i]));
				SpikeLogger::copy_to(bytes.data() + count+6, static_cast<int8_t>(plasticNeurons[2][i]));
				SpikeLogger::copy_to(bytes.data() + count+7, static_cast<int8_t>(plasticNeurons[3][i]));
				count += 8;
			}
            
            // saving to file
            saveFile.write(bytes.data(), bytes.size());
            
            // changing the previoud timestamp
            previousTimestamp = timestamp;
		}

	protected:
		// ----- IMPLEMENTATION VARIABLES -----
        std::ofstream saveFile;
        double        previousTimestamp;
	};
}
