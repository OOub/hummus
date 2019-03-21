/*
 * classificationLogger.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 20/11/2018
 *
 * Information: Add-on used to log the spikes from the output layer when the learning is off. The logger is constrained to reduce file size
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

namespace hummus {
    class ClassificationLogger : public AddOn {
        
    public:
    	// ----- CONSTRUCTOR -----
        ClassificationLogger(std::string filename) :
                previousTimestamp(0) {
                    
            saveFile.open(filename, std::ios::out | std::ios::binary);
            if (!saveFile.good()) {
                throw std::runtime_error("the file could not be opened");
            }
        }
		
		// ----- PUBLIC LOGGER METHODS -----
		void neuronFired(double timestamp, synapse* a, Network* network) override {
			// logging only after learning is stopped
			if (!network->getLearningStatus()) {
				// restrict only to the output layer
				if (a->postNeuron->getLayerID() == network->getLayers().back().ID) {
                    
                    // defining what to save and constraining it so that file size doesn't blow up
					std::array<char, 4> bytes;
					SpikeLogger::copy_to(bytes.data() + 0, static_cast<int16_t>((timestamp - previousTimestamp) * 100));
					SpikeLogger::copy_to(bytes.data() + 2, static_cast<int16_t>(a->postNeuron->getNeuronID()));
                    
                    // saving to file
                    saveFile.write(bytes.data(), bytes.size());
                    
                    // changing the previoud timestamp
                    previousTimestamp = timestamp;
				}
			}
		}

	protected:
		// ----- IMPLEMENTATION VARIABLES -----
        std::ofstream saveFile;
        double        previousTimestamp;
	};
}
