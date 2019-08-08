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
    class ClassificationLogger : public Addon {
        
    public:
    	// ----- CONSTRUCTOR AND DESTRUCTOR -----
        ClassificationLogger(std::string filename) :
                saveFile(filename, std::ios::out | std::ios::binary),
                previousTimestamp(0) {
                    
            if (!saveFile.good()) {
                throw std::runtime_error("the file could not be opened");
            }
        }
		
        virtual ~ClassificationLogger(){}
        
		// ----- PUBLIC LOGGER METHODS -----
        // select one neuron to track by its index
        void activate_for(size_t neuronIdx) override {
            neuron_mask.push_back(static_cast<size_t>(neuronIdx));
        }
        
        // select multiple neurons to track by passing a vector of indices
        void activate_for(std::vector<size_t> neuronIdx) override {
            neuron_mask.insert(neuron_mask.end(), neuronIdx.begin(), neuronIdx.end());
        }
        
		void neuronFired(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
			// logging only after learning is stopped
			if (!network->getLearningStatus()) {
                // defining what to save and constraining it so that file size doesn't blow up
                std::array<char, 6> bytes;
                SpikeLogger::copy_to(bytes.data() + 0, static_cast<int32_t>((timestamp - previousTimestamp) * 100));
                SpikeLogger::copy_to(bytes.data() + 4, static_cast<int16_t>(postsynapticNeuron->getNeuronID()));
                
                // saving to file
                saveFile.write(bytes.data(), bytes.size());
                
                // changing the previoud timestamp
                previousTimestamp = timestamp;
			}
		}

	protected:
		// ----- IMPLEMENTATION VARIABLES -----
        std::ofstream        saveFile;
        double               previousTimestamp;
	};
}
