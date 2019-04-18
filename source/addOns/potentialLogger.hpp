/*
 * potentialLogger.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 06/02/2019
 *
 * Information: Add-on used to log the potential of specified neurons or layers of neurons at every timestep (or every spike in event-based mode) after learning is off. The logger is constrained to reduce file size
 */

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <stdexcept>
#include <algorithm>

#include "../core.hpp"
#include "spikeLogger.hpp"
#include "../dataParser.hpp"

namespace hummus {
    class PotentialLogger : public Addon {
        
    public:
    	// ----- CONSTRUCTOR AND DESTRUCTOR -----
        // constructor to log all neurons of a layer
        PotentialLogger(std::string filename) :
                saveFile(filename, std::ios::out | std::ios::binary),
                previousTimestamp(0) {
            if (!saveFile.good()) {
                throw std::runtime_error("the file could not be opened");
            }
        }
        
        virtual ~PotentialLogger(){}
        
		// ----- PUBLIC LOGGER METHODS -----
        // select one neuron to track by its index
        void activate_for(size_t neuronIdx) override {
            neuron_mask.push_back(static_cast<size_t>(neuronIdx));
        }
        
        // select multiple neurons to track by passing a vector of indices
        void activate_for(std::vector<size_t> neuronIdx) override {
            neuron_mask.insert(neuron_mask.end(), neuronIdx.begin(), neuronIdx.end());
        }
        
        void incomingSpike(double timestamp, synapse* a, Network* network) override {
            // logging only after learning is stopped
            if (!network->getLearningStatus()) {
                // defining what to save and constraining it so that file size doesn't blow up
                std::array<char, 8> bytes;
                
                if (static_cast<int32_t>((timestamp - previousTimestamp) < 0)) {
                    std::cout << timestamp << " " << previousTimestamp << std::endl;
                }
                
                SpikeLogger::copy_to(bytes.data() + 0, static_cast<int32_t>((timestamp - previousTimestamp) * 100));
                SpikeLogger::copy_to(bytes.data() + 4, static_cast<int16_t>(a->postNeuron->getPotential() * 100));
                SpikeLogger::copy_to(bytes.data() + 6, static_cast<int16_t>(a->postNeuron->getNeuronID()));
                
                // saving to file
                saveFile.write(bytes.data(), bytes.size());
                
                // changing the previous timestamp
                previousTimestamp = timestamp;
            }
        }
        
        void neuronFired(double timestamp, synapse* a, Network* network) override {
            // logging only after learning is stopped
            if (!network->getLearningStatus()) {
                // defining what to save and constraining it so that file size doesn't blow up
                std::array<char, 8> bytes;
                SpikeLogger::copy_to(bytes.data() + 0, static_cast<int32_t>((timestamp - previousTimestamp) * 100));
                SpikeLogger::copy_to(bytes.data() + 4, static_cast<int16_t>(a->postNeuron->getPotential() * 100));
                SpikeLogger::copy_to(bytes.data() + 6, static_cast<int16_t>(a->postNeuron->getNeuronID()));
                
                // saving to file
                saveFile.write(bytes.data(), bytes.size());
                
                // changing the previous timestamp
                previousTimestamp = timestamp;
            }
        }
        
        void timestep(double timestamp, Network* network, Neuron* postNeuron) override {
            // logging only after learning is stopped
            if (!network->getLearningStatus()) {
                // defining what to save and constraining it so that file size doesn't blow up
                std::array<char, 8> bytes;
                SpikeLogger::copy_to(bytes.data() + 0, static_cast<int32_t>((timestamp - previousTimestamp) * 100));
                SpikeLogger::copy_to(bytes.data() + 4, static_cast<int16_t>(postNeuron->getPotential() * 100));
                SpikeLogger::copy_to(bytes.data() + 6, static_cast<int16_t>(postNeuron->getNeuronID()));
                
                // saving to file
                saveFile.write(bytes.data(), bytes.size());
                
                // changing the previous timestamp
                previousTimestamp = timestamp;
            }
        }

	protected:
		// ----- IMPLEMENTATION VARIABLES -----
        std::ofstream        saveFile;
        double               previousTimestamp;
	};
}
