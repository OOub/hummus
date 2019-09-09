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

namespace hummus {
    
    class Synapse;
    class Neuron;
    class Network;
    
    class ClassificationLogger : public Addon {
        
    public:
    	// ----- CONSTRUCTOR AND DESTRUCTOR -----
        ClassificationLogger(std::string filename) :
                save_file(filename, std::ios::out | std::ios::binary),
                previous_timestamp(0) {
                    
            if (!save_file.good()) {
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
        
		void neuron_fired(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
			// logging only after learning is stopped
            if (!network->get_learning_status()) {
                // defining what to save and constraining it so that file size doesn't blow up
                std::array<char, 6> bytes;
                copy_to(bytes.data() + 0, static_cast<int32_t>((timestamp - previous_timestamp) * 100));
                copy_to(bytes.data() + 4, static_cast<int16_t>(postsynapticNeuron->get_neuron_id()));
                
                // saving to file
                save_file.write(bytes.data(), bytes.size());
                
                // changing the previoud timestamp
                previous_timestamp = timestamp;
			}
		}

	protected:
		// ----- IMPLEMENTATION VARIABLES -----
        std::ofstream        save_file;
        double               previous_timestamp;
	};
}
