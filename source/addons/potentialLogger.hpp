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

namespace hummus {
    
    class Synapse;
    class Neuron;
    class Network;
    
    class PotentialLogger : public Addon {
        
    public:
    	// ----- CONSTRUCTOR AND DESTRUCTOR -----
        // constructor to log all neurons of a layer
        PotentialLogger(std::string filename, bool logLearning=true) :
                save_file(filename, std::ios::out | std::ios::binary),
                log_everything(logLearning),
                previous_timestamp(0) {
            if (!save_file.good()) {
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
        
        void incoming_spike(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            if (log_everything) {
                // defining what to save and constraining it so that file size doesn't blow up
                std::array<char, 8> bytes;
                copy_to(bytes.data() + 0, static_cast<int32_t>((timestamp - previous_timestamp) * 100));
                copy_to(bytes.data() + 4, static_cast<int16_t>(postsynapticNeuron->get_potential() * 100));
                copy_to(bytes.data() + 6, static_cast<int16_t>(postsynapticNeuron->get_neuron_id()));
                
                // saving to file
                save_file.write(bytes.data(), bytes.size());
                
                // changing the previous timestamp
                previous_timestamp = timestamp;
            } else {
                // logging only after learning is stopped
                if (!network->get_learning_status()) {
                    // defining what to save and constraining it so that file size doesn't blow up
                    std::array<char, 8> bytes;
                    copy_to(bytes.data() + 0, static_cast<int32_t>((timestamp - previous_timestamp) * 100));
                    copy_to(bytes.data() + 4, static_cast<int16_t>(network->get_neurons()[s->get_postsynaptic_neuron_id()]->get_potential() * 100));
                    copy_to(bytes.data() + 6, static_cast<int16_t>(s->get_postsynaptic_neuron_id()));
                    
                    // saving to file
                    save_file.write(bytes.data(), bytes.size());
                    
                    // changing the previous timestamp
                    previous_timestamp = timestamp;
                }
            }
        }
        
        void neuron_fired(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            if (log_everything) {
                // defining what to save and constraining it so that file size doesn't blow up
                std::array<char, 8> bytes;
                copy_to(bytes.data() + 0, static_cast<int32_t>((timestamp - previous_timestamp) * 100));
                copy_to(bytes.data() + 4, static_cast<int16_t>(postsynapticNeuron->get_potential() * 100));
                copy_to(bytes.data() + 6, static_cast<int16_t>(postsynapticNeuron->get_neuron_id()));
                
                // saving to file
                save_file.write(bytes.data(), bytes.size());
                
                // changing the previous timestamp
                previous_timestamp = timestamp;
            } else {
                // logging only after learning is stopped
                if (!network->get_learning_status()) {
                    // defining what to save and constraining it so that file size doesn't blow up
                    std::array<char, 8> bytes;
                    copy_to(bytes.data() + 0, static_cast<int32_t>((timestamp - previous_timestamp) * 100));
                    copy_to(bytes.data() + 4, static_cast<int16_t>(postsynapticNeuron->get_potential() * 100));
                    copy_to(bytes.data() + 6, static_cast<int16_t>(postsynapticNeuron->get_neuron_id()));
                    
                    // saving to file
                    save_file.write(bytes.data(), bytes.size());
                    
                    // changing the previous timestamp
                    previous_timestamp = timestamp;
                }
            }
        }
        
        void timestep(double timestamp, Neuron* postsynapticNeuron, Network* network) override {
            if (log_everything) {
                // defining what to save and constraining it so that file size doesn't blow up
                std::array<char, 8> bytes;
                copy_to(bytes.data() + 0, static_cast<int32_t>((timestamp - previous_timestamp) * 100));
                copy_to(bytes.data() + 4, static_cast<int16_t>(postsynapticNeuron->get_potential() * 100));
                copy_to(bytes.data() + 6, static_cast<int16_t>(postsynapticNeuron->get_neuron_id()));
                
                // saving to file
                save_file.write(bytes.data(), bytes.size());
                
                // changing the previous timestamp
                previous_timestamp = timestamp;
            } else {
                // logging only after learning is stopped
                if (!network->get_learning_status()) {
                    // defining what to save and constraining it so that file size doesn't blow up
                    std::array<char, 8> bytes;
                    copy_to(bytes.data() + 0, static_cast<int32_t>((timestamp - previous_timestamp) * 100));
                    copy_to(bytes.data() + 4, static_cast<int16_t>(postsynapticNeuron->get_potential() * 100));
                    copy_to(bytes.data() + 6, static_cast<int16_t>(postsynapticNeuron->get_neuron_id()));
                    
                    // saving to file
                    save_file.write(bytes.data(), bytes.size());
                    
                    // changing the previous timestamp
                    previous_timestamp = timestamp;
                }
            }
        }

	protected:
		// ----- IMPLEMENTATION VARIABLES -----
        std::ofstream        save_file;
        bool                 log_everything;
        double               previous_timestamp;
	};
}
