/*
 * myelin_plasticity_logger.hpp
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

namespace hummus {
    
    class Synapse;
    class Neuron;
    class Network;
    
    class MyelinPlasticityLogger : public Addon {
        
    public:
        // ----- CONSTRUCTOR AND DESTRUCTOR -----
        MyelinPlasticityLogger(std::string filename) :
        save_file(filename, std::ios::out | std::ios::binary),
        previous_timestamp(0) {
            
            if (!save_file.good()) {
                throw std::runtime_error("the file could not be opened");
            }
        }
        
        virtual ~MyelinPlasticityLogger(){}
        
        // ----- PUBLIC LOGGER METHODS -----
        // select one neuron to track by its index
        void activate_for(size_t neuronIdx) override {
            neuron_mask.push_back(static_cast<size_t>(neuronIdx));
        }
        
        // 0select multiple neurons to track by passing a vector of indices
        void activate_for(std::vector<size_t> neuronIdx) override {
            neuron_mask.insert(neuron_mask.end(), neuronIdx.begin(), neuronIdx.end());
        }
        
        void myelin_plasticity_event(double timestamp, Neuron* postsynapticNeuron, Network* network, const std::vector<float>& timeDifferences, const std::vector<Synapse*>& modifiedSynapses) {
            
            // defining what to save and constraining it so that file size doesn't blow up
            const int16_t bitSize = 8+4*timeDifferences.size()+5*modifiedSynapses.size();
            std::vector<char> bytes(bitSize);
            copy_to(bytes.data() + 0, static_cast<int16_t>(bitSize));
            copy_to(bytes.data() + 2, static_cast<int32_t>((timestamp - previous_timestamp) * 100));
            copy_to(bytes.data() + 6, static_cast<int16_t>(postsynapticNeuron->get_neuron_id()));

            int count = 8;
            for (int i=0; i<static_cast<int>(timeDifferences.size()); i++) {
                copy_to(bytes.data() + count,   static_cast<int32_t>(timeDifferences[i] * 100));
                copy_to(bytes.data() + count+4, static_cast<int16_t>(modifiedSynapses[i]->get_presynaptic_neuron_id()));
                copy_to(bytes.data() + count+6, static_cast<int16_t>(modifiedSynapses[i]->get_delay()*100));
                copy_to(bytes.data() + count+8, static_cast<int8_t>(modifiedSynapses[i]->get_weight()*100));
                count += 9;
            }
            
            // saving to file
            save_file.write(bytes.data(), bytes.size());
            
            // changing the previous timestamp
            previous_timestamp = timestamp;
        }
        
    protected:
        // ----- IMPLEMENTATION VARIABLES -----
        std::ofstream        save_file;
        double               previous_timestamp;
    };
}
