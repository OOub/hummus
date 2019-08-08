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
    class MyelinPlasticityLogger : public Addon {
        
    public:
        // ----- CONSTRUCTOR AND DESTRUCTOR -----
        MyelinPlasticityLogger(std::string filename) :
        saveFile(filename, std::ios::out | std::ios::binary),
        previousTimestamp(0) {
            
            if (!saveFile.good()) {
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
        
        void myelinPlasticityEvent(double timestamp, Neuron* postsynapticNeuron, Network* network, const std::vector<double>& timeDifferences, const std::vector<Synapse*>& modifiedSynapses) {
            
            // defining what to save and constraining it so that file size doesn't blow up
            const int16_t bitSize = 8+4*timeDifferences.size()+5*modifiedSynapses.size();
            std::vector<char> bytes(bitSize);
            SpikeLogger::copy_to(bytes.data() + 0, static_cast<int16_t>(bitSize));
            SpikeLogger::copy_to(bytes.data() + 2, static_cast<int32_t>((timestamp - previousTimestamp) * 100));
            SpikeLogger::copy_to(bytes.data() + 6, static_cast<int16_t>(postsynapticNeuron->getNeuronID()));

            int count = 8;
            for (auto i=0; i<timeDifferences.size(); i++) {
                SpikeLogger::copy_to(bytes.data() + count,   static_cast<int32_t>(timeDifferences[i] * 100));
                SpikeLogger::copy_to(bytes.data() + count+4, static_cast<int16_t>(modifiedSynapses[i]->getPresynapticNeuronID()));
                SpikeLogger::copy_to(bytes.data() + count+6, static_cast<int16_t>(modifiedSynapses[i]->getDelay()*100));
                SpikeLogger::copy_to(bytes.data() + count+8, static_cast<int8_t>(modifiedSynapses[i]->getWeight()*100));
                count += 9;
            }
            
            // saving to file
            saveFile.write(bytes.data(), bytes.size());
            
            // changing the previoud timestamp
            previousTimestamp = timestamp;
        }
        
    protected:
        // ----- IMPLEMENTATION VARIABLES -----
        std::ofstream        saveFile;
        double               previousTimestamp;
    };
}
