/*
 * spikeLogger.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/06/2018
 *
 * Information: Add-on used to write the spiking neural network output into binary file. The logger is constrained to reduce file size
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
    
    class SpikeLogger : public Addon {
        
    public:
    	// ----- CONSTRUCTOR AND DESTRUCTOR -----
        SpikeLogger(std::string filename) :
                saveFile(filename, std::ios::out | std::ios::binary),
                previousTimestamp(0) {
            if (!saveFile.good()) {
                throw std::runtime_error("the file could not be opened");
            }
        }
        
        virtual ~SpikeLogger(){}
        
		// ----- PUBLIC SPIKE LOGGER METHODS -----
        // select one neuron to track by its index
        void activate_for(size_t neuronIdx) override {
            neuron_mask.push_back(static_cast<size_t>(neuronIdx));
        }
        
        // select multiple neurons to track by passing a vector of indices
        void activate_for(std::vector<size_t> neuronIdx) override {
            neuron_mask.insert(neuron_mask.end(), neuronIdx.begin(), neuronIdx.end());
        }
        
        void onStart(Network* network) override {
            // learning off time header
            std::array<char, 8> bytes;
            copy_to(bytes.data() + 0, network->getLearningOffSignal());
            saveFile.write(bytes.data(), bytes.size());
        }
        
		void incomingSpike(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            
            // defining what to save and constraining it so that file size doesn't blow up
            std::array<char, 16> bytes;
            copy_to(bytes.data() + 0,  static_cast<int32_t>((timestamp - previousTimestamp) * 100));
            copy_to(bytes.data() + 4,  static_cast<int16_t>(s->getDelay()*100));
            copy_to(bytes.data() + 6,  static_cast<int8_t>(s->getWeight()*100));
            copy_to(bytes.data() + 7,  static_cast<int16_t>(postsynapticNeuron->getPotential() * 100));
            copy_to(bytes.data() + 9,  static_cast<int16_t>(postsynapticNeuron->getNeuronID()));
            copy_to(bytes.data() + 11, static_cast<int8_t>(postsynapticNeuron->getLayerID()));
            copy_to(bytes.data() + 12, static_cast<int8_t>(postsynapticNeuron->getRfCoordinates().first));
            copy_to(bytes.data() + 13, static_cast<int8_t>(postsynapticNeuron->getRfCoordinates().second));
            copy_to(bytes.data() + 14, static_cast<int8_t>(postsynapticNeuron->getXYCoordinates().first));
            copy_to(bytes.data() + 15, static_cast<int8_t>(postsynapticNeuron->getXYCoordinates().second));
            
            // saving to file
			saveFile.write(bytes.data(), bytes.size());
            
            // changing the previoud timestamp
            previousTimestamp = timestamp;
        }
		
        void onPredict(Network* network) override {
            previousTimestamp = 0;
        }
        
		void neuronFired(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            
            // defining what to save and constraining it so that file size doesn't blow up
            std::array<char, 16> bytes;
            copy_to(bytes.data() + 0,  static_cast<int32_t>((timestamp - previousTimestamp) * 100));
            copy_to(bytes.data() + 4,  static_cast<int16_t>(s->getDelay()*100));
            copy_to(bytes.data() + 6,  static_cast<int8_t>(s->getWeight()*100));
            copy_to(bytes.data() + 7,  static_cast<int16_t>(postsynapticNeuron->getPotential() * 100));
            copy_to(bytes.data() + 9,  static_cast<int16_t>(postsynapticNeuron->getNeuronID()));
            copy_to(bytes.data() + 11, static_cast<int8_t>(postsynapticNeuron->getLayerID()));
            copy_to(bytes.data() + 12, static_cast<int8_t>(postsynapticNeuron->getRfCoordinates().first));
            copy_to(bytes.data() + 13, static_cast<int8_t>(postsynapticNeuron->getRfCoordinates().second));
            copy_to(bytes.data() + 14, static_cast<int8_t>(postsynapticNeuron->getXYCoordinates().first));
            copy_to(bytes.data() + 15, static_cast<int8_t>(postsynapticNeuron->getXYCoordinates().second));
            
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
