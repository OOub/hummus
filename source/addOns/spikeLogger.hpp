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


#include "../core.hpp"

namespace hummus {
    class SpikeLogger : public AddOn {
        
    public:
    	// ----- CONSTRUCTOR -----
        SpikeLogger(std::string filename) :
                previousTimestamp(0) {
            saveFile.open(filename, std::ios::out | std::ios::binary);
            if (!saveFile.good()) {
                throw std::runtime_error("the file could not be opened");
            }
        }
		
		// ----- PUBLIC SPIKE LOGGER METHODS -----
        void onStart(Network* network) override {
            // learning off time header
            std::array<char, 8> bytes;
            copy_to(bytes.data() + 0, network->getLearningOffSignal());
            saveFile.write(bytes.data(), bytes.size());
        }
        
		void incomingSpike(double timestamp, synapse* a, Network* network) override {
            
            // defining what to save and constraining it so that file size doesn't blow up
            std::array<char, 16> bytes;
            copy_to(bytes.data() + 0,  static_cast<int32_t>((timestamp - previousTimestamp) * 100));
            copy_to(bytes.data() + 4,  static_cast<int16_t>(a->delay*100));
            copy_to(bytes.data() + 6,  static_cast<int8_t>(a->weight*100));
            copy_to(bytes.data() + 7,  static_cast<int16_t>(a->postNeuron->getPotential() * 100));
            copy_to(bytes.data() + 9,  static_cast<int16_t>(a->postNeuron->getNeuronID()));
            copy_to(bytes.data() + 11, static_cast<int8_t>(a->postNeuron->getLayerID()));
            copy_to(bytes.data() + 12, static_cast<int8_t>(a->postNeuron->getRfCoordinates().first));
            copy_to(bytes.data() + 13, static_cast<int8_t>(a->postNeuron->getRfCoordinates().second));
            copy_to(bytes.data() + 14, static_cast<int8_t>(a->postNeuron->getXYCoordinates().first));
            copy_to(bytes.data() + 15, static_cast<int8_t>(a->postNeuron->getXYCoordinates().second));
            
            // saving to file
			saveFile.write(bytes.data(), bytes.size());
            
            // changing the previoud timestamp
            previousTimestamp = timestamp;
        }
		
        void onPredict(Network* network) override {
            previousTimestamp = 0;
        }
        
		void neuronFired(double timestamp, synapse* a, Network* network) override {
            
            // defining what to save and constraining it so that file size doesn't blow up
            std::array<char, 16> bytes;
            copy_to(bytes.data() + 0,  static_cast<int32_t>((timestamp - previousTimestamp) * 100));
            copy_to(bytes.data() + 4,  static_cast<int16_t>(a->delay*100));
            copy_to(bytes.data() + 6,  static_cast<int8_t>(a->weight*100));
            copy_to(bytes.data() + 7,  static_cast<int16_t>(a->postNeuron->getPotential() * 100));
            copy_to(bytes.data() + 9,  static_cast<int16_t>(a->postNeuron->getNeuronID()));
            copy_to(bytes.data() + 11, static_cast<int8_t>(a->postNeuron->getLayerID()));
            copy_to(bytes.data() + 12, static_cast<int8_t>(a->postNeuron->getRfCoordinates().first));
            copy_to(bytes.data() + 13, static_cast<int8_t>(a->postNeuron->getRfCoordinates().second));
            copy_to(bytes.data() + 14, static_cast<int8_t>(a->postNeuron->getXYCoordinates().first));
            copy_to(bytes.data() + 15, static_cast<int8_t>(a->postNeuron->getXYCoordinates().second));
            
            // saving to file
            saveFile.write(bytes.data(), bytes.size());
            
            // changing the previoud timestamp
            previousTimestamp = timestamp;
        }
		
		template <typename T>
		static void copy_to(char* target, T t) {
		    *reinterpret_cast<T*>(target) = t;
		}
		
    protected:
    	// ----- IMPLEMENTATION VARIABLES -----
        std::ofstream saveFile;
        double        previousTimestamp;
    };
}
