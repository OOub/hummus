/*
 * spikeLogger.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/06/2018
 *
 * Information: Add-on used to write the spiking neural network output into binary file.
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
        SpikeLogger(std::string filename) {
            saveFile.open(filename, std::ios::out | std::ios::binary);
            if (!saveFile.good()) {
                throw std::runtime_error("the file could not be opened");
            }
        }
		
		// ----- PUBLIC SPIKE LOGGER METHODS -----
        void onStart(Network* network) override {
            std::array<char, 8> bytes;
            copy_to(bytes.data() + 0, network->getLearningOffSignal());
            saveFile.write(bytes.data(), bytes.size());
        }
        
		void incomingSpike(double timestamp, synapse* a, Network* network) override {
			std::array<char, 34> bytes;
			copy_to(bytes.data() + 0, timestamp);
			copy_to(bytes.data() + 8, a->delay);
			copy_to(bytes.data() + 12, a->weight);
			copy_to(bytes.data() + 16, a->postNeuron->getPotential());
			copy_to(bytes.data() + 20, a->preNeuron ? a->preNeuron->getNeuronID() : -1);
			copy_to(bytes.data() + 22, a->postNeuron->getNeuronID());
			copy_to(bytes.data() + 24, a->postNeuron->getLayerID());
			copy_to(bytes.data() + 26, a->postNeuron->getRfCoordinates().first);
			copy_to(bytes.data() + 28, a->postNeuron->getRfCoordinates().second);
			copy_to(bytes.data() + 30, a->postNeuron->getXYCoordinates().first);
			copy_to(bytes.data() + 32, a->postNeuron->getXYCoordinates().second);
			saveFile.write(bytes.data(), bytes.size());
        }
		
		void neuronFired(double timestamp, synapse* a, Network* network) override {
			std::array<char, 34> bytes;
			copy_to(bytes.data() + 0, timestamp);
			copy_to(bytes.data() + 8, a->delay);
			copy_to(bytes.data() + 12, a->weight);
			copy_to(bytes.data() + 16, a->postNeuron->getPotential());
			copy_to(bytes.data() + 20, a->preNeuron ? a->preNeuron->getNeuronID() : -1);
			copy_to(bytes.data() + 22, a->postNeuron->getNeuronID());
			copy_to(bytes.data() + 24, a->postNeuron->getLayerID());
			copy_to(bytes.data() + 26, a->postNeuron->getRfCoordinates().first);
			copy_to(bytes.data() + 28, a->postNeuron->getRfCoordinates().second);
			copy_to(bytes.data() + 30, a->postNeuron->getXYCoordinates().first);
			copy_to(bytes.data() + 32, a->postNeuron->getXYCoordinates().second);
			saveFile.write(bytes.data(), bytes.size());
        }
		
		template <typename T>
		static void copy_to(char* target, T t) {
		    *reinterpret_cast<T*>(target) = t;
		}
		
    protected:
    	// ----- IMPLEMENTATION VARIABLES -----
        std::ofstream saveFile;
    };
}
