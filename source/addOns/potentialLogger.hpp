/*
 * potentialLogger.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 06/02/2019
 *
 * Information: Add-on used to log the potential of specified neurons or layers of neurons at every timestep after learning is off
 * works only in clock-based
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

namespace adonis {
    class PotentialLogger : public AddOn {
        
    public:
    	// ----- CONSTRUCTOR -----
        PotentialLogger(std::string filename) {
            saveFile.open(filename, std::ios::out | std::ios::binary);
            if (!saveFile.good()) {
                throw std::runtime_error("the file could not be opened");
            }
        }
		
		// ----- PUBLIC LOGGER METHODS -----
        void incomingSpike(double timestamp, axon* a, Network* network) override {
            // logging only after learning is stopped
            if (!network->getLearningStatus()) {
                // restrict only to the output layer
                if (a->postNeuron->getLayerID() == network->getLayers().back().ID) {
                    std::array<char, 14> bytes;
                    SpikeLogger::copy_to(bytes.data() + 0, timestamp);
                    SpikeLogger::copy_to(bytes.data() + 8, a->postNeuron->getPotential());
                    SpikeLogger::copy_to(bytes.data() + 12, a->postNeuron->getNeuronID());
                    saveFile.write(bytes.data(), bytes.size());
                }
            }
        }
        
        void neuronFired(double timestamp, axon* a, Network* network) override {
            // logging only after learning is stopped
            if (!network->getLearningStatus()) {
                // restrict only to the output layer
                if (a->postNeuron->getLayerID() == network->getLayers().back().ID) {
                    std::array<char, 14> bytes;
                    SpikeLogger::copy_to(bytes.data() + 0, timestamp);
                    SpikeLogger::copy_to(bytes.data() + 8, a->postNeuron->getPotential());
                    SpikeLogger::copy_to(bytes.data() + 12, a->postNeuron->getNeuronID());
                    saveFile.write(bytes.data(), bytes.size());
                }
            }
        }
        
        void timestep(double timestamp, Network* network, Neuron* postNeuron) override {
            // logging only after learning is stopped
            if (!network->getLearningStatus()) {
                // restrict only to the output layer
                if (postNeuron->getNeuronID() == network->getLayers().back().ID) {
                    std::array<char, 14> bytes;
                    SpikeLogger::copy_to(bytes.data() + 0, timestamp);
                    SpikeLogger::copy_to(bytes.data() + 8, postNeuron->getPotential());
                    SpikeLogger::copy_to(bytes.data() + 12, postNeuron->getNeuronID());
                    saveFile.write(bytes.data(), bytes.size());
                }
            }
        }

	protected:
		// ----- IMPLEMENTATION VARIABLES -----
        std::ofstream        saveFile;
	};
}
