/*
 * potentialLogger.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 06/02/2019
 *
 * Information: Add-on used to log the potential of specified neurons or layers of neurons at every timestep after learning is off
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
    class PotentialLogger : public AddOn {
        
    public:
    	// ----- CONSTRUCTOR -----
        // constructor to log all neurons of a layer
        PotentialLogger(std::string filename) :
                initialisationTest(false) {
            saveFile.open(filename, std::ios::out | std::ios::binary);
            if (!saveFile.good()) {
                throw std::runtime_error("the file could not be opened");
            }
        }
        
		// ----- PUBLIC LOGGER METHODS -----
        // select one neuron to track by its index
        void neuronSelection(int _neuronID) {
            // error handling
            if (_neuronID < 0) {
                throw std::logic_error("the neuron IDs cannot be less than 0");
            } else {
                neuronIDs.push_back(static_cast<size_t>(_neuronID));
            }
            
            initialisationTest = true;
        }
        
        // select multiple neurons to track by passing a vector of indices
        void neuronSelection(std::vector<int> _neuronIDs) {
            // error handling
            for (auto nID: _neuronIDs) {
                if (nID < 0) {
                    throw std::logic_error("the neuron IDs cannot be less than 0");
                } else {
                    neuronIDs.push_back(static_cast<size_t>(nID));
                }
            }
            
            initialisationTest = true;
        }
        
        // select a whole layer to track
        void neuronSelection(layer _layer) {
            initialisationTest = true;
            neuronIDs = _layer.neurons;
        }
        
        void incomingSpike(double timestamp, synapse* a, Network* network) override {
            if (initialisationTest) {
                // logging only after learning is stopped
                if (!network->getLearningStatus()) {
                    // restrict only to the output layer
                    if (std::find(neuronIDs.begin(), neuronIDs.end(), static_cast<size_t>(a->postNeuron->getNeuronID())) != neuronIDs.end()) {
                        std::array<char, 14> bytes;
                        SpikeLogger::copy_to(bytes.data() + 0, timestamp);
                        SpikeLogger::copy_to(bytes.data() + 8, a->postNeuron->getPotential());
                        SpikeLogger::copy_to(bytes.data() + 12, a->postNeuron->getNeuronID());
                        saveFile.write(bytes.data(), bytes.size());
                    }
                }
            } else {
                throw std::logic_error("the method needs to be called after building all the layers of the network and before running it.");
            }
        }
        
        void neuronFired(double timestamp, synapse* a, Network* network) override {
            if (initialisationTest) {
                // logging only after learning is stopped
                if (!network->getLearningStatus()) {
                    // restrict only to the output layer
                    if (std::find(neuronIDs.begin(), neuronIDs.end(), static_cast<size_t>(a->postNeuron->getNeuronID())) != neuronIDs.end()) {
                        std::array<char, 14> bytes;
                        SpikeLogger::copy_to(bytes.data() + 0, timestamp);
                        SpikeLogger::copy_to(bytes.data() + 8, a->postNeuron->getPotential());
                        SpikeLogger::copy_to(bytes.data() + 12, a->postNeuron->getNeuronID());
                        saveFile.write(bytes.data(), bytes.size());
                    }
                }
            } else {
                throw std::logic_error("the method needs to be called after building all the layers of the network and before running it.");
            }
        }
        
        void timestep(double timestamp, Network* network, Neuron* postNeuron) override {
            if (initialisationTest) {
                // logging only after learning is stopped
                if (!network->getLearningStatus()) {
                    // restrict only to the output layer
                    if (std::find(neuronIDs.begin(), neuronIDs.end(), static_cast<size_t>(postNeuron->getNeuronID())) != neuronIDs.end()) {
                        std::array<char, 14> bytes;
                        SpikeLogger::copy_to(bytes.data() + 0, timestamp);
                        SpikeLogger::copy_to(bytes.data() + 8, postNeuron->getPotential());
                        SpikeLogger::copy_to(bytes.data() + 12, postNeuron->getNeuronID());
                        saveFile.write(bytes.data(), bytes.size());
                    }
                }
            } else {
                throw std::logic_error("the method needs to be called after building all the layers of the network and before running it.");
            }
        }

	protected:
		// ----- IMPLEMENTATION VARIABLES -----
        std::ofstream        saveFile;
        std::vector<size_t>  neuronIDs;
        bool                 initialisationTest;
	};
}
