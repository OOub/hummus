/*
 * potentialLogger.hpp
 * Hummus - spiking neural network simulator
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
        void neuronSelection(int _neuronID) {
            // error handling
            if (_neuronID < 0) {
                throw std::logic_error("the neuron ID cannot be less than 0");
            }
            
            initialisationTest = true;
            neuronID = _neuronID;
        }
        
        void neuronSelection(layer _layerToLog) {
            initialisationTest = true;
            neuronID = -2;
            layerID = _layerToLog.ID;
        }
        
        void incomingSpike(double timestamp, axon* a, Network* network) override {
            if (initialisationTest) {
                // logging only after learning is stopped
                if (!network->getLearningStatus()) {
                    // restrict only to the output layer
                    if (neuronID == -2) {
                        if (a->postNeuron->getLayerID() == layerID) {
                            std::array<char, 14> bytes;
                            SpikeLogger::copy_to(bytes.data() + 0, timestamp);
                            SpikeLogger::copy_to(bytes.data() + 8, a->postNeuron->getPotential());
                            SpikeLogger::copy_to(bytes.data() + 12, a->postNeuron->getNeuronID());
                            saveFile.write(bytes.data(), bytes.size());
                        }
                    } else {
                        if (a->postNeuron->getNeuronID() == neuronID) {
                            std::array<char, 14> bytes;
                            SpikeLogger::copy_to(bytes.data() + 0, timestamp);
                            SpikeLogger::copy_to(bytes.data() + 8, a->postNeuron->getPotential());
                            SpikeLogger::copy_to(bytes.data() + 12, a->postNeuron->getNeuronID());
                            saveFile.write(bytes.data(), bytes.size());
                        }
                    }
                }
            } else {
                throw std::logic_error("the method needs to be called after building all the layers of the network and before running it.");
            }
        }
        
        void neuronFired(double timestamp, axon* a, Network* network) override {
            if (initialisationTest) {
                // logging only after learning is stopped
                if (!network->getLearningStatus()) {
                    // restrict only to the output layer
                    if (neuronID == -2) {
                        if (a->postNeuron->getLayerID() == layerID) {
                            std::array<char, 14> bytes;
                            SpikeLogger::copy_to(bytes.data() + 0, timestamp);
                            SpikeLogger::copy_to(bytes.data() + 8, a->postNeuron->getPotential());
                            SpikeLogger::copy_to(bytes.data() + 12, a->postNeuron->getNeuronID());
                            saveFile.write(bytes.data(), bytes.size());
                        }
                    } else {
                        if (a->postNeuron->getNeuronID() == neuronID) {
                            std::array<char, 14> bytes;
                            SpikeLogger::copy_to(bytes.data() + 0, timestamp);
                            SpikeLogger::copy_to(bytes.data() + 8, a->postNeuron->getPotential());
                            SpikeLogger::copy_to(bytes.data() + 12, a->postNeuron->getNeuronID());
                            saveFile.write(bytes.data(), bytes.size());
                        }
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
                    if (neuronID == -2) {
                        if (postNeuron->getLayerID() == layerID) {
                            std::array<char, 14> bytes;
                            SpikeLogger::copy_to(bytes.data() + 0, timestamp);
                            SpikeLogger::copy_to(bytes.data() + 8, postNeuron->getPotential());
                            SpikeLogger::copy_to(bytes.data() + 12, postNeuron->getNeuronID());
                            saveFile.write(bytes.data(), bytes.size());
                        }
                    } else {
                        if (postNeuron->getNeuronID() == neuronID) {
                            std::array<char, 14> bytes;
                            SpikeLogger::copy_to(bytes.data() + 0, timestamp);
                            SpikeLogger::copy_to(bytes.data() + 8, postNeuron->getPotential());
                            SpikeLogger::copy_to(bytes.data() + 12, postNeuron->getNeuronID());
                            saveFile.write(bytes.data(), bytes.size());
                        }
                    }
                }
            } else {
                throw std::logic_error("the method needs to be called after building all the layers of the network and before running it.");
            }
        }

	protected:
		// ----- IMPLEMENTATION VARIABLES -----
        std::ofstream        saveFile;
        int                  layerID;
        int                  neuronID;
        bool                 initialisationTest;
	};
}
