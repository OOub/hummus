/*
 * weightMaps.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 06/02/2019
 *
 * Information: Add-on used to log weight maps for chosen neurons via their neuronID (index in the neuron vector) at the end of every pattern (so end time of the patterns needs to be known
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
    class WeightMaps : public AddOn {
        
    public:
    	// ----- CONSTRUCTOR -----
        // constructor to log all neurons of a layer
        WeightMaps(std::string filename, std::string _trainingLabels, std::string _testLabels="") :
        		trainingLabels({}),
        		testString(_testLabels),
        		testLabels({}),
        		train(true),
                initialisationTest(false) {
					
			// opening a new binary file to save data in
            saveFile.open(filename, std::ios::out | std::ios::binary);
            if (!saveFile.good()) {
                throw std::runtime_error("the file could not be opened");
            }
			
			// reading labels
			trainingLabels = parser.readLabels(_trainingLabels);
			trainingLabels.pop_front(); // remove first element which point to the start of the first pattern
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
		
		void onPredict(Network* network) override {
			if (!testString.empty()) {
				train = false;
				testLabels = parser.readLabels(testString);
				testLabels.pop_front(); // remove first element which point to the start of the first pattern
				
				for (auto& n: neuronIDs) {
				const int16_t bitSize = 6+3*static_cast<int16_t>(network->getNeurons()[n]->getPreSynapses().size());
				std::vector<char> bytes(bitSize);
	
				SpikeLogger::copy_to(bytes.data() + 0, static_cast<int16_t>(bitSize));
				SpikeLogger::copy_to(bytes.data() + 2, static_cast<int16_t>(n));
				SpikeLogger::copy_to(bytes.data() + 4, static_cast<int8_t>(network->getNeurons()[n]->getLayerID()));
				SpikeLogger::copy_to(bytes.data() + 5, static_cast<int8_t>(network->getNeurons()[n]->getSublayerID()));
				int count = 6;
				for (auto& preSynapses: network->getNeurons()[n]->getPreSynapses()) {
					SpikeLogger::copy_to(bytes.data() + count, static_cast<int8_t>(preSynapses->weight*100));
					SpikeLogger::copy_to(bytes.data() + count+1, static_cast<int8_t>(preSynapses->preNeuron->getXYCoordinates().first));
					SpikeLogger::copy_to(bytes.data() + count+2, static_cast<int8_t>(preSynapses->preNeuron->getXYCoordinates().second));
					count += 3;
				}
				
				// saving to file
				saveFile.write(bytes.data(), bytes.size());
			}
			
			} else {
				if (network->getVerbose() != 0) {
					std::cout << "test data was fed into the network but a corresponding test label .txt file was not provided to the weight maps constructor. Weight maps for the test dataset won't be saved" << std::endl;
				}
			}
		}
		
		void onCompleted(Network* network) override {
			for (auto& n: neuronIDs) {
				const int16_t bitSize = 6+3*static_cast<int16_t>(network->getNeurons()[n]->getPreSynapses().size());
				std::vector<char> bytes(bitSize);
	
				SpikeLogger::copy_to(bytes.data() + 0, static_cast<int16_t>(bitSize));
				SpikeLogger::copy_to(bytes.data() + 2, static_cast<int16_t>(n));
				SpikeLogger::copy_to(bytes.data() + 4, static_cast<int8_t>(network->getNeurons()[n]->getLayerID()));
				SpikeLogger::copy_to(bytes.data() + 5, static_cast<int8_t>(network->getNeurons()[n]->getSublayerID()));
				int count = 6;
				for (auto& preSynapses: network->getNeurons()[n]->getPreSynapses()) {
					SpikeLogger::copy_to(bytes.data() + count, static_cast<int8_t>(preSynapses->weight*100));
					SpikeLogger::copy_to(bytes.data() + count+1, static_cast<int8_t>(preSynapses->preNeuron->getXYCoordinates().first));
					SpikeLogger::copy_to(bytes.data() + count+2, static_cast<int8_t>(preSynapses->preNeuron->getXYCoordinates().second));
					count += 3;
				}
				
				// saving to file
				saveFile.write(bytes.data(), bytes.size());
			}
		}
		
        void incomingSpike(double timestamp, synapse* a, Network* network) override {
        	if (initialisationTest) {
				if (train) {
					if (!trainingLabels.empty() && timestamp >= trainingLabels.front().onset) {
						for (auto& n: neuronIDs) {
							const int16_t bitSize = 6+3*static_cast<int16_t>(network->getNeurons()[n]->getPreSynapses().size());
							std::vector<char> bytes(bitSize);
							
							SpikeLogger::copy_to(bytes.data() + 0, static_cast<int16_t>(bitSize));
							SpikeLogger::copy_to(bytes.data() + 2, static_cast<int16_t>(n));
							SpikeLogger::copy_to(bytes.data() + 4, static_cast<int8_t>(network->getNeurons()[n]->getLayerID()));
							SpikeLogger::copy_to(bytes.data() + 5, static_cast<int8_t>(network->getNeurons()[n]->getSublayerID()));
							int count = 6;
							for (auto& preSynapses: network->getNeurons()[n]->getPreSynapses()) {
								SpikeLogger::copy_to(bytes.data() + count, static_cast<int8_t>(preSynapses->weight*100));
								SpikeLogger::copy_to(bytes.data() + count+1, static_cast<int8_t>(preSynapses->preNeuron->getXYCoordinates().first));
								SpikeLogger::copy_to(bytes.data() + count+2, static_cast<int8_t>(preSynapses->preNeuron->getXYCoordinates().second));
								count += 3;
							}
							
							// saving to file
							saveFile.write(bytes.data(), bytes.size());
						}
						trainingLabels.pop_front();
					}
				} else {
					if (!testLabels.empty() && timestamp >= testLabels.front().onset) {
						for (auto& n: neuronIDs) {
							const int16_t bitSize = 6+3*static_cast<int16_t>(network->getNeurons()[n]->getPreSynapses().size());
							std::vector<char> bytes(bitSize);
							
							SpikeLogger::copy_to(bytes.data() + 0, static_cast<int16_t>(bitSize));
							SpikeLogger::copy_to(bytes.data() + 2, static_cast<int16_t>(n));
							SpikeLogger::copy_to(bytes.data() + 4, static_cast<int8_t>(network->getNeurons()[n]->getLayerID()));
							SpikeLogger::copy_to(bytes.data() + 5, static_cast<int8_t>(network->getNeurons()[n]->getSublayerID()));
							int count = 6;
							for (auto& preSynapses: network->getNeurons()[n]->getPreSynapses()) {
								SpikeLogger::copy_to(bytes.data() + count, static_cast<int8_t>(preSynapses->weight*100));
								SpikeLogger::copy_to(bytes.data() + count+1, static_cast<int8_t>(preSynapses->preNeuron->getXYCoordinates().first));
								SpikeLogger::copy_to(bytes.data() + count+2, static_cast<int8_t>(preSynapses->preNeuron->getXYCoordinates().second));
								count += 3;
							}
							
							// saving to file
							saveFile.write(bytes.data(), bytes.size());
						}
						testLabels.pop_front();
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
        std::deque<label>    trainingLabels;
        std::deque<label>    testLabels;
        std::string          testString;
        bool                 train;
        DataParser           parser;
	};
}
