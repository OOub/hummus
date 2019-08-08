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
    class WeightMaps : public Addon {
        
    public:
    	// ----- CONSTRUCTOR AND DESTRUCTOR -----
        // constructor to log all neurons of a layer
        WeightMaps(std::string filename, std::string _trainingLabels, std::string _testLabels="") :
                saveFile(filename, std::ios::out | std::ios::binary),
        		trainingLabels({}),
        		testString(_testLabels),
        		testLabels({}),
        		train(true) {
					
			// opening a new binary file to save data in
            if (!saveFile.good()) {
                throw std::runtime_error("the file could not be opened");
            }
			
			// reading labels
			trainingLabels = parser.readLabels(_trainingLabels);
			trainingLabels.pop_front(); // remove first element which point to the start of the first pattern
        }
        
        virtual ~WeightMaps(){}
        
		// ----- PUBLIC LOGGER METHODS -----
        // select one neuron to track by its index
        void activate_for(size_t neuronIdx) override {
            neuron_mask.push_back(static_cast<size_t>(neuronIdx));
        }
        
        // select multiple neurons to track by passing a vector of indices
        void activate_for(std::vector<size_t> neuronIdx) override {
            neuron_mask.insert(neuron_mask.end(), neuronIdx.begin(), neuronIdx.end());
        }
		
		void onPredict(Network* network) override {
			if (!testString.empty()) {
				train = false;
				testLabels = parser.readLabels(testString);
				testLabels.pop_front(); // remove first element which point to the start of the first pattern
				
				for (auto& n: neuron_mask) {
                    const int16_t bitSize = 5+1*static_cast<int16_t>(network->getNeurons()[n]->getDendriticTree().size());
                    std::vector<char> bytes(bitSize);
	
                    SpikeLogger::copy_to(bytes.data() + 0, static_cast<int16_t>(bitSize));
                    SpikeLogger::copy_to(bytes.data() + 2, static_cast<int16_t>(n));
                    SpikeLogger::copy_to(bytes.data() + 4, static_cast<int8_t>(network->getNeurons()[n]->getSublayerID()));
                    
                    int count = 5;
                    for (auto& dendrite: network->getNeurons()[n]->getDendriticTree()) {
                        SpikeLogger::copy_to(bytes.data() + count, static_cast<int8_t>(dendrite->getWeight()*100));
                        count += 1;
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
			for (auto& n: neuron_mask) {
				const int16_t bitSize = 5+1*static_cast<int16_t>(network->getNeurons()[n]->getDendriticTree().size());
				std::vector<char> bytes(bitSize);
	
				SpikeLogger::copy_to(bytes.data() + 0, static_cast<int16_t>(bitSize));
				SpikeLogger::copy_to(bytes.data() + 2, static_cast<int16_t>(n));
				SpikeLogger::copy_to(bytes.data() + 4, static_cast<int8_t>(network->getNeurons()[n]->getSublayerID()));
                
				int count = 5;
				for (auto& dendrite: network->getNeurons()[n]->getDendriticTree()) {
					SpikeLogger::copy_to(bytes.data() + count, static_cast<int8_t>(dendrite->getWeight()*100));
					count += 1;
				}
				
				// saving to file
				saveFile.write(bytes.data(), bytes.size());
			}
		}
		
        void incomingSpike(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            if (train) {
                if (!trainingLabels.empty() && timestamp >= trainingLabels.front().onset) {
                    for (auto& n: neuron_mask) {
                        const int16_t bitSize = 5+1*static_cast<int16_t>(network->getNeurons()[n]->getDendriticTree().size());
                        std::vector<char> bytes(bitSize);
                        
                        SpikeLogger::copy_to(bytes.data() + 0, static_cast<int16_t>(bitSize));
                        SpikeLogger::copy_to(bytes.data() + 2, static_cast<int16_t>(n));
                        SpikeLogger::copy_to(bytes.data() + 4, static_cast<int8_t>(network->getNeurons()[n]->getSublayerID()));
                        
                        int count = 5;
                        for (auto& dendrite: network->getNeurons()[n]->getDendriticTree()) {
                            SpikeLogger::copy_to(bytes.data() + count, static_cast<int8_t>(dendrite->getWeight()*100));
                            count += 1;
                        }
                        
                        // saving to file
                        saveFile.write(bytes.data(), bytes.size());
                    }
                    trainingLabels.pop_front();
                }
            } else {
                if (!testLabels.empty() && timestamp >= testLabels.front().onset) {
                    for (auto& n: neuron_mask) {
                        const int16_t bitSize = 5+1*static_cast<int16_t>(network->getNeurons()[n]->getDendriticTree().size());
                        std::vector<char> bytes(bitSize);
                        
                        SpikeLogger::copy_to(bytes.data() + 0, static_cast<int16_t>(bitSize));
                        SpikeLogger::copy_to(bytes.data() + 2, static_cast<int16_t>(n));
                        SpikeLogger::copy_to(bytes.data() + 4, static_cast<int8_t>(network->getNeurons()[n]->getSublayerID()));
                        
                        int count = 5;
                        for (auto& dendrite: network->getNeurons()[n]->getDendriticTree()) {
                            SpikeLogger::copy_to(bytes.data() + count, static_cast<int8_t>(dendrite->getWeight()*100));
                            count += 1;
                        }
                        
                        // saving to file
                        saveFile.write(bytes.data(), bytes.size());
                    }
                    testLabels.pop_front();
                }
            }
        }
		
	protected:
		// ----- IMPLEMENTATION VARIABLES -----
		std::ofstream        saveFile;
        std::deque<label>    trainingLabels;
        std::deque<label>    testLabels;
        std::string          testString;
        bool                 train;
        DataParser           parser;
	};
}
