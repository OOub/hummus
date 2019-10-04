/*
 * weight_maps.hpp
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

#include "../data_parser.hpp"

namespace hummus {
    
    class Synapse;
    class Neuron;
    class Network;
    
    class WeightMaps : public Addon {
        
    public:
    	// ----- CONSTRUCTOR AND DESTRUCTOR -----
        // constructor to log all neurons of a layer
        WeightMaps(std::string filename, std::string _trainingLabels, std::string _testLabels="") :
                save_file(filename, std::ios::out | std::ios::binary),
        		training_labels({}),
        		test_string(_testLabels),
        		test_labels({}),
        		train(true) {
					
			// opening a new binary file to save data in
            if (!save_file.good()) {
                throw std::runtime_error("the file could not be opened");
            }
			
			// reading labels
			training_labels = parser.read_txt_labels(_trainingLabels);
			training_labels.pop_front(); // remove first element which point to the start of the first pattern
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
		
		void on_predict(Network* network) override {
			if (!test_string.empty()) {
				train = false;
				test_labels = parser.read_txt_labels(test_string);
				test_labels.pop_front(); // remove first element which point to the start of the first pattern
				
				for (auto& n: neuron_mask) {
                    const int16_t bitSize = 5+1*static_cast<int16_t>(network->get_neurons()[n]->get_dendritic_tree().size());
                    std::vector<char> bytes(bitSize);
	
                    copy_to(bytes.data() + 0, static_cast<int16_t>(bitSize));
                    copy_to(bytes.data() + 2, static_cast<int16_t>(n));
                    copy_to(bytes.data() + 4, static_cast<int8_t>(network->get_neurons()[n]->get_sublayer_id()));
                    
                    int count = 5;
                    for (auto& dendrite: network->get_neurons()[n]->get_dendritic_tree()) {
                        copy_to(bytes.data() + count, static_cast<int8_t>(dendrite->get_weight()*100));
                        count += 1;
                    }
				
                    // saving to file
                    save_file.write(bytes.data(), bytes.size());
                }
			
			} else {
                if (network->get_verbose() != 0) {
					std::cout << "test data was fed into the network but a corresponding test label .txt file was not provided to the weight maps constructor. Weight maps for the test dataset won't be saved" << std::endl;
				}
			}
		}
		
		void on_completed(Network* network) override {
			for (auto& n: neuron_mask) {
                const int16_t bitSize = 5+1*static_cast<int16_t>(network->get_neurons()[n]->get_dendritic_tree().size());
				std::vector<char> bytes(bitSize);
	
				copy_to(bytes.data() + 0, static_cast<int16_t>(bitSize));
				copy_to(bytes.data() + 2, static_cast<int16_t>(n));
				copy_to(bytes.data() + 4, static_cast<int8_t>(network->get_neurons()[n]->get_sublayer_id()));
                
				int count = 5;
                for (auto& dendrite: network->get_neurons()[n]->get_dendritic_tree()) {
                    copy_to(bytes.data() + count, static_cast<int8_t>(dendrite->get_weight()*100));
					count += 1;
				}
				
				// saving to file
				save_file.write(bytes.data(), bytes.size());
			}
		}
		
        void incoming_spike(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            if (train) {
                if (!training_labels.empty() && timestamp >= training_labels.front().onset) {
                    for (auto& n: neuron_mask) {
                        const int16_t bitSize = 5+1*static_cast<int16_t>(network->get_neurons()[n]->get_dendritic_tree().size());
                        std::vector<char> bytes(bitSize);
                        
                        copy_to(bytes.data() + 0, static_cast<int16_t>(bitSize));
                        copy_to(bytes.data() + 2, static_cast<int16_t>(n));
                        copy_to(bytes.data() + 4, static_cast<int8_t>(network->get_neurons()[n]->get_sublayer_id()));
                        
                        int count = 5;
                        for (auto& dendrite: network->get_neurons()[n]->get_dendritic_tree()) {
                            copy_to(bytes.data() + count, static_cast<int8_t>(dendrite->get_weight()*100));
                            count += 1;
                        }
                        
                        // saving to file
                        save_file.write(bytes.data(), bytes.size());
                    }
                    training_labels.pop_front();
                }
            } else {
                if (!test_labels.empty() && timestamp >= test_labels.front().onset) {
                    for (auto& n: neuron_mask) {
                        const int16_t bitSize = 5+1*static_cast<int16_t>(network->get_neurons()[n]->get_dendritic_tree().size());
                        std::vector<char> bytes(bitSize);
                        
                        copy_to(bytes.data() + 0, static_cast<int16_t>(bitSize));
                        copy_to(bytes.data() + 2, static_cast<int16_t>(n));
                        copy_to(bytes.data() + 4, static_cast<int8_t>(network->get_neurons()[n]->get_sublayer_id()));
                        
                        int count = 5;
                        for (auto& dendrite: network->get_neurons()[n]->get_dendritic_tree()) {
                            copy_to(bytes.data() + count, static_cast<int8_t>(dendrite->get_weight()*100));
                            count += 1;
                        }
                        
                        // saving to file
                        save_file.write(bytes.data(), bytes.size());
                    }
                    test_labels.pop_front();
                }
            }
        }
		
	protected:
		// ----- IMPLEMENTATION VARIABLES -----
		std::ofstream        save_file;
        std::deque<label>    training_labels;
        std::deque<label>    test_labels;
        std::string          test_string;
        bool                 train;
        DataParser           parser;
	};
}
