/*
 * weight_maps.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 06/02/2019
 *
 * Information: Add-on used to log weight maps for chosen neurons via their neuronID (index in the neuron vector) at the end of every pattern. works only in coordination with es run methods
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
        WeightMaps(std::string filename, int _step=1) :
            save_file(filename, std::ios::out | std::ios::binary),
            step(_step),
            step_couter(1) {
            
            // error handling
            if (step == 0) {
                throw std::logic_error("the step is necessarily > 0");
            }
                
            // opening a new binary file to save data in
            if (!save_file.good()) {
                throw std::runtime_error("the file could not be opened");
            }
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
		
        void on_pattern_end(Network* network) override {
            if (step_couter % step == 0) {
                for (auto& n: neuron_mask) {
                    const int16_t bitSize = 4+8*static_cast<int16_t>(network->get_neurons()[n]->get_dendritic_tree().size());
                    std::vector<char> bytes(bitSize);

                    copy_to(bytes.data() + 0, static_cast<int16_t>(bitSize));
                    copy_to(bytes.data() + 2, static_cast<int16_t>(n));

                    int count = 4;
                    for (auto& dendrite: network->get_neurons()[n]->get_dendritic_tree()) {
                        copy_to(bytes.data() + count, static_cast<double>(dendrite->get_weight()));
                        count += 8;
                    }

                    // saving to file
                    save_file.write(bytes.data(), bytes.size());
                }
            }
            step_couter++;
        }
		
	protected:
		// ----- IMPLEMENTATION VARIABLES -----
		std::ofstream        save_file;
        int                  step;
        int                  step_couter;
	};
}
