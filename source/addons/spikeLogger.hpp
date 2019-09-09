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
                save_file(filename, std::ios::out | std::ios::binary),
                previous_timestamp(0) {
            if (!save_file.good()) {
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
        
        void on_start(Network* network) override {
            // learning off time header
            std::array<char, 8> bytes;
            copy_to(bytes.data() + 0, network->get_learning_off_signal());
            save_file.write(bytes.data(), bytes.size());
        }
        
		void incoming_spike(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            
            // defining what to save and constraining it so that file size doesn't blow up
            std::array<char, 16> bytes;
            copy_to(bytes.data() + 0,  static_cast<int32_t>((timestamp - previous_timestamp) * 100));
            copy_to(bytes.data() + 4,  static_cast<int16_t>(s->get_delay()*100));
            copy_to(bytes.data() + 6,  static_cast<int8_t>(s->get_weight()*100));
            copy_to(bytes.data() + 7,  static_cast<int16_t>(postsynapticNeuron->get_potential() * 100));
            copy_to(bytes.data() + 9,  static_cast<int16_t>(postsynapticNeuron->get_neuron_id()));
            copy_to(bytes.data() + 11, static_cast<int8_t>(postsynapticNeuron->get_layer_id()));
            copy_to(bytes.data() + 12, static_cast<int8_t>(postsynapticNeuron->get_rf_coordinates().first));
            copy_to(bytes.data() + 13, static_cast<int8_t>(postsynapticNeuron->get_rf_coordinates().second));
            copy_to(bytes.data() + 14, static_cast<int8_t>(postsynapticNeuron->get_xy_coordinates().first));
            copy_to(bytes.data() + 15, static_cast<int8_t>(postsynapticNeuron->get_xy_coordinates().second));
            
            // saving to file
			save_file.write(bytes.data(), bytes.size());
            
            // changing the previoud timestamp
            previous_timestamp = timestamp;
        }
		
        void on_predict(Network* network) override {
            previous_timestamp = 0;
        }
        
		void neuron_fired(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            
            // defining what to save and constraining it so that file size doesn't blow up
            std::array<char, 16> bytes;
            copy_to(bytes.data() + 0,  static_cast<int32_t>((timestamp - previous_timestamp) * 100));
            copy_to(bytes.data() + 4,  static_cast<int16_t>(s->get_delay()*100));
            copy_to(bytes.data() + 6,  static_cast<int8_t>(s->get_weight()*100));
            copy_to(bytes.data() + 7,  static_cast<int16_t>(postsynapticNeuron->get_potential() * 100));
            copy_to(bytes.data() + 9,  static_cast<int16_t>(postsynapticNeuron->get_neuron_id()));
            copy_to(bytes.data() + 11, static_cast<int8_t>(postsynapticNeuron->get_layer_id()));
            copy_to(bytes.data() + 12, static_cast<int8_t>(postsynapticNeuron->get_rf_coordinates().first));
            copy_to(bytes.data() + 13, static_cast<int8_t>(postsynapticNeuron->get_rf_coordinates().second));
            copy_to(bytes.data() + 14, static_cast<int8_t>(postsynapticNeuron->get_xy_coordinates().first));
            copy_to(bytes.data() + 15, static_cast<int8_t>(postsynapticNeuron->get_xy_coordinates().second));
            
            // saving to file
            save_file.write(bytes.data(), bytes.size());
            
            // changing the previoud timestamp
            previous_timestamp = timestamp;
        }
		
    protected:
    	// ----- IMPLEMENTATION VARIABLES -----
        std::ofstream        save_file;
        double               previous_timestamp;
    };
}
