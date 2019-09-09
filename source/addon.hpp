/*
 * addon.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: The addon class is polymorphic class to handle add-ons. It contains a series of methods acting as messages that can be used throughout the network for different purposes
 */

#pragma once

#include "synapse.hpp"

namespace hummus {
    class Synapse;
    class Neuron;
    class Network;
    
	// polymorphic class for addons
	class Addon {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        Addon() = default;
		virtual ~Addon(){}
		
		// ----- PUBLIC METHODS -----
        
        // message that is actived before the network starts running
		virtual void on_start(Network* network){}
		
		// message that is actived before the network starts running on the test data
		virtual void on_predict(Network* network){}
		
        // message that is actived when the network finishes running
		virtual void on_completed(Network* network){}
        
        // message that is activated whenever a neuron receives a spike
		virtual void incoming_spike(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network){}
        
        // message that is activated whenever a neuron emits a spike
		virtual void neuron_fired(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network){}
        
        // message that is activated on every timestep on the synchronous network only. This allows decay equations and the GUI to keep calculating even when neurons don't receive any spikes
		virtual void timestep(double timestamp, Neuron* postsynapticNeuron, Network* network){}
        
        // message that is activated whenever a neuron wants to learn
        virtual void learn(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network){};
        
        // select which neurons the addon is active on
        virtual void activate_for(size_t neuronIdx){};
        
        // select which neurons the addon is active on
        virtual void activate_for(std::vector<size_t> neuronIdx){};
        
        template <typename T>
        static void copy_to(char* target, T t) {
            *reinterpret_cast<T*>(target) = t;
        }
        
        // ----- SETTERS AND GETTERS -----
        const std::vector<size_t>& get_mask() {
            return neuron_mask;
        }
        
        bool no_automatic_include() {
            return do_not_automatically_include;
        }
        
    protected:
        std::vector<size_t> neuron_mask;
        bool                do_not_automatically_include;
	};
}
