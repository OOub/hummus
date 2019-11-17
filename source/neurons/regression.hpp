/*
 * regression.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: Neurons used to train a logistic regression classifier. These neurons will call decision-making neurons for class selection. only works with the es_database run method so far.
 *
 * NEURON TYPE 5 (in JSON SAVE FILE)
 */

#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <torch/torch.h> // requires the libtorch option to be ON in cmake
#include "../../third_party/json.hpp"

namespace hummus {

    class MyDataset : public torch::data::Dataset<MyDataset> {
        public:
            explicit MyDataset(torch::Tensor _data, torch::Tensor _labels) :
                data(_data),
                labels(_labels) {};

        torch::data::Example<> get(size_t index) override {
            return {data[index], labels[index]};
        }
        
        protected:
            torch::Tensor data, labels;
    };
    
    class Synapse;
    class Neuron;
    class Network;

	class Regression : public Neuron {
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        Regression(int _neuronID, int _layerID, int _sublayerID, int _rf_id,  std::pair<int, int> _xyCoordinates, std::string _classLabel="", float _learning_rate=0, float _momentum=0, float _weight_decay=0, int _epochs=10, int _batch_size=32, int _presentations_before_training=0, float _threshold=-50, float _restingPotential=-70) :
                Neuron(_neuronID, _layerID, _sublayerID, _rf_id, _xyCoordinates, 0, 200, 10, 20, _threshold, _restingPotential, _classLabel),
                learning_rate(_learning_rate),
                momentum(_momentum),
                weight_decay(_weight_decay),
                epochs(_epochs),
                batch_size(_batch_size),
                computation_layer(false),
                neuron_id_shift(0),
                number_of_output_neurons(0),
                presentations_before_training(0),
                computation_id(0) {

            // Regression neuron type = 5 for JSON save
            neuron_type = 5;
                    
            // computation or decision layer of regression
            if (_classLabel.empty()) {
                computation_layer = true;
            }
        }

		virtual ~Regression(){}

        // ----- PUBLIC REGRESSION NEURON METHODS -----
       virtual void initialisation(Network* network) override {
           // searching for addons that are relevant to this neuron. if addons do not have a mask they are automatically relevant / not filtered out
           for (auto& addon: network->get_addons()) {
               if (addon->get_mask().empty() && !addon->no_automatic_include()) {
                   add_relevant_addon(addon.get());
               } else {
                   if (auto it = std::find(addon->get_mask().begin(), addon->get_mask().end(), static_cast<size_t>(neuron_id)); it != addon->get_mask().end()) {
                       add_relevant_addon(addon.get());
                   }
               }
           }
           
           if (computation_layer) {
               auto& previous_layer_neurons = network->get_layers()[layer_id-1].neurons;
               
               // save the number of output neurons
               number_of_output_neurons = static_cast<int>(previous_layer_neurons.size());
               
               // identify the neuron shift to be able to directly access the tensor with neuron ids from the output neuron layer
               neuron_id_shift = static_cast<int>(*std::min_element(std::begin(previous_layer_neurons), std::end(previous_layer_neurons)));
               
               // initialise the x_online
               x_online = torch::zeros(number_of_output_neurons);
               
               computation_id = neuron_id;
               
           } else {
               auto& previous_layer_neurons = network->get_layers()[layer_id-2].neurons;
               number_of_output_neurons = static_cast<int>(previous_layer_neurons.size());
               computation_id = network->get_layers()[layer_id-1].neurons[0];
           }
       }
        
        virtual void update(double timestamp, Synapse* s, Network* network, float timestep, spike_type type) override {
            // none spikes are used  by the computation layer for training the logistic regression
            if (type == spike_type::none && computation_layer) {
            
                // 1. load dataset
                
                // converting label vector to tensor
                torch::Tensor X_labels = torch::from_blob(std::data(labels), {static_cast<int>(labels.size()), 1});
                
                // converting data vector to tensor
                torch::Tensor X = torch::cat(x_training, 0);
                
                // create custom dataloader
                
                // 2. create model and instantiate it
                
                // 3. instantiate optimizer
                
                // 4. instantiate loss function
                
                // 5. loop through epochs to train logistic regression on the training batches
                
            // generated spikes are used to collect features
            } else if (type == spike_type::generated && computation_layer) {
                
                if (network->get_learning_status()) {
                    // during learning only change x_online after waiting for a number of training presentations defined by the variable: presentations_before_training
                    if (network->get_presentation_counter() == presentations_before_training) {
                        x_online[s->get_presynaptic_neuron_id() - neuron_id_shift] = x_online[s->get_presynaptic_neuron_id() - neuron_id_shift] + 1;
                        labels.emplace_back(network->get_classes_map()[network->get_current_label()]);
                    }
                } else {
                    // during testing we can start saving into x_online immediately
                    x_online[s->get_presynaptic_neuron_id() - neuron_id_shift] = x_online[s->get_presynaptic_neuron_id() - neuron_id_shift] + 1;
                }
    
            // decision spikes are used by the decision layer to predict the class
            } else if (type == spike_type::decision) {
                
                // concerning the computation neuron
                if (computation_layer) {
                    
                    // during the training set the computation neuron collects data
                    if (network->get_learning_status()) {
                        x_training.emplace_back(x_online);
                        
                        // reset x_online each time it is included in the training vector
                        x_online = torch::zeros(number_of_output_neurons);
                    
                    // during the test set the computation neuron is used to decide which regression neuron from the decision layer should fire
                    } else {
                        
                        // 1. predict winner
                        
                        // 2. send a spike to the winner decision neuron (next regression layer)
                        
                    }
                   
                // concerning decision neurons
                } else {
                    // make the neuron with the correct class fire
                    potential = threshold;

                    if (network->get_verbose() >= 1) {
                        std::cout << "t=" << timestamp << " class " << class_label << " --> DECISION" << std::endl;
                    }

                    for (auto& addon: relevant_addons) {
                        addon->neuron_fired(timestamp, s, this, network);
                    }

                    if (network->get_main_thread_addon()) {
                        network->get_main_thread_addon()->neuron_fired(timestamp, s, this, network);
                    }
                    
                    potential = resting_potential;
                    
                    // reset the computation x_online each time a decision is made
                    dynamic_cast<Regression*>(network->get_neurons()[computation_id].get())->x_online = torch::zeros(number_of_output_neurons);
                }
            }
        }
        
        // reset a neuron to its initial status
        virtual void reset_neuron(Network* network, bool clearAddons=true) override {
            if (clearAddons) {
                relevant_addons.clear();
            }
            
            x_online = torch::zeros(number_of_output_neurons);
        }
        
    protected:
        
        std::vector<torch::Tensor> x_training;
        torch::Tensor              x_online;
        std::vector<int>           labels;
        float                      learning_rate;
        float                      momentum;
        float                      weight_decay;
        int                        epochs;
        int                        batch_size;
        bool                       computation_layer;
        int                        neuron_id_shift;
        int                        number_of_output_neurons;
        int                        presentations_before_training;
        int                        computation_id;
	};
}
