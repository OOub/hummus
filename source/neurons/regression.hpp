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
#include "../../third_party/numpy.hpp"

namespace hummus {

    class CustomDataset : public torch::data::Dataset<CustomDataset> {
    public:
        explicit CustomDataset(std::vector<torch::Tensor> _data, std::vector<int>& _labels, int number_of_output_neurons) :
            data_(torch::stack(_data, 0)),
            labels_(torch::tensor(_labels)),
            data_size(_labels.size()),
            out_dim(number_of_output_neurons) {};

        torch::data::Example<> get(size_t index) override {
            return {data_[index], labels_[index]};
        };
        
        torch::optional<size_t> size() const override {
            return data_size;
        };
        
        protected:
            torch::Tensor data_;
            torch::Tensor labels_;
            size_t        data_size;
            int           out_dim;
        
    };
    
    class Synapse;
    class Neuron;
    class Network;

	class Regression : public Neuron {
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        Regression(int _neuronID, int _layerID, int _sublayerID, int _rf_id,  std::pair<int, int> _xyCoordinates, std::string _classLabel="", float _learning_rate=0, float _momentum=0, float _weight_decay=0, int _epochs=10, int _batch_size=32, int _log_interval=10, int _presentations_before_training=0, bool save_tensor=false, float _threshold=-50, float _restingPotential=-70) :
                Neuron(_neuronID, _layerID, _sublayerID, _rf_id, _xyCoordinates, 0, 200, 10, 20, _threshold, _restingPotential, _classLabel),
                learning_rate(_learning_rate),
                momentum(_momentum),
                weight_decay(_weight_decay),
                epochs(_epochs),
                batch_size(_batch_size),
                computation_layer(false),
                neuron_id_shift(0),
                number_of_output_neurons(0),
                presentations_before_training(_presentations_before_training),
                computation_id(0),
                log_interval(_log_interval),
                model(100,3),
                debug_mode(save_tensor) {

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
        
        virtual void end(Network* network) override {
            // save training and test tensor into numpy array
            if (debug_mode && computation_layer) {
                // parsing training data
                torch::Tensor tmp_tr_data = torch::stack(x_training, 0);
                torch::Tensor tmp_tr_labels = torch::tensor(labels_train);
                
                // converting data to stl container
                const int tr_data_shape[2] = {static_cast<int>(tmp_tr_data.size(0)),static_cast<int>(tmp_tr_data.size(1))};
                std::vector<int> tr_data_stl;
                for (auto i=0; i<tmp_tr_data.size(0); ++i) {
                    for (auto j=0; j<tmp_tr_data.size(1); ++j) {
                        tr_data_stl.emplace_back(tmp_tr_data[i][j].item<int>());
                    }
                }
                
                // converting labels to stl container
                const int tr_label_shape[2] = {static_cast<int>(tmp_tr_labels.size(0))};
                std::vector<int> tr_label_stl;
                for (auto i=0; i<tmp_tr_labels.size(0); ++i) {
                    tr_label_stl.emplace_back(tmp_tr_labels[i].item<int>());
                }
                
                // saving training set to npy file
                aoba::SaveArrayAsNumpy("logistic_tr_set.npy", false, 2, &tr_data_shape[0], &tr_data_stl[0]);
                aoba::SaveArrayAsNumpy("logistic_tr_label.npy", false, 1, &tr_label_shape[0], &tr_label_stl[0]);
                
                // parsing test data
                torch::Tensor tmp_te_data = torch::stack(x_test, 0);
                torch::Tensor tmp_te_labels = torch::tensor(labels_test);;
                
                // converting data to stl container
                const int te_data_shape[2] = {static_cast<int>(tmp_te_data.size(0)),static_cast<int>(tmp_te_data.size(1))};
                std::vector<int> te_data_stl;
                for (auto i=0; i<tmp_te_data.size(0); ++i) {
                    for (auto j=0; j<tmp_te_data.size(1); ++j) {
                        te_data_stl.emplace_back(tmp_te_data[i][j].item<int>());
                    }
                }
                
                // converting labels to stl container
                const int te_label_shape[2] = {static_cast<int>(tmp_te_labels.size(0))};
                std::vector<int> te_label_stl;
                for (auto i=0; i<tmp_te_labels.size(0); ++i) {
                    te_label_stl.emplace_back(tmp_te_labels[i].item<int>());
                }
                
                // saving test set to npy file
                aoba::SaveArrayAsNumpy("logistic_te_set.npy", false, 2, &te_data_shape[0], &te_data_stl[0]);
                aoba::SaveArrayAsNumpy("logistic_te_label.npy", false, 1, &te_label_shape[0], &te_label_stl[0]);
            }
        }
        
        virtual void update(double timestamp, Synapse* s, Network* network, float timestep, spike_type type) override {
            
            std::cout << "regression accessed" << std::endl;
            
            // none spikes are used  by the computation layer for training the logistic regression
            if (type == spike_type::none && computation_layer) {
                
                train_model(network);
                
            // generated spikes are used to collect features
            } else if (type == spike_type::generated && computation_layer) {
                
                if (network->get_learning_status()) {
                    // during learning only change x_online after waiting for a number of training presentations defined by the variable: presentations_before_training
                    if (network->get_presentation_counter() >= presentations_before_training) {
                        x_online[s->get_presynaptic_neuron_id() - neuron_id_shift] = x_online[s->get_presynaptic_neuron_id() - neuron_id_shift] + 1;
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
                    if (network->get_learning_status() && network->get_presentation_counter() >= presentations_before_training) {
                        x_training.emplace_back(x_online);
                        labels_train.emplace_back(network->get_classes_map()[network->get_current_label()]);
                        
                        // reset x_online each time it is included in the training vector
                        x_online = torch::zeros(number_of_output_neurons);
                    
                    // during the test set the computation neuron is used to decide which regression neuron from the decision layer should fire
                    } else if (!network->get_learning_status()){
                        x_test.emplace_back(x_online);
                        labels_test.emplace_back(network->get_classes_map()[network->get_current_label()]);
                        
                        // predict winner
                        test_model(timestamp, timestep, network);
                        
                        // reset x_online
                        x_online = torch::zeros(number_of_output_neurons);
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
        
        virtual void train_model(Network* network) {
            
            if (x_training.empty()) {
                throw std::runtime_error("the training data vector is empty");
            }
            
            // generate data set. we can add transforms to the data set, e.g. stack batches into a single tensor.
            auto data_set = CustomDataset(x_training, labels_train, number_of_output_neurons).map(torch::data::transforms::Stack<>());
            
            // generate a data loader
            auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                std::move(data_set),
                batch_size);
            
            int dataset_size = data_set.size().value();
            
            // create model and instantiate it (size, dimension)
            model = torch::nn::Linear(number_of_output_neurons, network->get_classes_map().size());
            
            // instantiate optimizer
            torch::optim::SGD optimizer(model->parameters(),
                                        torch::optim::SGDOptions(learning_rate).momentum(momentum).weight_decay(weight_decay));
            
            // loop through epochs to train logistic regression on the training batches
            for (auto epoch=1; epoch <= epochs; ++epoch) {
                
                // Track loss.
                int batch_idx = 0;
                
                for (auto& batch : *data_loader) {
                    auto tr_data = batch.data;
                    auto tr_labels = batch.target.squeeze();
                    
                    // format data and tr_labels to the accepted torch data types
                    tr_data = tr_data.to(torch::kF32);
                    tr_labels = tr_labels.to(torch::kInt64);
                    
                    // reset gradients
                    optimizer.zero_grad();
                    
                    // forward pass
                    auto tr_output = torch::nll_loss(torch::log_softmax(model(tr_data),1) ,tr_labels);
                    auto loss = tr_output.item<float>();
                    
                    // backward pass
                    tr_output.backward();
                    
                    // apply gradients
                    optimizer.step();

                    ++batch_idx;
                    if (network->get_verbose() >= 1 && batch_idx % log_interval == 0) {
                        std::printf(
                        "\rTrain Epoch: %d/%d [%5d/%5d] Loss: %.4f",
                        epoch,
                        epochs,
                        batch_idx * static_cast<int>(batch.data.size(0)),
                        dataset_size,
                        loss);
                    }
                }
            }
        }
        
        virtual void test_model(double timestamp, float timestep, Network* network) {
            
            x_online = x_online.to(torch::kF32);
            
            torch::Tensor output = model(x_online);
            
            auto pred = output.argmax(0);
            auto class_label = network->get_reverse_classes_map()[pred.item<int>()];
                
            auto& decision = network->get_layers()[network->get_decision_parameters().layer_number+1].neurons;
            for (auto& n: decision) {
                auto& neuron = network->get_neurons()[n];
                if (neuron->get_class_label() == class_label) {
                    neuron->update(timestamp, nullptr, network, timestep, spike_type::decision);
                }
            }
        }
        
        std::vector<torch::Tensor> x_training;
        std::vector<torch::Tensor> x_test;
        torch::Tensor              x_online;
        std::vector<int>           labels_train;
        std::vector<int>           labels_test;
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
        int                        log_interval;
        torch::nn::Linear          model;
        bool                       debug_mode;
	};
}
