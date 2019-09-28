/*
 * analysis.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 29/11/2018
 *
 * Information: The Analysis class checks the classification accuracy of the spiking neural network
 */

#pragma once

#include <vector>

#include "../dataParser.hpp"

namespace hummus {
    
    class Synapse;
    class Neuron;
    class Network;
    
	class Analysis : public Addon {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		Analysis(std::string testLabels) {
			DataParser parser;
			labels = parser.read_txt_labels(testLabels);
			for (auto label: labels) {
				actual_labels.emplace_back(label.name);
			}
		}
		
        Analysis(std::deque<label> testLabels) {
            labels = testLabels;
            for (auto label: labels) {
                actual_labels.emplace_back(label.name);
            }
        }
        
        virtual ~Analysis(){}
        
		// ----- PUBLIC METHODS -----
		void accuracy() {
			if (!classified_labels.empty() && classified_labels.size() == actual_labels.size()) {
				std::vector<std::string> correctLabels;
				for (auto i=0; i<actual_labels.size(); i++) {
					if (classified_labels[i] == actual_labels[i]) {
						correctLabels.emplace_back(classified_labels[i]);
					}
				}
				
				double accuracy = (static_cast<double>(correctLabels.size())/actual_labels.size())*100.;
				std::cout << "the classification accuracy is: " << accuracy << "%" << std::endl;
			} else {
				throw std::logic_error("there is a problem with the classified and actual labels");
			}
		}
		
		void neuron_fired(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            if (network->get_decision_making()) {
                // logging only after learning is stopped and restrict only to the decision-making layer
                if (!network->get_learning_status() && postsynapticNeuron->get_layer_id() == network->get_decision_parameters().layer_number) {
                    classified_spikes.emplace_back(std::make_pair(timestamp, postsynapticNeuron));
                }
            } else {
                throw std::logic_error("the analysis class works only when decision-making neurons are added to the network");
            }
		}
		
		void on_completed(Network* network) override {
            labels.emplace_back(label{"end", labels.back().onset+10000});

            for (auto i=1; i<labels.size(); i++) {
                auto it = std::find_if(classified_spikes.begin(), classified_spikes.end(), [&](std::pair<double, Neuron*> const& a){return a.first >= labels[i-1].onset && a.first < labels[i].onset;});
                if (it != classified_spikes.end()) {
                    auto idx = std::distance(classified_spikes.begin(), it);
                    classified_labels.emplace_back(classified_spikes[idx].second->get_class_label());
                } else {
                    classified_labels.emplace_back("NaN");
                }
            }
		}
		
	protected:
		// ----- IMPLEMENTATION VARIABLES -----
		std::vector<std::pair<double, Neuron*>>  classified_spikes;
		std::deque<label>                        labels;
		std::deque<std::string>                  actual_labels;
		std::deque<std::string>                  classified_labels;
	};
}
