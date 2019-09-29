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
#include <limits>
#include <algorithm>

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
        
        void decision_failed(double timestamp, Network* network) override {
            classified_spikes.emplace_back(std::make_pair(timestamp, nullptr));
        }
		
		void on_completed(Network* network) override {
            // if choose_winner_eof is used
            if (network->get_decision_parameters().timer == 0) {
                for (auto& s: classified_spikes) {
                    if (s.second) {
                        classified_labels.emplace_back(s.second->get_class_label());
                    } else {
                        classified_labels.emplace_back("NaN");
                    }
                }
            // if choose_winner_online is used
            } else {
                // if the labels are not timestamped
                
                if (labels[0].onset == -1) {
                    // find all the nullptrs delimiting the eof of each pattern
                    long previous_idx = 0;
                    for (auto it = std::find_if(classified_spikes.begin(), classified_spikes.end(), [&](std::pair<double, Neuron*> const& a){return a.second == nullptr;});
                         it != classified_spikes.end();
                         it = std::find_if(++it, classified_spikes.end(), [&](std::pair<double, Neuron*> const& a){return a.second == nullptr;}))
                    {
                        std::vector<std::string> labels_interval;
                        if (it != classified_spikes.begin() && it != classified_spikes.end()) {
                            auto idx = std::distance(classified_spikes.begin(), std::prev(it));
                            for (auto i = previous_idx; i <= idx; i++) {
                                labels_interval.emplace_back(classified_spikes[idx].second->get_class_label());
                            }
                            previous_idx = std::distance(classified_spikes.begin(), std::next(it));
                        }


                        if (!labels_interval.empty()) {
                            // find the most occurring class
                            std::unordered_map<std::string, int> freq_class;
                            for (auto& label: labels_interval) {
                                freq_class[label]++;
                            }

                            // return the element with the maximum number of spikes
                            auto max_label = *std::max_element(freq_class.begin(), freq_class.end(), [](const std::pair<std::string, int> &p1,
                                                                                                        const std::pair<std::string, int> &p2) {
                                                                                            return p1.second < p2.second;
                                                                                        });
                            classified_labels.emplace_back(max_label.first);
                        } else {
                            classified_labels.emplace_back("NaN");
                        }
                    }

                } else {
                    labels.emplace_back(label{"end", std::numeric_limits<double>::max()});

                    for (auto i=1; i<labels.size(); i++) {

                        std::vector<std::string> labels_interval;
                        // get all labels between each interval
                        for (auto it = std::find_if(classified_spikes.begin(), classified_spikes.end(), [&](std::pair<double, Neuron*> const& a){return a.first >= labels[i-1].onset && a.first < labels[i].onset;});
                             it != classified_spikes.end();
                             it = std::find_if(++it, classified_spikes.end(), [&](std::pair<double, Neuron*> const& a){return a.first >= labels[i-1].onset && a.first < labels[i].onset;}))
                        {
                            if (it != classified_spikes.end()) {
                                auto idx = std::distance(classified_spikes.begin(), it);
                                labels_interval.emplace_back(classified_spikes[idx].second->get_class_label());
                            }
                        }

                        if (!labels_interval.empty()) {
                            // find the most occurring class
                            std::unordered_map<std::string, int> freq_class;
                            for (auto& label: labels_interval) {
                                freq_class[label]++;
                            }

                            // return the element with the maximum number of spikes
                            auto max_label = *std::max_element(freq_class.begin(), freq_class.end(), [](const std::pair<std::string, int> &p1,
                                                                                                        const std::pair<std::string, int> &p2) {
                                                                                            return p1.second < p2.second;
                                                                                        });
                            classified_labels.emplace_back(max_label.first);
                        } else {
                            classified_labels.emplace_back("NaN");
                        }
                    }
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
