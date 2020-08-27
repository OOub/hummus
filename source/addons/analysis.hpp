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

#include "../data_parser.hpp"

namespace hummus {
    
    class Synapse;
    class Neuron;
    class Network;
    
	class Analysis : public Addon {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        Analysis(std::deque<label>& testLabels, std::string _filename="") :
        filename(_filename),
        labels(testLabels) {
            // save label ids
            std::transform(testLabels.begin(), testLabels.end(),std::back_inserter(actual_labels),[](label& l) {return l.id;});
        }
        
        virtual ~Analysis(){}
        
		// ----- PUBLIC METHODS -----
		float accuracy(int verbose=1) {
			if (!predicted_labels.empty() && predicted_labels.size() == actual_labels.size()) {
				std::vector<int> correctLabels;
				for (int i=0; i<static_cast<int>(actual_labels.size()); i++) {
					if (predicted_labels[i] == actual_labels[i]) {
						correctLabels.emplace_back(predicted_labels[i]);
					}
				}
				
                // save labels
                if (!filename.empty()) {
                    std::ofstream ofs(filename);
                    for (int i=0; i < static_cast<int>(actual_labels.size()); i++) {
                        ofs << actual_labels[i] << " " << predicted_labels[i] << "\n";
                    }
                }
                
				float accuracy = (static_cast<float>(correctLabels.size())/actual_labels.size())*100.;
                
                if (verbose > 0) {
                    std::cout << "the classification accuracy is: " << accuracy << "%" << std::endl;
                }
                return accuracy;
			} else {
                if (verbose > 0) {
                    std::cout << "there is a problem with the classified and actual labels" << std::endl;
                }
                return -1;
			}
		}
		
		void neuron_fired(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            if (network->get_decision_making()) {
                // logging only after learning is stopped and restrict only to the decision-making layer
                if (!network->get_learning_status() && postsynapticNeuron->get_layer_id() == network->get_decision_parameters().layer_number) {
                    classified_spikes.emplace_back(std::make_pair(timestamp, postsynapticNeuron));
                }
            } else if (network->get_logistic_regression()) {
                // logging only after learning is stopped and restrict only to the logistic regression decision layer
                if (!network->get_learning_status() && postsynapticNeuron->get_layer_id() == network->get_decision_parameters().layer_number+1) {
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
                        predicted_labels.emplace_back(s.second->get_class_label());
                    } else {
                        predicted_labels.emplace_back(-1);
                    }
                }
            // if choose_winner_online is used
            } else {
                // if the labels are not timestamped
                
                if (labels[0].timestamp == -1) {
                    // find all the nullptrs delimiting the eof of each pattern
                    long previous_idx = 0;
                    for (auto it = std::find_if(classified_spikes.begin(), classified_spikes.end(), [&](std::pair<double, Neuron*> const& a){return a.second == nullptr;});
                         it != classified_spikes.end();
                         it = std::find_if(++it, classified_spikes.end(), [&](std::pair<double, Neuron*> const& a){return a.second == nullptr;}))
                    {
                        std::vector<int> labels_interval;
                        if (it != classified_spikes.begin() && it != classified_spikes.end()) {
                            auto idx = std::distance(classified_spikes.begin(), std::prev(it));
                            for (auto i = previous_idx; i <= idx; i++) {
                                labels_interval.emplace_back(classified_spikes[idx].second->get_class_label());
                            }
                            previous_idx = std::distance(classified_spikes.begin(), std::next(it));
                        }


                        if (!labels_interval.empty()) {
                            // find the most occurring class
                            std::unordered_map<int, int> freq_class;
                            for (auto& label: labels_interval) {
                                freq_class[label]++;
                            }

                            // return the element with the maximum number of spikes
                            auto max_label = *std::max_element(freq_class.begin(), freq_class.end(), [](const std::pair<int, int> &p1,
                                                                                                        const std::pair<int, int> &p2) {
                                                                                            return p1.second < p2.second;
                                                                                        });
                            predicted_labels.emplace_back(max_label.first);
                        } else {
                            predicted_labels.emplace_back(-1);
                        }
                    }

                } else {
                    labels.emplace_back(label{-2, std::numeric_limits<double>::max()});

                    for (int i=1; i<static_cast<int>(labels.size()); i++) {

                        std::vector<int> labels_interval;
                        // get all labels between each interval
                        for (auto it = std::find_if(classified_spikes.begin(), classified_spikes.end(), [&](std::pair<double, Neuron*> const& a){return a.first >= labels[i-1].timestamp && a.first < labels[i].timestamp;});
                             it != classified_spikes.end();
                             it = std::find_if(++it, classified_spikes.end(), [&](std::pair<double, Neuron*> const& a){return a.first >= labels[i-1].timestamp && a.first < labels[i].timestamp;}))
                        {
                            if (it != classified_spikes.end()) {
                                auto idx = std::distance(classified_spikes.begin(), it);
                                labels_interval.emplace_back(classified_spikes[idx].second->get_class_label());
                            }
                        }

                        if (!labels_interval.empty()) {
                            // find the most occurring class
                            std::unordered_map<int, int> freq_class;
                            for (auto& label: labels_interval) {
                                freq_class[label]++;
                            }

                            // return the element with the maximum number of spikes
                            auto max_label = *std::max_element(freq_class.begin(), freq_class.end(), [](const std::pair<int, int> &p1,
                                                                                                        const std::pair<int, int> &p2) {
                                                                                            return p1.second < p2.second;
                                                                                        });
                            predicted_labels.emplace_back(max_label.first);
                        } else {
                            predicted_labels.emplace_back(-1);
                        }
                    }
                }
            }
		}
		
	protected:
		// ----- IMPLEMENTATION VARIABLES -----
        std::string                              filename;
		std::vector<std::pair<double, Neuron*>>  classified_spikes;
        std::deque<label>&                       labels;
		std::vector<int>                         actual_labels;
		std::vector<int>                         predicted_labels;
	};
}
