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
			labels = parser.readLabels(testLabels);
			for (auto label: labels) {
				actualLabels.emplace_back(label.name);
			}
		}
		
        virtual ~Analysis(){}
        
		// ----- PUBLIC METHODS -----
		void accuracy() {
			if (!classifiedLabels.empty() && classifiedLabels.size() == actualLabels.size()) {
				std::vector<std::string> correctLabels;
				for (auto i=0; i<actualLabels.size(); i++) {
					if (classifiedLabels[i] == actualLabels[i]) {
						correctLabels.emplace_back(classifiedLabels[i]);
					}
				}
				
				double accuracy = (static_cast<double>(correctLabels.size())/actualLabels.size())*100;
				std::cout << "the classification accuracy is: " << accuracy << "%" << std::endl;
			} else {
				throw std::logic_error("there is a problem with the classified and actual labels");
			}
		}
		
		void neuronFired(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            if (network->getDecisionMaking()) {
                // logging only after learning is stopped and restrict only to the decision-making layer
                if (!network->getLearningStatus() && postsynapticNeuron->getLayerID() == network->getDecisionParameters().layer_number) {
                    classifiedSpikes.emplace_back(std::make_pair(timestamp, postsynapticNeuron));
                }
            } else {
                throw std::logic_error("the analysis class works only when decision-making neurons are added to the network");
            }
		}
		
		void onCompleted(Network* network) override {
            labels.emplace_back(label{"end", labels.back().onset+10000});

            for (auto i=1; i<labels.size(); i++) {
                auto it = std::find_if(classifiedSpikes.begin(), classifiedSpikes.end(), [&](std::pair<double, Neuron*> const& a){return a.first >= labels[i-1].onset && a.first < labels[i].onset;});
                if (it != classifiedSpikes.end()) {
                    auto idx = std::distance(classifiedSpikes.begin(), it);
                    classifiedLabels.emplace_back(classifiedSpikes[idx].second->getClassLabel());
                } else {
                    classifiedLabels.emplace_back("NaN");
                }
            }
		}
		
	protected:
		// ----- IMPLEMENTATION VARIABLES -----
		std::vector<std::pair<double, Neuron*>>  classifiedSpikes;
		std::deque<label>                        labels;
		std::deque<std::string>                  actualLabels;
		std::deque<std::string>                  classifiedLabels;
	};
}
