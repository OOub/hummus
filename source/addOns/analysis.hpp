/*
 * analysis.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 29/11/2018
 *
 * Information: The Analysis class checks the classification accuracy of the spiking neural network
 */

#pragma once

#include <vector>

#include "../core.hpp"
#include "../neurons/decisionMakingNeuron.hpp"
#include "../dataParser.hpp"

namespace adonis {
	class Analysis : public AddOn {
        
	public:
		// ----- CONSTRUCTOR -----
		Analysis(std::string testLabels) {
			DataParser parser;
			labels = parser.readLabels(testLabels);
			for (auto label: labels) {
				actualLabels.emplace_back(label.name);
			}
		}
		
		// ----- PUBLIC METHODS -----
		void accuracy() {
			if (!predictedLabels.empty() && predictedLabels.size() == actualLabels.size()) {
				std::vector<std::string> correctLabels;
				for (auto i=0; i<actualLabels.size(); i++) {
					if (predictedLabels[i] == actualLabels[i]) {
						correctLabels.emplace_back(predictedLabels[i]);
					}
				}
				
				double accuracy = (static_cast<double>(correctLabels.size())/actualLabels.size())*100;
				std::cout << "the classification accuracy is: " << accuracy << "%" << std::endl;
			} else {
				throw std::logic_error("there is a problem with the predicted and actual labels");
			}
		}
		
		void neuronFired(double timestamp, axon* a, Network* network) override {
			// logging only after learning is stopped
			if (!network->getLearningStatus()) {
				// restrict only to the decision-making layer
                if (DecisionMakingNeuron* neuron = dynamic_cast<DecisionMakingNeuron*>(a->postNeuron)) {
					if (dynamic_cast<DecisionMakingNeuron*>(a->postNeuron)->getClassLabel() != "") {
						predictedSpikes.emplace_back(spike{timestamp, a});
					}
					else {
						throw std::logic_error("the output neurons are unlabelled. Please use the addDecisionMakingLayer method to create the output layer");
					}
				}
			}
		}
		
		void onCompleted(Network* network) override {
			labels.emplace_back(label{"end", labels.back().onset+10000});
			
			for (auto i=1; i<labels.size(); i++) {
				auto it = std::find_if(predictedSpikes.begin(), predictedSpikes.end(), [&](spike a){return a.timestamp >= labels[i-1].onset && a.timestamp < labels[i].onset;});
				if (it != predictedSpikes.end()) {
					auto idx = std::distance(predictedSpikes.begin(), it);
					predictedLabels.emplace_back(dynamic_cast<DecisionMakingNeuron*>(predictedSpikes[idx].propagationAxon->postNeuron)->getClassLabel());
				} else {
					predictedLabels.emplace_back("NaN");
				}
			}
		}
		
	protected:
		// ----- IMPLEMENTATION VARIABLES -----
		std::vector<spike>       predictedSpikes;
		std::deque<label>        labels;
		std::deque<std::string>  actualLabels;
		std::deque<std::string>  predictedLabels;
	};
}
