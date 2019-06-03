/*
 * myelinPlasticity.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: The MyelinPlasticity learning rule compatible only with leaky integrate-and-fire neurons.
 *
 * LEARNING RULE TYPE 0 (in JSON SAVE FILE)
 */

#pragma once

#include <stdexcept>

#include "../learningRule.hpp"
#include "../neurons/LIF.hpp"
#include "../addons/myelinPlasticityLogger.hpp"

namespace hummus {
	class Neuron;
	
	class MyelinPlasticity : public LearningRule {
        
	public:
		// ----- CONSTRUCTOR -----
        MyelinPlasticity(float _delay_alpha=1, float _delay_lambda=1, float _weight_alpha=1, float _weight_lambda=1) :
                delay_alpha(_delay_alpha),
                delay_lambda(_delay_lambda),
                weight_alpha(_weight_alpha),
                weight_lambda(_weight_lambda) {}
		
		// ----- PUBLIC METHODS -----
        virtual void initialise(Network* network) override {
            for (auto& n: network->getNeurons()) {
                for (auto& addon: n->getRelevantAddons()) {
                    if (addon == this) {
                        if (LIF* typeCheck = dynamic_cast<LIF*>(n.get())) {
                            n->addLearningInfo(std::pair<int, std::vector<float>>(0, {delay_alpha, delay_lambda, weight_alpha, weight_lambda}));
                        } else {
                            throw std::logic_error("The myelin plasticity learning rule is only compatible with Leaky Integrate-and-Fire (LIF) neurons");
                        }
                    }
                }
            }
        }
        
        virtual void learn(double timestamp, synapse* a, Network* network) override {
            std::vector<double> timeDifferences;
            std::vector<int> plasticID;
            std::vector<std::vector<int>> plasticCoordinates(4);
            if (network->getVerbose() >= 1) {
                std::cout << "New learning epoch at t=" << timestamp << std::endl;
            }

            for (auto& inputSynapse: n->getPreSynapses()) {
                // discarding inhibitory synapses
                if (inputSynapse->weight >= 0) {

                    if (inputSynapse->preNeuron->getEligibilityTrace() > 0.1) {
                        // saving relevant information in vectors for potential logging
                        plasticID.push_back(inputSynapse->preNeuron->getNeuronID());
                        plasticCoordinates[0].push_back(inputSynapse->preNeuron->getXYCoordinates().first);
                        plasticCoordinates[1].push_back(inputSynapse->preNeuron->getXYCoordinates().second);
                        plasticCoordinates[2].push_back(inputSynapse->preNeuron->getRfCoordinates().first);
                        plasticCoordinates[3].push_back(inputSynapse->preNeuron->getRfCoordinates().second);
                        timeDifferences.push_back(timestamp - inputSynapse->previousInputTime - inputSynapse->delay);

                        float delta_delay = 0;

                        if (timeDifferences.back() > 0) {
                            delta_delay = delay_lambda*(1/(n->getSynapticKernel()->getSynapseTimeConstant()-n->getDecayPotential())) * n->getCurrent() * (std::exp(-delay_alpha*timeDifferences.back()/n->getSynapticKernel()->getSynapseTimeConstant()) - std::exp(-delay_alpha*timeDifferences.back()/n->getDecayPotential()))*n->getSynapticEfficacy();
                            
                            inputSynapse->delay += delta_delay;
                            if (network->getVerbose() >= 1) {
                                std::cout << timestamp << " " << inputSynapse->preNeuron->getNeuronID() << " " << inputSynapse->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << delta_delay << std::endl;
                            }
                        } else if (timeDifferences.back() < 0) {
                            delta_delay = -delay_lambda*((1/(n->getSynapticKernel()->getSynapseTimeConstant()-n->getDecayPotential())) * n->getCurrent() * (std::exp(delay_alpha*timeDifferences.back()/n->getSynapticKernel()->getSynapseTimeConstant()) - std::exp(delay_alpha*timeDifferences.back()/n->getDecayPotential())))*n->getSynapticEfficacy();
                            
                            inputSynapse->delay += delta_delay;
                            if (network->getVerbose() >= 1) {
                                std::cout << timestamp << " " << inputSynapse->preNeuron->getNeuronID() << " " << inputSynapse->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << delta_delay << std::endl;
                            }
                        }
                        n->setSynapticEfficacy(-std::exp(- timeDifferences.back() * timeDifferences.back())+1);
                    }
                }
            }

            // shifting weights to be equal to the number of plastic neurons
            float desiredWeight = 1./plasticID.size();

            for (auto i=0; i<a->postNeuron->getPreSynapses().size(); i++) {
                // discarding inhibitory synapses
                if (a->postNeuron->getPreSynapses()[i]->weight >= 0) {
                    int ID = a->postNeuron->getPreSynapses()[i]->preNeuron->getNeuronID();
                    if (std::find(plasticID.begin(), plasticID.end(), ID) != plasticID.end()) {
                        float weightDifference = desiredWeight - a->postNeuron->getPreSynapses()[i]->weight;
                        float change = - std::exp(- (weight_alpha*weightDifference) * (weight_alpha*weightDifference)) + 1;
                        if (weightDifference >= 0) {
                            a->postNeuron->getPreSynapses()[i]->weight += weight_lambda*change * (1 - a->weight);
                        } else {
                            a->postNeuron->getPreSynapses()[i]->weight -= weight_lambda*change * (1 - a->weight);
                        }
                    } else {
                        a->postNeuron->getPreSynapses()[i]->weight -= weight_lambda * (1 - a->weight);
                    }
                }
            }

            for (auto& addon: network->getAddons()) {
                if (MyelinPlasticityLogger* myelinLogger = dynamic_cast<MyelinPlasticityLogger*>(addon.get())) {
                    dynamic_cast<MyelinPlasticityLogger*>(addon.get())->myelinPlasticityEvent(timestamp, network, a->postNeuron, timeDifferences, plasticCoordinates);
                }
            }
        }
		
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
        float                delay_alpha;
        float                delay_lambda;
        float                weight_alpha;
        float                weight_lambda;
	};
}
