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

#include "../addon.hpp"
#include "../neurons/LIF.hpp"
#include "../addons/myelinPlasticityLogger.hpp"

namespace hummus {
	class Neuron;
	
	class MyelinPlasticity : public Addon {
        
	public:
		// ----- CONSTRUCTOR -----
        MyelinPlasticity(float _delay_alpha=1, float _delay_lambda=1, float _weight_alpha=1, float _weight_lambda=1) :
                delay_alpha(_delay_alpha),
                delay_lambda(_delay_lambda),
                weight_alpha(_weight_alpha),
                weight_lambda(_weight_lambda) {}
		
		// ----- PUBLIC METHODS -----
        // select one neuron to track by its index
        void activate_for(size_t neuronIdx) override {
            neuron_mask.push_back(static_cast<size_t>(neuronIdx));
        }
        
        // select multiple neurons to track by passing a vector of indices
        void activate_for(std::vector<size_t> neuronIdx) override {
            neuron_mask.insert(neuron_mask.end(), neuronIdx.begin(), neuronIdx.end());
        }
        
        virtual void learn(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            // forcing the neuron to be a LIF
            LIF* n = dynamic_cast<LIF*>(postsynapticNeuron);
            
            std::vector<double> timeDifferences;
            std::vector<int> plasticID;
            std::vector<std::vector<int>> plasticCoordinates(4);
            if (network->getVerbose() >= 1) {
                std::cout << "New learning epoch at t=" << timestamp << std::endl;
            }

            for (auto& inputSynapse: n->getDendriticTree()) {
                // discarding inhibitory synapses
                if (inputSynapse->getWeight() >= 0) {
                    auto& presynapticNeuron = network->getNeurons()[inputSynapse->getPresynapticNeuronID()];

                    if (presynapticNeuron->getEligibilityTrace() > 0.1) {
                        // saving relevant information in vectors for potential logging
                        plasticID.push_back(presynapticNeuron->getNeuronID());
                        plasticCoordinates[0].push_back(presynapticNeuron->getXYCoordinates().first);
                        plasticCoordinates[1].push_back(presynapticNeuron->getXYCoordinates().second);
                        plasticCoordinates[2].push_back(presynapticNeuron->getRfCoordinates().first);
                        plasticCoordinates[3].push_back(presynapticNeuron->getRfCoordinates().second);
                        timeDifferences.push_back(timestamp - inputSynapse->getPreviousInputTime() - inputSynapse->getDelay());

                        float delta_delay = 0;

                        if (timeDifferences.back() > 0) {
                            delta_delay = delay_lambda*(1/(s->getSynapseTimeConstant()-n->getDecayPotential())) * n->getCurrent() * (std::exp(-delay_alpha*timeDifferences.back()/s->getSynapseTimeConstant()) - std::exp(-delay_alpha*timeDifferences.back()/n->getDecayPotential()));
                            
                            inputSynapse->setDelay(delta_delay);
                            
                            if (network->getVerbose() >= 1) {
                                std::cout << timestamp << " " << presynapticNeuron->getNeuronID() << " " << n->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << delta_delay << std::endl;
                            }
                        } else if (timeDifferences.back() < 0) {
                            delta_delay = -delay_lambda*((1/(s->getSynapseTimeConstant()-n->getDecayPotential())) * n->getCurrent() * (std::exp(delay_alpha*timeDifferences.back()/s->getSynapseTimeConstant()) - std::exp(delay_alpha*timeDifferences.back()/n->getDecayPotential())));
                            
                            inputSynapse->setDelay(delta_delay);
                            
                            if (network->getVerbose() >= 1) {
                                std::cout << timestamp << " " << presynapticNeuron->getNeuronID() << " " << n->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << delta_delay << std::endl;
                            }
                        }
                    }
                }
            }

            // shifting weights to be equal to the number of plastic neurons
            float desiredWeight = 1./plasticID.size();

            for (auto i=0; i<postsynapticNeuron->getDendriticTree().size(); i++) {
                auto& dendrite = postsynapticNeuron->getDendriticTree()[i];
                // discarding inhibitory synapses
                if (dendrite->getWeight() >= 0) {
                    int ID = network->getNeurons()[dendrite->getPresynapticNeuronID()]->getNeuronID();
                    if (std::find(plasticID.begin(), plasticID.end(), ID) != plasticID.end()) {
                        float weightDifference = desiredWeight - dendrite->getWeight();
                        float change = - std::exp(- (weight_alpha*weightDifference) * (weight_alpha*weightDifference)) + 1;
                        if (weightDifference >= 0) {
                            dendrite->setWeight(weight_lambda*change * (1 - dendrite->getWeight()));
                        } else {
                            dendrite->setWeight(-weight_lambda*change * (1 - dendrite->getWeight()));
                        }
                    } else {
                        dendrite->setWeight(-weight_lambda * (1 - dendrite->getWeight()));
                    }
                }
            }

            for (auto& addon: network->getAddons()) {
                if (MyelinPlasticityLogger* myelinLogger = dynamic_cast<MyelinPlasticityLogger*>(addon.get())) {
                    dynamic_cast<MyelinPlasticityLogger*>(addon.get())->myelinPlasticityEvent(timestamp, postsynapticNeuron, network, timeDifferences, plasticCoordinates);
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
