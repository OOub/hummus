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

#include "../neurons/LIF.hpp"
#include "../addOns/myelinPlasticityLogger.hpp"
#include "../globalLearningRuleHandler.hpp"

namespace hummus {
	class Neuron;
	
	class MyelinPlasticity : public GlobalLearningRuleHandler {
        
	public:
		// ----- CONSTRUCTOR -----
        MyelinPlasticity(float _delay_alpha=1, float _delay_lambda=1, float _weight_alpha=1, float _weight_lambda=1) :
                delay_alpha(_delay_alpha),
                delay_lambda(_delay_lambda),
                weight_alpha(_weight_alpha),
                weight_lambda(_weight_lambda) {}
		
		// ----- PUBLIC METHODS -----
        virtual void onStart(Network* network) override {
            for (auto& n: network->getNeurons()) {
                for (auto& rule: n->getLearningRules()) {
                    if (rule == this) {
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
            // forcing the neuron to be a LIF
            LIF* n = dynamic_cast<LIF*>(a->postNeuron);

            std::vector<double> timeDifferences;
            std::vector<int> plasticID;
            std::vector<std::vector<int>> plasticCoordinates(4);
#ifndef NDEBUG
            std::cout << "New learning epoch at t=" << timestamp << std::endl;
#endif

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
                            delta_delay = delay_lambda*(n->getMembraneResistance()/(n->getSynapticKernel()->getSynapseTimeConstant()-n->getDecayPotential())) * n->getCurrent() * (std::exp(-delay_alpha*timeDifferences.back()/n->getSynapticKernel()->getSynapseTimeConstant()) - std::exp(-delay_alpha*timeDifferences.back()/n->getDecayPotential()))*n->getSynapticEfficacy();

                            inputSynapse->delay += delta_delay;
#ifndef NDEBUG
                            std::cout << timestamp << " " << inputSynapse->preNeuron->getNeuronID() << " " << inputSynapse->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << delta_delay << std::endl;
#endif
                        } else if (timeDifferences.back() < 0) {
                            delta_delay = -delay_lambda*((n->getMembraneResistance()/(n->getSynapticKernel()->getSynapseTimeConstant()-n->getDecayPotential())) * n->getCurrent() * (std::exp(delay_alpha*timeDifferences.back()/n->getSynapticKernel()->getSynapseTimeConstant()) - std::exp(delay_alpha*timeDifferences.back()/n->getDecayPotential())))*n->getSynapticEfficacy();

                            inputSynapse->delay += delta_delay;
#ifndef NDEBUG
                            std::cout << timestamp << " " << inputSynapse->preNeuron->getNeuronID() << " " << inputSynapse->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << delta_delay << std::endl;
#endif
                        }
                        n->setSynapticEfficacy(-std::exp(-std::pow(timeDifferences.back(),2))+1);

                        // myelin plasticity rule sends a feedback to upstream neurons
                        for (auto& n: network->getNeurons())
                        {
                            //reducing their ability to learn as the current neurons learn
                            if (n->getLayerID() < a->postNeuron->getLayerID())
                            {
                                n->setSynapticEfficacy(-std::exp(-std::pow(timeDifferences.back(),2))+1);
                            }
                        }

                        // resetting eligibility trace in plastic input neurons
                        inputSynapse->preNeuron->setEligibilityTrace(0);
                    }
                }
            }

            // shifting weights to be equal to the number of plastic neurons
            float desiredWeight = 1./plasticID.size()*(1/a->postNeuron->getMembraneResistance());

            for (auto i=0; i<a->postNeuron->getPreSynapses().size(); i++) {
                // discarding inhibitory synapses
                if (a->postNeuron->getPreSynapses()[i]->weight >= 0) {
                    int ID = a->postNeuron->getPreSynapses()[i]->preNeuron->getNeuronID();
                    if (std::find(plasticID.begin(), plasticID.end(), ID) != plasticID.end()) {
                        float weightDifference = (desiredWeight* a->postNeuron->getMembraneResistance()) - (a->postNeuron->getPreSynapses()[i]->weight*a->postNeuron->getMembraneResistance());
                        float change = - std::exp( - std::pow(weight_alpha*weightDifference,2)) + 1;
                        if (weightDifference >= 0) {
                            a->postNeuron->getPreSynapses()[i]->weight += weight_lambda*change*(1/a->postNeuron->getMembraneResistance());
                        } else {
                            a->postNeuron->getPreSynapses()[i]->weight -= weight_lambda*change*(1/a->postNeuron->getMembraneResistance());
                        }
                    } else {
                        if (a->postNeuron->getPreSynapses()[i]->weight > 0) {
                            a->postNeuron->getPreSynapses()[i]->weight -= 0.01* 1/a->postNeuron->getMembraneResistance();
                            if (a->postNeuron->getPreSynapses()[i]->weight < 0) {
                                a->postNeuron->getPreSynapses()[i]->weight = 0;
                            }
                        }
                    }
                }
            }

            for (auto addon: network->getAddOns()) {
                if(MyelinPlasticityLogger* myelinLogger = dynamic_cast<MyelinPlasticityLogger*>(addon)) {
                    dynamic_cast<MyelinPlasticityLogger*>(addon)->myelinPlasticityEvent(timestamp, network, a->postNeuron, timeDifferences, plasticCoordinates);
                }
            }
        }
		
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
        float delay_alpha;
        float delay_lambda;
        float weight_alpha;
        float weight_lambda;
	};
}
