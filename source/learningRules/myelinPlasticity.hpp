/*
 * myelinPlasticity.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: The MyelinPlasticity learning rule compatible only with leaky integrate-and-fire neurons
 */

#pragma once

#include <stdexcept>

#include "../neurons/leakyIntegrateAndFire.hpp"
#include "../addOns/myelinPlasticityLogger.hpp"
#include "../globalLearningRuleHandler.hpp"

namespace adonis
{
	class Neuron;
	
	class MyelinPlasticity : public GlobalLearningRuleHandler
	{
	public:
		// ----- CONSTRUCTOR -----
        MyelinPlasticity(float _delay_alpha=1, float _delay_lambda=1, float _weight_alpha=1, float _weight_lambda=1) :
        delay_alpha(_delay_alpha),
        delay_lambda(_delay_lambda),
        weight_alpha(_weight_alpha),
        weight_lambda(_weight_lambda)
		{}
		
		// ----- PUBLIC METHODS -----
        virtual void onStart(Network* network) override
        {
            for (auto& n: network->getNeurons())
            {
                for (auto& rule: n->getLearningRuleHandler())
                {
                    if (rule == this)
                    {
                        if (LIF* typeCheck = dynamic_cast<LIF*>(n.get()))
                        {
                            continue;
                        }
                        else
                        {
                            throw std::logic_error("The myelin plasticity learning rule is only compatible with Leaky Integrate-and-Fire (LIF) neurons");
                        }
                    }
                }
            }
        }
        
		virtual void learn(double timestamp, Neuron* neuron, Network* network) override
		{
            // forcing the neuron to be a LIF
            LIF* n = dynamic_cast<LIF*>(neuron);

			std::vector<double> timeDifferences;
			std::vector<int16_t> plasticID;
			std::vector<std::vector<int16_t>> plasticCoordinates(4);
			#ifndef NDEBUG
			std::cout << "New learning epoch at t=" << timestamp << std::endl;
			#endif
			
			for (auto inputAxon: n->getPreAxons())
			{
				// selecting plastic neurons
				if (inputAxon->preNeuron->getEligibilityTrace() > 0.1)
				{
					plasticID.push_back(inputAxon->preNeuron->getNeuronID());
					plasticCoordinates[0].push_back(inputAxon->preNeuron->getX());
					plasticCoordinates[1].push_back(inputAxon->preNeuron->getY());
					plasticCoordinates[2].push_back(inputAxon->preNeuron->getRfRow());
					plasticCoordinates[3].push_back(inputAxon->preNeuron->getRfCol());
					
					float change = 0;
					
					timeDifferences.push_back(timestamp - inputAxon->previousInputTime - inputAxon->delay);
					if (timeDifferences.back() > 0)
					{
						change = delay_lambda*(n->getMembraneResistance()/(n->getDecayCurrent()-n->getDecayPotential())) * n->getCurrent() * (std::exp(-delay_alpha*timeDifferences.back()/n->getDecayCurrent()) - std::exp(-delay_alpha*timeDifferences.back()/n->getDecayPotential()))*n->getSynapticEfficacy();
						inputAxon->delay += change;
						#ifndef NDEBUG
						std::cout << inputAxon->preNeuron->getLayerID() << " " << inputAxon->preNeuron->getNeuronID() << " " << inputAxon->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << change << std::endl;
						#endif
					}

					else if (timeDifferences.back() < 0)
					{
						change = -delay_lambda*((n->getMembraneResistance()/(n->getDecayCurrent()-n->getDecayPotential())) * n->getCurrent() * (std::exp(delay_alpha*timeDifferences.back()/n->getDecayCurrent()) - std::exp(delay_alpha*timeDifferences.back()/n->getDecayPotential())))*n->getSynapticEfficacy();
						inputAxon->delay += change;
						#ifndef NDEBUG
						std::cout << inputAxon->preNeuron->getLayerID() << " " << inputAxon->preNeuron->getNeuronID() << " " << inputAxon->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << change << std::endl;
						#endif
					}
					n->setSynapticEfficacy(-std::exp(-std::pow(timeDifferences.back(),2))+1);
                    inputAxon->preNeuron->setEligibilityTrace(0);
                    
                    // myelin plasticity rule sends a feedback to upstream neurons
                    for (auto& neurons: network->getNeurons())
                    {
                        //reducing their ability to learn as the current neurons learn
                        if (neurons->getLayerID() < n->getLayerID())
                        {
                            neurons->setSynapticEfficacy(-std::exp(-std::pow(timeDifferences.back(),2))+1);
                        }
                    }
				}
			}
			
            // looping through all axons from the winner
            for (auto& allAxons: neuron->getPreAxons())
            {
                // discarding inhibitory axons
                if (allAxons->weight > 0)
                {
                    int16_t ID = allAxons->preNeuron->getNeuronID();
                    if (std::find(plasticID.begin(), plasticID.end(), ID) != plasticID.end())
                    {
                        allAxons->weight += weight_lambda*std::exp(-std::pow(weight_alpha*timeDifferences.back(),2))*(1/allAxons->postNeuron->getMembraneResistance() - allAxons->weight) * neuron->getSynapticEfficacy();
                    }
                    else
                    {
                        // negative reinforcement on other axons going towards the winner to prevent other neurons from triggering it
                        allAxons->weight -= weight_lambda*std::exp(-std::pow(weight_alpha*timeDifferences.back(),2))*(1/allAxons->postNeuron->getMembraneResistance() - allAxons->weight) * neuron->getSynapticEfficacy();
                        if (allAxons->weight < 0)
                        {
                            allAxons->weight = 0;
                        }
                    }
                }
            }
            
            for (auto addon: network->getStandardAddOns())
            {
                if(MyelinPlasticityLogger* myelinLogger = dynamic_cast<MyelinPlasticityLogger*>(addon))
                {
                    dynamic_cast<MyelinPlasticityLogger*>(addon)->myelinPlasticityEvent(timestamp, network, neuron, timeDifferences, plasticCoordinates);
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
