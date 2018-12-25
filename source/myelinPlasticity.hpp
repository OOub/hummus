/*
 * myelinPlasticity.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 11/12/2018
 *
 * Information: The MyelinPlasticity learning rule
 */

#pragma once

#include "myelinPlasticityLogger.hpp"

namespace adonis_c
{
	class Neuron;
	
	class MyelinPlasticity : public LearningRuleHandler
	{
	public:
		// ----- CONSTRUCTOR -----
		MyelinPlasticity(float _alpha=1, float _lambda=1, bool _weightReinforcement=false) :
		alpha(_alpha),
		lambda(_lambda),
		weightReinforcement(_weightReinforcement)
		{}
		
		// ----- PUBLIC METHODS -----
		virtual void learn(double timestamp, Neuron* neuron, Network* network) override
		{
			std::vector<double> timeDifferences;
			std::vector<int16_t> plasticID;
			std::vector<std::vector<int16_t>> plasticCoordinates(4);
			#ifndef NDEBUG
			std::cout << "New learning epoch at t=" << timestamp << std::endl;
			#endif
			
			for (auto inputAxon: neuron->getPreAxons())
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
					
					timeDifferences.push_back(timestamp - inputAxon->lastInputTime - inputAxon->delay);
					if (timeDifferences.back() > 0)
					{
						change = lambda*(neuron->getInputResistance()/(neuron->getDecayCurrent()-neuron->getDecayPotential())) * neuron->getCurrent() * (std::exp(-alpha*timeDifferences.back()/neuron->getDecayCurrent()) - std::exp(-alpha*timeDifferences.back()/neuron->getDecayPotential()))*neuron->getSynapticEfficacy();
						inputAxon->delay += change;
						#ifndef NDEBUG
						std::cout << inputAxon->preNeuron->getLayerID() << " " << inputAxon->preNeuron->getNeuronID() << " " << inputAxon->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << change << std::endl;
						#endif
					}

					else if (timeDifferences.back() < 0)
					{
						change = -lambda*((neuron->getInputResistance()/(neuron->getDecayCurrent()-neuron->getDecayPotential())) * neuron->getCurrent() * (std::exp(alpha*timeDifferences.back()/neuron->getDecayCurrent()) - std::exp(alpha*timeDifferences.back()/neuron->getDecayPotential())))*neuron->getSynapticEfficacy();
						inputAxon->delay += change;
						#ifndef NDEBUG
						std::cout << inputAxon->preNeuron->getLayerID() << " " << inputAxon->preNeuron->getNeuronID() << " " << inputAxon->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << change << std::endl;
						#endif
					}
					neuron->setSynapticEfficacy(-std::exp(-std::pow(timeDifferences.back(),2))+1);
					
//					// myelin plasticity rule sends a feedback to upstream neurons
//					for (auto& n: network->getNeurons())
//					{
//						//reducing their ability to learn as the current neurons learn
//						if (n.getLayerID() < neuron->getLayerID())
//						{
//							n.setSynapticEfficacy(-std::exp(-std::pow(timeDifferences.back(),2))+1);
//						}
//					}
				}
			}
			
			for (auto addon: network->getStandardAddOns())
			{
				if(MyelinPlasticityLogger* myelinLogger = dynamic_cast<MyelinPlasticityLogger*>(addon))
				{
					dynamic_cast<MyelinPlasticityLogger*>(addon)->myelinPlasticityEvent(timestamp, network, neuron, timeDifferences, plasticCoordinates);
				}
			}
			
			// weight reinforcement
			if (weightReinforcement)
			{
				// looping through all axons from the winner
				for (auto& allAxons: neuron->getPreAxons())
				{
					// discarding inhibitory axons
					if (allAxons->weight > 0)
					{
						int16_t ID = allAxons->preNeuron->getNeuronID();
						if (std::find(plasticID.begin(), plasticID.end(), ID) != plasticID.end())
						{
							allAxons->weight += std::exp(-std::pow(timeDifferences.back(),2))*(1/allAxons->postNeuron->getInputResistance() - allAxons->weight) * neuron->getSynapticEfficacy();
						}
						else
						{
							// negative reinforcement on other axons going towards the winner to prevent other neurons from triggering it
							allAxons->weight -= std::exp(-std::pow(timeDifferences.back(),2))*(1/allAxons->postNeuron->getInputResistance() - allAxons->weight) * neuron->getSynapticEfficacy();
							if (allAxons->weight < 0)
							{
								allAxons->weight = 0;
							}
						}
					}
				}
            }
		}
		
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
		float alpha;
		float lambda;
		bool  weightReinforcement;
	};
}
