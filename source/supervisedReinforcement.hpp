/*
 * supervisedReinforcement.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 25/09/2018
 *
 * Information: The SupervisedReinforcement learning rule requires adding labels to the network before running it. This learning rule can only be used on the output layer. Each unique label will be assigned to a specific output neuron in order to positively or negatively reinforce the active axons depending on whether or not an output neuron predicted the correct label.
 */

#pragma once

namespace adonis_c
{
	class Neuron;
	
	class SupervisedReinforcement : public LearningRuleHandler
	{
	public:
		// ----- CONSTRUCTOR -----
		SupervisedReinforcement() = default;
		
		// ----- PUBLIC METHODS -----
		virtual void learn(double timestamp, Neuron* neuron, Network* network) override
		{
			if (neuron->getLayerID() == network->getLayers().back().ID)
			{
				if (network->getLabels())
				{
					auto it = std::find_if(network->getSupervisedNeurons().begin(), network->getSupervisedNeurons().end(), [neuron](supervisedOutput out){return out.neuron == neuron->getNeuronID();});
					
					auto idx = std::distance(network->getSupervisedNeurons().begin(), it);
					
					for (auto preAxon: neuron->getPreAxons())
					{
						// selecting plastic neurons
						if (preAxon->preNeuron->getEligibilityTrace() > 0.1)
						{
							// positive reinforcement for correct label
							if (network->getSupervisedNeurons()[idx].label == network->getCurrentLabel())
							{
								preAxon->weight += (preAxon->weight*20)/100;
								if (preAxon->weight > 1/preAxon->preNeuron->getInputResistance())
								{
									preAxon->weight = 1/preAxon->preNeuron->getInputResistance();
								}
							}
							// negative reinforcement for incorrect label
							else
							{
								if (preAxon->weight > 0)
								{
									preAxon->weight -= (preAxon->weight*20)/100;
									if (preAxon->weight < 0)
									{
										preAxon->weight = 0;
									}
								}
							}
						}
					}
				}
				else
				{
					throw std::logic_error("The supervised reinforcement learning rule cannot be used without first adding labels, before runnning the network");
				}
			}
			else
			{
				throw std::logic_error("The supervised reinforcement learning rule can only be used on the output layer");
			}
		}
		
	protected:
		// ----- IMPLEMENTATION VARIABLES -----
		std::vector<int16_t> outputNeuronLabelAssignment;
	};
}
