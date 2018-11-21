/*
 * supervisedReinforcement.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 25/09/2018
 *
 * Information: The SupervisedReinforcement learning rule requires adding labels to the network before running it. This learning rule can only be used on the output layer. Each unique label will be assigned to a specific output neuron in order to positively or negatively reinforce the active projections depending on whether or not an output neuron predicted the correct label.
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
					
					for (auto preProjection: neuron->getPreProjections())
					{
						// selecting plastic neurons
						if (preProjection->preNeuron->getEligibilityTrace() > 0.1)
						{
							// positive reinforcement for correct label
							if (network->getSupervisedNeurons()[idx].label == network->getCurrentLabel())
							{
								preProjection->weight += (preProjection->weight*10)/100;
								if (preProjection->weight > 1/preProjection->preNeuron->getInputResistance())
								{
									preProjection->weight = 1/preProjection->preNeuron->getInputResistance();
								}
							}
							// negative reinforcement for incorrect label
							else
							{
								if (preProjection->weight > 0)
								{
									preProjection->weight -= (preProjection->weight*10)/100;
									if (preProjection->weight < 0)
									{
										preProjection->weight = 0;
									}
								}
							}
						}
					}
				}
				else
				{
					throw std::logic_error("The supervised WTA learning rule cannot be used without first adding labels, before runnning the network");
				}
			}
			else
			{
				throw std::logic_error("The supervised WTA learning rule can only be used on the output layer");
			}
		}
		
	protected:
		// ----- IMPLEMENTATION VARIABLES -----
		std::vector<int16_t> outputNeuronLabelAssignment;
	};
}