/*
 * rewardModulatedSTDP.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 11/12/2018
 *
 * Information: The rewardModulatedSTDP learning rule
 */

#pragma once

#include "dataParser.hpp"

namespace adonis_c
{
	class Neuron;
	
	class RewardModulatedSTDP : public LearningRuleHandler
	{
	public:
		// ----- CONSTRUCTOR -----
		RewardModulatedSTDP() = default;
		
		// ----- PUBLIC METHODS -----
		virtual void learn(double timestamp, Neuron* neuron, Network* network) override
		{
			if (neuron->getLayerID() == network->getLayers().back().ID)
			{
				
				for (auto preAxon: neuron->getPreAxons())
				{
					// positive reinforcement for correct label
					if (neuron->getClassLabel() == network->getCurrentLabel())
					{
					}
					// negative reinforcement for incorrect label
					else
					{
					}
				}
			}
			else
			{
				throw std::logic_error("The reward modulation learning rule can only be used on the output layer");
			}
		}
	};
}
