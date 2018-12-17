/*
 * rewardModulatedSTDP.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/12/2018
 *
 * Information: The rewardModulatedSTDP learning rule has to be on a postsynaptic layer because it automatically detects the presynaptic layer.
 */

#pragma once

#include "globalLearningRuleHandler.hpp"

namespace adonis_c
{
	struct reinforcementLayers
	{
		int postLayer;
		int preLayer;
	};
	
	class Neuron;
	
	class RewardModulatedSTDP : public GlobalLearningRuleHandler
	{
	public:
		// ----- CONSTRUCTOR -----
		RewardModulatedSTDP(float _Ar_plus=1, float _Ar_minus=-1, float _Ap_plus=1, float _Ap_minus=-1) :
		Ar_plus(_Ar_plus),
		Ar_minus(_Ar_minus),
		Ap_plus(_Ap_plus),
		Ap_minus(_Ap_minus)
		{
			if (Ar_plus <= 0 || Ap_plus <= 0)
			{
				throw std::logic_error("Ar_plus and Ap_plus need to be positive");
			}
			
			if (Ar_minus >= 0 || Ap_minus >= 0)
			{
				throw std::logic_error("Ar_minus and Ap_minus need to be negative");
			}
		}
		
		// ----- PUBLIC METHODS -----
		virtual void onStart(Network* network) override
		{
			for (auto& l: network->getLayers())
			{
				for (auto& rule: network->getNeurons()[l.sublayers[0].receptiveFields[0].neurons[0]].getLearningRuleHandler())
				{
					if (rule == this)
					{
						if (network->getNeurons()[l.sublayers[0].receptiveFields[0].neurons[0]].getLayerID()-1 >= 0)
						{
							rl.emplace_back(reinforcementLayers{network->getNeurons()[l.sublayers[0].receptiveFields[0].neurons[0]].getLayerID(), network->getNeurons()[l.sublayers[0].receptiveFields[0].neurons[0]].getLayerID()-1});
						}
						else
						{
							throw std::logic_error("the reward-modulated STDP learning rule has to be on a postsynaptic layer");
						}
					}
				}
			}
			
			// add rstdp to decision-making layer which is on the last layer
			for (auto& sub: network->getLayers().back().sublayers)
			{
				for (auto& rf: sub.receptiveFields)
				{
					for (auto& n: rf.neurons)
					{
						if (network->getNeurons()[n].getClassLabel() != "")
						{
							network->getNeurons()[n].addLearningRule(this);
						}
					}
				}
			}
		}
		
		virtual void learn(double timestamp, Neuron* neuron, Network* network) override
		{
			if (neuron->getLayerID() == network->getLayers().back().ID)
			{
				// reward and punishement signal from the decision-making layer
				int alpha = 0;
				int beta = 0;
				if (neuron->getClassLabel() == network->getCurrentLabel())
				{
					alpha = 1;
				}
				else
				{
					beta = 1;
				}
				
				// propagating the error signal to every layer using the R-STDP learning rule
				for (auto& layer: rl)
				{
					// if preTime - postTime is positive
					for (auto& sub: network->getLayers()[layer.preLayer].sublayers)
					{
						for (auto& rf: sub.receptiveFields)
						{
							for (auto& n: rf.neurons)
							{
								if (network->getNeurons()[n].getEligibilityTrace() > 0.1)
								{
									for (auto& postAxon: network->getNeurons()[n].getPostAxons())
									{
										if (postAxon->postNeuron->getEligibilityTrace() > 0.1)
										{
											double delta = alpha*Ar_minus+beta*Ap_plus;
											postAxon->weight += delta * postAxon->weight * (1./postAxon->postNeuron->getInputResistance() - postAxon->weight);
										}
									}
								}
							}
						}
					}

					// if preTime - postTime is negative
					for (auto& sub: network->getLayers()[layer.postLayer].sublayers)
					{
						for (auto& rf: sub.receptiveFields)
						{
							for (auto& n: rf.neurons)
							{
								if (network->getNeurons()[n].getEligibilityTrace() > 0.1)
								{
									for (auto& preAxon: network->getNeurons()[n].getPreAxons())
									{
										if (preAxon->preNeuron->getEligibilityTrace() > 0.1)
										{
											double delta = alpha*Ar_plus+beta*Ap_minus;
											preAxon->weight += delta * preAxon->weight * (1./preAxon->preNeuron->getInputResistance() - preAxon->weight);
										}
									}
								}
							}
						}
					}
				}
			}
		}
		
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
		std::vector<reinforcementLayers> rl;
		float                            Ar_plus;
		float                            Ar_minus;
		float                            Ap_plus;
		float                            Ap_minus;
	};
}
