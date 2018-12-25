/*
 * stdp.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/12/2018
 *
 * Information: The STDP learning rule has to be on a postsynaptic layer because it automatically detects the presynaptic layer.
 */

#pragma once

#include "globalLearningRuleHandler.hpp"

namespace adonis_c
{
	class Neuron;
	
	class STDP : public GlobalLearningRuleHandler
	{
	public:
		// ----- CONSTRUCTOR -----
		STDP(float _A_plus=1, float _A_minus=1, float _tau_plus=20, float _tau_minus=20) :
		A_plus(_A_plus),
		A_minus(_A_minus),
		tau_plus(_tau_plus),
		tau_minus(_tau_minus)
		{}
		
		// ----- PUBLIC METHODS -----
		virtual void onStart(Network* network) override
		{
			for (auto& n: network->getNeurons())
			{
				for (auto& rule: n.getLearningRuleHandler())
				{
					if(rule == this)
					{
						if (n.getLayerID() > 0)
						{
						postLayer = n.getLayerID();
						preLayer = postLayer-1;
						}
						else
						{
							throw std::logic_error("the STDP learning rule has to be on a postsynaptic layer");
						}
					}
				}
			}
			
			for (auto& sub: network->getLayers()[preLayer].sublayers)
			{
				for (auto& rf: sub.receptiveFields)
				{
					for (auto& n: rf.neurons)
					{
						network->getNeurons()[n].addLearningRule(this);
					}
				}
			}
		}
		
		virtual void learn(double timestamp, Neuron* neuron, Network* network) override
		{
			// LTD whenever a neuron from the presynaptic layer spikes
			if (neuron->getLayerID() == preLayer)
			{
				for (auto& postAxon: neuron->getPostAxons())
				{
					// if a postNeuron fired, the deltaT (preTime - postTime) should be positive (negative postTrace)
					if (postAxon->weight >= 0 && postAxon->postNeuron->getEligibilityTrace() > 0.1)
					{
						float postTrace = (-(timestamp - postAxon->postNeuron->getLastSpikeTime())/tau_minus * A_minus*std::exp(-(timestamp - postAxon->postNeuron->getLastSpikeTime())/tau_minus));//*neuron->getSynapticEfficacy();

						postAxon->weight += postTrace*(1/postAxon->postNeuron->getInputResistance()) * postAxon->weight;
						postAxon->postNeuron->setPlasticityTrace(postTrace);
					}
				}
			}
			
			// LTP whenever a neuron from the postsynaptic layer spikes
			else if (neuron->getLayerID() == postLayer)
			{
				for (auto preAxon: neuron->getPreAxons())
				{
					// if a preNeuron already fired, the deltaT (preTime - postTime) should be negative (positive preTrace)
					if (preAxon->weight >= 0 && preAxon->preNeuron->getEligibilityTrace() > 0.1)
					{
						float preTrace = (-(preAxon->preNeuron->getLastSpikeTime() - timestamp)/tau_plus * A_plus*std::exp((preAxon->preNeuron->getLastSpikeTime() - timestamp)/tau_plus));//*neuron->getSynapticEfficacy();

						preAxon->weight += preTrace*(1/preAxon->preNeuron->getInputResistance()) * (1./preAxon->preNeuron->getInputResistance()-preAxon->weight);
						preAxon->preNeuron->setPlasticityTrace(preTrace);
					}
				}
			}
		}
		
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
		int preLayer;
		int postLayer;
		float A_plus;
		float A_minus;
		float tau_plus;
		float tau_minus;
	};
}
