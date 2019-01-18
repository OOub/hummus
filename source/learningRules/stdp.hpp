/*
 * STDP.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: The STDP learning rule has to be on a postsynaptic layer because it automatically detects the presynaptic layer.
 */

#pragma once

#include "../globalLearningRuleHandler.hpp"
#include "../neurons/inputNeuron.hpp"
#include "../neurons/leakyIntegrateAndFire.hpp"

namespace adonis
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
				for (auto& rule: n->getLearningRuleHandler())
				{
					if(rule == this)
					{
						if (n->getLayerID() > 0)
						{
                            postLayer = n->getLayerID();
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
						network->getNeurons()[n]->addLearningRule(this);
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
                    // if a postNeuron fired, the deltaT (preTime - postTime) should be positive
                    if (postAxon->postNeuron->getEligibilityTrace() > 0.1)
                    {
                        float postTrace = - (timestamp - postAxon->postNeuron->getPreviousSpikeTime())/tau_minus * A_minus*std::exp(-(timestamp - postAxon->postNeuron->getPreviousSpikeTime())/tau_minus);
                        
                        if (postAxon->weight > 0)
                        {
                            postAxon->weight += postTrace*(1/postAxon->postNeuron->getMembraneResistance());

                            if (postAxon->weight < 0)
                            {
                                postAxon->weight = 0;
                            }
                        }                        
                    }
                }
            }
			
			// LTP whenever a neuron from the postsynaptic layer spikes
			else if (neuron->getLayerID() == postLayer)
			{
				for (auto& preAxon: neuron->getPreAxons())
				{
					// if a preNeuron already fired, the deltaT (preTime - postTime) should be negative
					if (preAxon->preNeuron->getEligibilityTrace() > 0.1)
					{
						float preTrace = -(preAxon->preNeuron->getPreviousSpikeTime() - timestamp)/tau_plus * A_plus*std::exp((preAxon->preNeuron->getPreviousSpikeTime() - timestamp)/tau_plus);

                        if (preAxon->weight < 1/preAxon->preNeuron->getMembraneResistance())
                        {
                            preAxon->weight += preTrace*(1/preAxon->preNeuron->getMembraneResistance());

                            if (preAxon->weight > 1/preAxon->preNeuron->getMembraneResistance())
                            {
                                preAxon->weight = 1/preAxon->preNeuron->getMembraneResistance();
                            }
                        }
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
