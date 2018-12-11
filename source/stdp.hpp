/*
 * STDP.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 25/09/2018
 *
 * Information: The STDP class
 */

#pragma once

namespace adonis_c
{
	class Neuron;
	
	class STDP : public LearningRuleHandler
	{
	public:
		// ----- CONSTRUCTOR -----
		STDP(int _preLayer, int _postLayer, float _A_plus=1, float _A_minus=1, float _tau_plus=20, float _tau_minus=20) :
		preLayer(_preLayer),
		postLayer(_postLayer),
		A_plus(_A_plus),
		A_minus(_A_minus),
		tau_plus(_tau_plus),
		tau_minus(_tau_minus)
		{}
		
		// ----- PUBLIC METHODS -----
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
						float postTrace = - (timestamp - postAxon->postNeuron->getLastSpikeTime())/tau_minus * A_minus*std::exp(-(timestamp - postAxon->postNeuron->getLastSpikeTime())/tau_minus);

						if (postAxon->weight > 0)
						{
							postAxon->weight += postTrace*(1/postAxon->postNeuron->getInputResistance());
							if (postAxon->weight < 0)
							{
								postAxon->weight = 0;
							}
						}
						postAxon->postNeuron->setPlasticityTrace(postTrace);
					}
				}
			}
			
			// LTP whenever a neuron from the postsynaptic layer spikes
			else if (neuron->getLayerID() == postLayer)
			{
				for (auto preAxon: neuron->getPreAxons())
				{
					// if a preNeuron already fired, the deltaT (preTime - postTime) should be negative
					if (preAxon->preNeuron->getEligibilityTrace() > 0.1)
					{
						float preTrace = -(preAxon->preNeuron->getLastSpikeTime() - timestamp)/tau_plus * A_plus*std::exp((preAxon->preNeuron->getLastSpikeTime() - timestamp)/tau_plus);

						if (preAxon->weight < 1/preAxon->preNeuron->getInputResistance())
						{
							preAxon->weight += preTrace*(1/preAxon->preNeuron->getInputResistance());
							if (preAxon->weight > 1/preAxon->preNeuron->getInputResistance())
							{
								preAxon->weight = 1/preAxon->preNeuron->getInputResistance();
							}
						}
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
