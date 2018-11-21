/*
 * stdp.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 25/09/2018
 *
 * Information: The Stdp class
 */

#pragma once

namespace adonis_c
{
	class Neuron;
	
	class Stdp : public LearningRuleHandler
	{
	public:
		// ----- CONSTRUCTOR -----
		Stdp(int _preLayer, int _postLayer, float _A_plus=1, float _A_minus=1, float _tau_plus=20, float _tau_minus=20) :
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
				for (auto& postProjection: neuron->getPostProjections())
				{
					// if a postNeuron fired, the deltaT (preTime - postTime) should be positive
					if (postProjection->postNeuron->getEligibilityTrace() > 0.1)
					{
						float postTrace = - (timestamp - postProjection->postNeuron->getLastSpikeTime())/tau_minus * A_minus*std::exp(-(timestamp - postProjection->postNeuron->getLastSpikeTime())/tau_minus);

						if (postProjection->weight > 0)
						{
							postProjection->weight += postTrace*(1/postProjection->postNeuron->getInputResistance());
							if (postProjection->weight < 0)
							{
								postProjection->weight = 0;
							}
						}
						postProjection->postNeuron->setPlasticityTrace(postTrace);
					}
				}
			}
			
			// LTP whenever a neuron from the postsynaptic layer spikes
			else if (neuron->getLayerID() == postLayer)
			{
				for (auto preProjection: neuron->getPreProjections())
				{
					// if a preNeuron already fired, the deltaT (preTime - postTime) should be negative
					if (preProjection->preNeuron->getEligibilityTrace() > 0.1)
					{
						float preTrace = -(preProjection->preNeuron->getLastSpikeTime() - timestamp)/tau_plus * A_plus*std::exp((preProjection->preNeuron->getLastSpikeTime() - timestamp)/tau_plus);

						if (preProjection->weight < 1/preProjection->preNeuron->getInputResistance())
						{
							preProjection->weight += preTrace*(1/preProjection->preNeuron->getInputResistance());
							if (preProjection->weight > 1/preProjection->preNeuron->getInputResistance())
							{
								preProjection->weight = 1/preProjection->preNeuron->getInputResistance();
							}
						}
						preProjection->preNeuron->setPlasticityTrace(preTrace);
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
