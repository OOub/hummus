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
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		Stdp(float _A_plus=1, float _A_minus=1, float _tau_plus=20, float _tau_minus=20) :
		A_plus(_A_plus),
		A_minus(_A_minus),
		tau_plus(_tau_plus),
		tau_minus(_tau_minus)
		{}
		
		// ----- PUBLIC METHODS -----
		virtual void learn(double timestamp, Neuron* neuron, Network* network) override
		{		
			for (auto inputProjection: neuron->getPreProjections())
			{
				if (inputProjection->preNeuron->getEligibilityTrace() > 0.1)
				{
					float preTrace = inputProjection->preNeuron->getPlasticityTrace()*A_plus*std::exp(-(timestamp - inputProjection->preNeuron->getLastSpikeTime())/tau_plus);
					inputProjection->preNeuron->setPlasticityTrace(preTrace);
					
					inputProjection->weight += preTrace*(1/inputProjection->preNeuron->getInputResistance());
				}
			}
		
			for (auto& outputProjection: neuron->getPostProjections())
			{
				if (outputProjection->postNeuron->getEligibilityTrace() > 0.1)
				{
					float postTrace = outputProjection->postNeuron->getPlasticityTrace()*A_minus*std::exp(-(timestamp - outputProjection->postNeuron->getLastSpikeTime())/tau_minus);
					outputProjection->postNeuron->setPlasticityTrace(postTrace);

					outputProjection->weight -= postTrace*(1/outputProjection->postNeuron->getInputResistance());
				}
			}
		}
		
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
		float A_plus;
		float A_minus;
		float tau_plus;
		float tau_minus;
	};
}
