/*
 * myelinPlasticity.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 25/09/2018
 *
 * Information: The MyelinPlasticity class
 */

#pragma once

namespace adonis_c
{
	class Neuron;
	
	class MyelinPlasticity : public LearningRuleHandler
	{
	public:
		// ----- CONSTRUCTOR -----
		MyelinPlasticity(float _alpha=1, float _lambda=1) :
		alpha(_alpha),
		lambda(_lambda)
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
			
			for (auto inputProjection: neuron->getPreProjections())
			{
				// selecting plastic neurons
				if (inputProjection->preNeuron->getEligibilityTrace() > 0.1)
				{
					plasticID.push_back(inputProjection->preNeuron->getNeuronID());
					plasticCoordinates[0].push_back(inputProjection->preNeuron->getX());
					plasticCoordinates[1].push_back(inputProjection->preNeuron->getY());
					plasticCoordinates[2].push_back(inputProjection->preNeuron->getRfRow());
					plasticCoordinates[3].push_back(inputProjection->preNeuron->getRfCol());
					
					float change = 0;
					
					timeDifferences.push_back(timestamp - inputProjection->lastInputTime - inputProjection->delay);
					if (timeDifferences.back() > 0)
					{
						change = lambda*(neuron->getInputResistance()/(neuron->getDecayCurrent()-neuron->getDecayPotential())) * neuron->getCurrent() * (std::exp(-alpha*timeDifferences.back()/neuron->getDecayCurrent()) - std::exp(-alpha*timeDifferences.back()/neuron->getDecayPotential()))*neuron->getSynapticEfficacy();
						inputProjection->delay += change;
						#ifndef NDEBUG
						std::cout << inputProjection->preNeuron->getLayerID() << " " << inputProjection->preNeuron->getNeuronID() << " " << inputProjection->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << change << std::endl;
						#endif
					}

					else if (timeDifferences.back() < 0)
					{
						change = -lambda*((neuron->getInputResistance()/(neuron->getDecayCurrent()-neuron->getDecayPotential())) * neuron->getCurrent() * (std::exp(alpha*timeDifferences.back()/neuron->getDecayCurrent()) - std::exp(alpha*timeDifferences.back()/neuron->getDecayPotential())))*neuron->getSynapticEfficacy();
						inputProjection->delay += change;
						#ifndef NDEBUG
						std::cout << inputProjection->preNeuron->getLayerID() << " " << inputProjection->preNeuron->getNeuronID() << " " << inputProjection->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << change << std::endl;
						#endif
					}
					neuron->setSynapticEfficacy(-std::exp(-std::pow(timeDifferences.back(),2))+1);

				}
			}
			
			for (auto delegate: network->getStandardDelegates())
			{
				delegate->learningEpoch(timestamp, network, neuron, timeDifferences, plasticCoordinates);
			}
			
			if (network->getMainThreadDelegate())
			{
				network->getMainThreadDelegate()->learningEpoch(timestamp, network, neuron, timeDifferences, plasticCoordinates);
			}
		}
		
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
		float alpha;
		float lambda;
	};
}
