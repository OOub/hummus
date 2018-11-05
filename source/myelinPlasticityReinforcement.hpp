/*
 * myelinPlasticityReinforcement.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 25/09/2018
 *
 * Information: The MyelinPlasticityReinforcement class
 */

#pragma once

namespace adonis_c
{
	class Neuron;
	
	class MyelinPlasticityReinforcement : public LearningRuleHandler
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		MyelinPlasticityReinforcement(float _alpha=1, float _lambda=1) :
		alpha(_alpha),
		lambda(_lambda)
		{}
		
		// ----- PUBLIC METHODS -----
		virtual void learn(double timestamp, Neuron* neuron, Network* network) override
		{
			std::vector<double> timeDifferences;
			std::vector<int16_t> plasticID;
			std::vector<std::vector<int16_t>> plasticCoordinates(4);
			bool supervise = false;
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
					if (network->getTeachingProgress()) // supervised learning
					{
						if (timestamp >= network->getTeacher()->front() - neuron->getDecayPotential() || timestamp <= network->getTeacher()->front()+neuron->getDecayPotential())
						{
							timeDifferences.push_back(network->getTeacher()->front() - inputProjection->lastInputTime - inputProjection->delay);
							supervise = true;
							if (timeDifferences.back() > 0)
							{
								change = lambda*(neuron->getInputResistance()/(neuron->getDecayCurrent()-neuron->getDecayPotential())) * neuron->getCurrent() * (std::exp(-alpha*timeDifferences.back()/neuron->getDecayCurrent()) - std::exp(-alpha*timeDifferences.back()/neuron->getDecayPotential()))*neuron->getSynapticEfficacy();
								inputProjection->delay += change;
								#ifndef NDEBUG
								std::cout << inputProjection->preNeuron->getLayerID() << " " << inputProjection->preNeuron->getNeuronID() << " " << inputProjection->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << change << " weight: " << inputProjection->weight << std::endl;
								#endif
							}

							else if (timeDifferences.back() < 0)
							{
								change = -lambda*((neuron->getInputResistance()/(neuron->getDecayCurrent()-neuron->getDecayPotential())) * neuron->getCurrent() * (std::exp(alpha*timeDifferences.back()/neuron->getDecayCurrent()) - std::exp(alpha*timeDifferences.back()/neuron->getDecayPotential())))*neuron->getSynapticEfficacy();
								inputProjection->delay += change;
								#ifndef NDEBUG
								std::cout << inputProjection->preNeuron->getLayerID() << " " << inputProjection->preNeuron->getNeuronID() << " " << inputProjection->postNeuron->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << change << " weight: " << inputProjection->weight << std::endl;
								#endif
							}
						}
						else
						{
							timeDifferences.push_back(0);
						}
						neuron->setSynapticEfficacy(-std::exp(-std::pow(timeDifferences.back(),2))+1);
					}
					else // unsupervised learning
					{
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
			}
			
			if (supervise)
			{
				network->getTeacher()->pop_front();
			}
			
			for (auto delegate: network->getStandardDelegates())
			{
				delegate->learningEpoch(timestamp, network, neuron, timeDifferences, plasticCoordinates);
			}
			
			if (network->getMainThreadDelegate())
			{
				network->getMainThreadDelegate()->learningEpoch(timestamp, network, neuron, timeDifferences, plasticCoordinates);
			}
			
			// weight reinforcement
			// looping through all projections from the winner
            for (auto& allProjections: neuron->getPreProjections())
            {
                int16_t ID = allProjections->preNeuron->getNeuronID();
                // if the projection is plastic
                if (std::find(plasticID.begin(), plasticID.end(), ID) != plasticID.end())
                {
					// positive reinforcement on winner projections
					if (allProjections->weight < (1/neuron->getInputResistance())/plasticID.size())
                    {
                     	allProjections->weight += allProjections->weight*neuron->getSynapticEfficacy()*0.1/plasticID.size();
                    }
                }
                else
                {
                    if (allProjections->weight > 0)
                    {
                        // negative reinforcement on other projections going towards the winner to prevent other neurons from triggering it
                        allProjections->weight -= allProjections->weight*neuron->getSynapticEfficacy()*0.1/plasticID.size();
						if (allProjections->weight < 0)
						{
							allProjections->weight = 0;
						}
                    }
				}
            }
		}
		
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
		float alpha;
		float lambda;
	};
}
