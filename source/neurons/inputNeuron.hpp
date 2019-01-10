/*
 * inputNeuron.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/06/2018
 *
 * Information: input neuron which takes in spikes to distribute to the rest of the network
 */

#pragma once

#include "../core.hpp"

namespace adonis
{
	class InputNeuron : public Neuron
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		InputNeuron(int16_t _neuronID, int16_t _rfRow=0, int16_t _rfCol=0, int16_t _sublayerID=0, int16_t _layerID=0, int16_t _xCoordinate=-1, int16_t _yCoordinate=-1, std::vector<LearningRuleHandler*> _learningRuleHandler={}, float _threshold=1, float _restingPotential=0) :
			Neuron(_neuronID, _rfRow, _rfCol, _sublayerID, _layerID, _xCoordinate, _yCoordinate, _learningRuleHandler, _threshold, _restingPotential)
		{}
		
		virtual ~InputNeuron(){}
		
		// ----- PUBLIC INPUT NEURON METHODS -----
		void initialisation(Network* network) override
		{
			for (auto& rule: learningRuleHandler)
			{
				if(StandardAddOn* globalRule = dynamic_cast<StandardAddOn*>(rule))
				{
					if (std::find(network->getStandardAddOns().begin(), network->getStandardAddOns().end(), dynamic_cast<StandardAddOn*>(rule)) == network->getStandardAddOns().end())
					{
						network->getStandardAddOns().emplace_back(dynamic_cast<StandardAddOn*>(rule));
					}
				}
			}
		}
		
		void update(double timestamp, axon* a, Network* network, double timestep) override
		{
			throw std::logic_error("not implemented yet");
		}
		
		void updateSync(double timestamp, axon* a, Network* network, double timestep) override
		{
			throw std::logic_error("not implemented yet");
		}
        
    protected:
        
        // loops through any learning rules and activates them
        void learn(double timestamp, Network* network) override
        {
            if (network->getLearningStatus())
            {
                if (!learningRuleHandler.empty())
                {
                    for (auto& learningRule: learningRuleHandler)
                    {
                        learningRule->learn(timestamp, this, network);
                    }
                }
            }
        }
	};
}
