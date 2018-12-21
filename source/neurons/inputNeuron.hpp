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
	class InputNeuron : public PreNeuron
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		InputNeuron(int16_t _neuronID, int16_t _rfRow=0, int16_t _rfCol=0, int16_t _sublayerID=0, int16_t _layerID=0, int16_t _xCoordinate=-1, int16_t _yCoordinate=-1, std::vector<LearningRuleHandler*> _learningRuleHandler={}, float _threshold=-50, float _restingPotential=-70) :
			PreNeuron(_neuronID, _rfRow, _rfCol, _sublayerID, _layerID, _xCoordinate, _yCoordinate),
			threshold(_threshold),
			potential(_restingPotential),
			learningRuleHandler(_learningRuleHandler)
		{}
		
		virtual ~InputNeuron(){}
		
		// ----- PUBLIC INPUT NEURON METHODS -----
		virtual void initialisation(Network* network) override
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
		
		virtual void update(double timestamp, axon* a, Network* network) override
		{
			throw std::logic_error("not implemented yet");
		}
		
		virtual void updateSync(double timestamp, axon* a, Network* network) override
		{
			throw std::logic_error("not implemented yet");
		}
		
		// ----- SETTERS AND GETTERS -----
		float getThreshold() const
        {
            return threshold;
        }
		
        float getPotential() const
        {
            return potential;
        }
	
		std::vector<LearningRuleHandler*> getLearningRuleHandler() const
		{
			return learningRuleHandler;
		}
		
		void addLearningRule(LearningRuleHandler* newRule)
		{
			learningRuleHandler.emplace_back(newRule);
		}
		
	protected:
	
		// ----- INPUT NEURON PARAMETERS -----
		float                              threshold;
		float                              potential;
		std::vector<LearningRuleHandler*>  learningRuleHandler;
	};
}
