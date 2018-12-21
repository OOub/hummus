/*
 * inputNeuron.hpp
 * Adonis - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/06/2018
 *
 * Information: input neuron which takes in spikes to distribute to the rest of the network
 */

#pragma once

#include "core.hpp"

namespace adonis
{
	class InputNeuron : public Neuron
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		InputNeuron(int16_t _neuronID, int16_t _rfRow=0, int16_t _rfCol=0, int16_t _sublayerID=0, int16_t _layerID=0, int16_t _xCoordinate=-1, int16_t _yCoordinate=-1, std::vector<LearningRuleHandler*> _learningRuleHandler={}, float _threshold=-50, float _restingPotential=-70)
			threshold(_threshold),
			potential(_restingPotential),
			initialAxon{nullptr, nullptr, 1, 0, -1},
			learningRuleHandler(_learningRuleHandler),
			xCoordinate(_xCoordinate),
            yCoordinate(_yCoordinate),
		{}
		
		virtual ~InputNeuron(){}
		
		// ----- PUBLIC NEURON METHODS -----
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
		{}
		
		spike prepareInitialSpike(double timestamp)
        {
            if (!initialAxon.postNeuron)
            {
                initialAxon.postNeuron = this;
            }
            return spike{timestamp, &initialAxon};
        }
		
		void addAxon(Neuron* postNeuron, float weight=1., int delay=0, int probability=100, bool redundantConnections=true)
        {
            if (postNeuron)
            {
            	if (connectionProbability(probability))
            	{
					if (redundantConnections == false)
					{
						int16_t ID = postNeuron->neuronID;
						auto result = std::find_if(postAxons.begin(), postAxons.end(), [ID](const std::unique_ptr<axon>& p){return p->postNeuron->neuronID == ID;});
						
						if (result == postAxons.end())
						{
							postAxons.emplace_back(new axon{this, postNeuron, weight*(1/inputResistance), delay, -1});
							postNeuron->preAxons.push_back(postAxons.back().get());
						}
						else
						{
							#ifndef NDEBUG
							std::cout << "axon " << neuronID << "->" << postNeuron->neuronID << " already exists" << std::endl;
							#endif
						}
					}
					else
					{
						postAxons.emplace_back(new axon{this, postNeuron, weight*(1/inputResistance), delay, -1});
						postNeuron->preAxons.push_back(postAxons.back().get());
					}
                }
            }
            else
            {
                throw std::logic_error("Neuron does not exist");
            }
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
		
        axon* getInitialAxon()
		{
			return &initialAxon;
		}
	
		std::vector<LearningRuleHandler*> getLearningRuleHandler() const
		{
			return learningRuleHandler;
		}
		
		void addLearningRule(LearningRuleHandler* newRule)
		{
			learningRuleHandler.emplace_back(newRule);
		}
		
		int16_t getX() const
		{
		    return xCoordinate;
		}
		
		int16_t getY() const
		{
		    return yCoordinate;
		}
		
	protected:
	
		// ----- NEURON PARAMETERS -----
		float                              threshold;
		float                              potential;
		axon                               initialAxon;
		std::vector<LearningRuleHandler*>  learningRuleHandler;
		int16_t                            xCoordinate;
		int16_t                            yCoordinate;
	};
}
