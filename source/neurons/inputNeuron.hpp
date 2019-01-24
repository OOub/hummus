/*
 * inputNeuron.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: input neurons take in spikes or events and instantly propagate them in the network. The potential does not decay and there is no refractory period. 
 */

#pragma once

#include "../core.hpp"

namespace adonis
{
	class InputNeuron : public Neuron
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		InputNeuron(int16_t _neuronID, int16_t _rfRow=0, int16_t _rfCol=0, int16_t _sublayerID=0, int16_t _layerID=0, int16_t _xCoordinate=-1, int16_t _yCoordinate=-1, std::vector<LearningRuleHandler*> _learningRuleHandler={}, float _eligibilityDecay=20, float _threshold=-50, float _restingPotential=-70, float _membraneResistance=50e9) :
			Neuron(_neuronID, _rfRow, _rfCol, _sublayerID, _layerID, _xCoordinate, _yCoordinate, _learningRuleHandler, _eligibilityDecay, _threshold, _restingPotential, _membraneResistance)
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
		
		void update(double timestamp, axon* a, Network* network) override
		{
            // eligibility trace decay
            eligibilityTrace *= std::exp(-(timestamp - previousSpikeTime)/eligibilityDecay);
            
            // instantly making the input neuron fire at every input spike
            a->previousInputTime = timestamp;
            potential = threshold;
            eligibilityTrace = 1;

            #ifndef NDEBUG
            std::cout << "t=" << timestamp << " " << neuronID << " w=" << a->weight << " d=" << a->delay << " --> INPUT" << std::endl;
            #endif
            
            if (network->getMainThreadAddOn())
            {
                network->getMainThreadAddOn()->incomingSpike(timestamp, a, network);
            }
            
            for (auto addon: network->getStandardAddOns())
            {
                addon->neuronFired(timestamp, a, network);
            }
            
            if (network->getMainThreadAddOn())
            {
                network->getMainThreadAddOn()->neuronFired(timestamp, a, network);
            }
            
            for (auto& p : postAxons)
            {
                network->injectGeneratedSpike(spike{timestamp + p->delay, p.get()});
            }
            
            requestLearning(timestamp, network);
            previousSpikeTime = timestamp;
            potential = restingPotential;
		}
        
        void updateSync(double timestamp, axon* a, Network* network, double timestep) override
        {
            if (timestamp != 0 && timestamp - previousSpikeTime == 0)
            {
                timestep = 0;
            }
            
            // eligibility trace decay
            eligibilityTrace *= std::exp(-timestep/eligibilityDecay);
            
            if (a)
            {
                a->previousInputTime = timestamp;
                potential = threshold;
                eligibilityTrace = 1;
                
                #ifndef NDEBUG
                std::cout << "t=" << timestamp << " " << neuronID << " w=" << a->weight << " d=" << a->delay << " --> INPUT" << std::endl;
                #endif
                
                if (network->getMainThreadAddOn())
                {
                    network->getMainThreadAddOn()->incomingSpike(timestamp, a, network);
                }
                
                for (auto addon: network->getStandardAddOns())
                {
                    addon->neuronFired(timestamp, a, network);
                }
                
                if (network->getMainThreadAddOn())
                {
                    network->getMainThreadAddOn()->neuronFired(timestamp, a, network);
                }
                
                for (auto& p : postAxons)
                {
                    network->injectGeneratedSpike(spike{timestamp + p->delay, p.get()});
                }
                
                requestLearning(timestamp, network);
                previousSpikeTime = timestamp;
                potential = restingPotential;
            }
            else
            {
                if (timestep > 0)
                {
                    for (auto addon: network->getStandardAddOns())
                    {
                        addon->timestep(timestamp, network, this);
                    }
                    if (network->getMainThreadAddOn())
                    {
                        network->getMainThreadAddOn()->timestep(timestamp, network, this);
                    }
                }
            }
        }
        
    protected:
        
        // loops through any learning rules and activates them
        void requestLearning(double timestamp, Network* network) override
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
