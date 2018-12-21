/*
 * leakyIntegrateAndFire.hpp
 * Adonis - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/06/2018
 *
 * Information: basic building blocks for biological neurons
 */

#pragma once

#include "core.hpp"

namespace adonis
{
	class BiologicalNeuron : public Neuron
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		BiologicalNeuron(int16_t _neuronID, int16_t _rfRow=0, int16_t _rfCol=0, int16_t _sublayerID=0, int16_t _layerID=0 int _refractoryPeriod=3, float _restingPotential=-70, float _decayCurrent=10, float _decayPotential=20, bool _burstingActivity=false, float _eligibilityDecay=20, float _threshold=-50, float _restingPotential=-70, float _inputResistance=50e9, float _externalCurrent=100, int16_t _xCoordinate=-1, int16_t _yCoordinate=-1, std::vector<LearningRuleHandler*> _learningRuleHandler={}, bool _homeostasis=false, float _decayHomeostasis=10, float _homeostasisBeta=1, bool _wta=false) :
    		refractoryPeriod(_refractoryPeriod),
    		potential(_restingPotential),
			learningRuleHandler(_learningRuleHandler),
			decayCurrent(_decayCurrent),
			decayPotential(_decayPotential),
			synapticEfficacy(1),
			threshold(_threshold),
			restingPotential(_restingPotential),
			inputResistance(_inputResistance),
			externalCurrent(_externalCurrent),
			current(0),
			active(true),
            lastSpikeTime(0),
            eligibilityTrace(0),
            eligibilityDecay(_eligibilityDecay),
            xCoordinate(_xCoordinate),
            yCoordinate(_yCoordinate),
			plasticityTrace(0),
			burstingActivity(_burstingActivity),
			homeostasis(_homeostasis),
			restingThreshold(-50),
			decayHomeostasis(_decayHomeostasis),
			homeostasisBeta(_homeostasisBeta),
			inhibited(false),
			inhibitionTime(0),
			wta(_wta)
		{
			// error handling
			if (decayCurrent == decayPotential)
            {
                throw std::logic_error("The current decay and the potential decay cannot be equal: a division by 0 occurs");
            }
			
			if (decayCurrent == 0)
            {
                throw std::logic_error("The current decay cannot be 0");
            }
			
    	    if (decayPotential == 0)
            {
                throw std::logic_error("The potential decay cannot be 0");
            }
		}
		
		virtual ~BiologicalNeuron(){}
		
		// ----- PUBLIC NEURON METHODS -----
		virtual void update(double timestamp, axon* a, Network* network) override = 0;
		
		virtual void updateSync(double timestamp, axon* a, Network* network){} override
		
		virtual void resetNeuron() override
		{
			lastSpikeTime = 0;
			current = 0;
			potential = restingPotential;
			eligibilityTrace = 0;
			inhibited = false;
			active = true;
			threshold = restingThreshold;
		}
		
		virtual void learn(double timestamp, Network* network)
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
			if (wta)
			{
				WTA(timestamp);
			}
			resetLearning();
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
		bool getActivity() const
		{
			return active;
		}
		
		float getThreshold() const
        {
            return threshold;
        }
		
        float setThreshold(float _threshold)
        {
            return threshold = _threshold;
        }
		
		float getDecayPotential() const
        {
            return decayPotential;
        }
		
        float getDecayCurrent() const
        {
            return decayCurrent;
        }
		
        float getCurrent() const
        {
        	return current;
		}
		
		void setCurrent(float newCurrent)
		{
			current = newCurrent;
		}
		
		float getExternalCurrent() const
		{
			return externalCurrent;
		}
		
		void setExternalCurrent(float newCurrent)
		{
			externalCurrent = newCurrent;
		}
		
		int16_t getX() const
		{
		    return xCoordinate;
		}
		
		int16_t getY() const
		{
		    return yCoordinate;
		}
		
		float getEligibilityTrace() const
		{
			return eligibilityTrace;
		}
		
		float getSynapticEfficacy() const
		{
			return synapticEfficacy;
		}
		
		float setSynapticEfficacy(float newEfficacy)
		{
			return synapticEfficacy = newEfficacy;
		}
		
		float getInputResistance() const
		{
			return inputResistance;
		}
		
		float getPlasticityTrace() const
		{
			return plasticityTrace;
		}
		
		void setPlasticityTrace(float newtrace)
		{
			plasticityTrace = newtrace;
		}
		
		double getLastSpikeTime() const
		{
			return lastSpikeTime;
		}
		
		void setInhibition(double timestamp, bool inhibitionStatus)
		{
			inhibitionTime = timestamp;
			inhibited = inhibitionStatus;
		}
		
		std::vector<LearningRuleHandler*> getLearningRuleHandler() const
		{
			return learningRuleHandler;
		}
		
		void addLearningRule(LearningRuleHandler* newRule)
		{
			learningRuleHandler.emplace_back(newRule);
		}
		
		float getPotential() const
        {
            return potential;
        }
		
        float setPotential(float newPotential)
        {
            return potential = newPotential;
        }
		
		std::vector<axon*>& getPreAxons()
		{
			return preAxons;
		}
		
		std::vector<std::unique_ptr<axon>>& getPostAxons()
		{
			return postAxons;
		}
		
	protected:
		
		// ----- NEURON BEHAVIOR -----
		void WTA(double timestamp, Network* network)
		{
			for (auto rf: network->getLayers()[layerID].sublayers[sublayerID].receptiveFields)
			{
				if (rf.row == rfRow && rf.col == rfCol)
				{
					for (auto n: rf.neurons)
					{
						if (network->getNeurons()[n].neuronID != neuronID)
						{
							network->getNeurons()[n].inhibited = true;
							network->getNeurons()[n].inhibitionTime = timestamp;
							network->getNeurons()[n].current = 0;
							network->getNeurons()[n].potential = restingPotential;
						}
					}
				}
			}
		}
		
		void resetLearning()
		{
			// resetting plastic neurons
			for (auto& inputAxon: preAxons)
			{
				inputAxon->preNeuron->eligibilityTrace = 0;
			}
		}
		
		// ----- NEURON PARAMETERS -----
		float                                    decayCurrent;
		float                                    decayPotential;
        float                                    threshold;
        float                                    inputResistance;
        float                                    current;
		bool                                     active;
		bool                                     inhibited;
		double                                   inhibitionTime;
		std::vector<axon*>                       preAxons;
		std::vector<std::unique_ptr<axon>>       postAxons;
		float                                    potential;
		float                                    restingPotential;
		float                                    refractoryPeriod;
		std::vector<LearningRuleHandler*>        learningRuleHandler;
		float                                    synapticEfficacy;
		float                                    externalCurrent;
		float                                    eligibilityTrace;
		float                                    eligibilityDecay;
		int16_t                                  xCoordinate;
		int16_t                                  yCoordinate;
		bool                                     burstingActivity;
		bool                                     homeostasis;
		float                                    restingThreshold;
		float                                    decayHomeostasis;
		float                                    homeostasisBeta;
		bool                                     wta;
		axon                                     activeAxon;
        double                                   lastSpikeTime;
        float                                    plasticityTrace;
        std::string                              classLabel;
	};
}
