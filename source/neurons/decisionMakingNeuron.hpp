/*
 * decisionMakingNeuron.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: Decision-making neuron for classification
 */

#pragma once

#include "../core.hpp"
#include "leakyIntegrateAndFire.hpp"

namespace adonis
{
	class DecisionMakingNeuron : public LIF
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		DecisionMakingNeuron(int16_t _neuronID, int16_t _rfRow=0, int16_t _rfCol=0, int16_t _sublayerID=0, int16_t _layerID=0, int16_t _xCoordinate=-1, int16_t _yCoordinate=-1, std::vector<LearningRuleHandler*> _learningRuleHandler={},  bool _homeostasis=false, float _decayCurrent=10, float _decayPotential=20, int _refractoryPeriod=1000, float _eligibilityDecay=20, float _decayHomeostasis=10, float _homeostasisBeta=1, float _threshold=-50, float _restingPotential=-70, float _membraneResistance=50e9, float _externalCurrent=100, std::string _classLabel="") :
			LIF(_neuronID, _rfRow, _rfCol, _sublayerID, _layerID, _xCoordinate, _yCoordinate, _learningRuleHandler, _homeostasis, _decayCurrent, _decayPotential, _refractoryPeriod, true, false, _eligibilityDecay, _decayHomeostasis, _homeostasisBeta, _threshold, _restingPotential, _membraneResistance, _externalCurrent),
			classLabel(_classLabel)
			{}
		
		virtual ~DecisionMakingNeuron(){}
		
		// ----- SETTERS AND GETTERS -----
		std::string getClassLabel() const
		{
			return classLabel;
		}
		
		void setClassLabel(std::string newLabel)
		{
			classLabel = newLabel;
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
            WTA(timestamp, network);
        }
    
		// ----- DECISION-MAKING NEURON PARAMETERS -----
        std::string        classLabel;
	};
}
