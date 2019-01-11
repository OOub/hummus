/*
 * decisionMakingNeuron.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/06/2018
 *
 * Information: Decision-making neuron for classification
 */

#pragma once

#include "../core.hpp"

namespace adonis
{
	class DecisionMakingNeuron : public Neuron
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		DecisionMakingNeuron(int16_t _neuronID, int16_t _rfRow=0, int16_t _rfCol=0, int16_t _sublayerID=0, int16_t _layerID=0, int16_t _xCoordinate=-1, int16_t _yCoordinate=-1, std::vector<LearningRuleHandler*> _learningRuleHandler={}, float _threshold=-50, float _restingPotential=-70, float _membraneResistance=1, int _refractoryPeriod=1000, std::string _classLabel="") :
			Neuron(_neuronID, _rfRow, _rfCol, _sublayerID, _layerID, _xCoordinate, _yCoordinate, _learningRuleHandler, _threshold, _restingPotential, _membraneResistance),
            refractoryPeriod(_refractoryPeriod),
			classLabel(_classLabel)
			{}
		
		virtual ~DecisionMakingNeuron(){}
		
		// ----- PUBLIC DECISION-MAKING NEURON METHODS -----
		void update(double timestamp, axon* a, Network* network, double timestep) override
		{
			throw std::logic_error("not implemented yet");
		}
		
		void updateSync(double timestamp, axon* a, Network* network, double timestep) override
		{
			throw std::logic_error("not implemented yet");
		}
		
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
        
        // winner-take-all algorithm
        void WTA(double timestamp, Network* network) override
        {
            for (auto rf: network->getLayers()[layerID].sublayers[sublayerID].receptiveFields)
            {
                if (rf.row == rfRow && rf.col == rfCol)
                {
                    for (auto n: rf.neurons)
                    {
                        if (network->getNeurons()[n]->getNeuronID() != neuronID)
                        {
                            network->getNeurons()[n]->setPotential(restingPotential);
                        }
                    }
                }
            }
        }
        
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
        int                refractoryPeriod;
	};
}
