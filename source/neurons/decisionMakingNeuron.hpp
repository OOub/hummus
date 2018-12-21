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
		DecisionMakingNeuron(int16_t _neuronID, int16_t _rfRow=0, int16_t _rfCol=0, int16_t _sublayerID=0, int16_t _layerID=0, int16_t _xCoordinate=-1, int16_t _yCoordinate=-1, std::string _classLabel="") :
			Neuron(_neuronID, _rfRow, _rfCol, _sublayerID, _layerID, _xCoordinate, _yCoordinate),
			classLabel(_classLabel)
			{}
		
		virtual ~DecisionMakingNeuron(){}
		
		// ----- PUBLIC DECISION-MAKING NEURON METHODS -----
		void update(double timestamp, axon* a, Network* network) override
		{
			throw std::logic_error("not implemented yet");
		}
		
		void updateSync(double timestamp, axon* a, Network* network) override
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
		
		std::vector<axon*>& getPreAxons()
		{
			return preAxons;
		}
		
		// ----- DECISION-MAKING NEURON PARAMETERS -----
		std::vector<axon*> preAxons;
	};
}
