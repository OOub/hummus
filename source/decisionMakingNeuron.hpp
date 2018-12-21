/*
 * decisionMakingNeuron.hpp
 * Adonis - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/06/2018
 *
 * Information: Decision-making neuron for classification
 */

#pragma once

#include "core.hpp"

namespace adonis
{
	class DecisionMakingNeuron : public Neuron
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		DecisionMakingNeuron(int16_t _neuronID, int16_t _rfRow=0, int16_t _rfCol=0, int16_t _sublayerID=0, int16_t _layerID=0, std::string _classLabel="") :
			classLabel(_classLabel)
			{}
		
		virtual ~DecisionMakingNeuron(){}
		
		// ----- PUBLIC NEURON METHODS -----
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
		
		// ----- NEURON PARAMETERS -----
		std::vector<axon*> preAxons;
	};
}
