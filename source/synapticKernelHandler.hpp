/*
 * synapticKernelHandler.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 25/09/2018
 *
 * Information: The SynapticKernelHandler class is called from a neuron to apply a synaptic kernel and update the current
 */

#pragma once

#include "dependencies/json.hpp"

namespace hummus {
	class Network;
	class Neuron;
	
	class SynapticKernelHandler {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		SynapticKernelHandler() :
				gaussianStdDev(0),
				type(0),
				kernelID(0),
				synapseTimeConstant(0) {}
		
		virtual ~SynapticKernelHandler(){}
		
		// ----- PUBLIC METHODS -----
		
		// pure virtual method that updates the status of current before integrating a spike
        virtual double updateCurrent(double timestamp, double timestep, double previousInputTime, float neuronCurrent) = 0;
		
        // pure virtual method that outputs an updated current value
		virtual float integrateSpike(float neuronCurrent, float externalCurrent, double synapseWeight) = 0;
		
		// write synaptic kernel parameters in a JSON format
        virtual void toJson(nlohmann::json& output) {}
		
		// ----- SETTERS AND GETTERS -----
		float getSynapseTimeConstant() const {
			return synapseTimeConstant;
		}
		
		int getKernelID() const {
			return kernelID;
		}
		
		void setKernelID(int ID) {
			kernelID = ID;
		}
		
	protected:
		int   kernelID;
		float gaussianStdDev;
		int   type;
		float synapseTimeConstant;
	};
}

