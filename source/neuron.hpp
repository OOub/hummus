/*
 * neuron.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 25/09/2018
 *
 * Information: The Neuron class defines a neuron and its parameters. It can take in a pointer to a LearningRuleHandler object to define which learning rule it follows. The weight is automatically scaled depending on the input resistance used
 */

#pragma once

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <deque>

#include "networkAddOn.hpp"
#include "learningRuleHandler.hpp"

namespace adonis_c
{
	class Neuron;
	
	struct axon
	{
		Neuron*     preNeuron;
		Neuron*     postNeuron;
		float       weight;
		float       delay;
		double      lastInputTime;
	};
	
	struct spike
    {
        double      timestamp;
        axon*       axon;
    };
	
    class Neuron
    {
    public:
		
    	// ----- CONSTRUCTOR AND DESTRUCTOR -----
    	Neuron(int16_t _neuronID, int16_t _rfRow=0, int16_t _rfCol=0, int16_t _sublayerID=0, int16_t _layerID=0, float _decayCurrent=10, float _decayPotential=20, int _refractoryPeriod=3, bool _burstingActivity=false, float _eligibilityDecay=20, float _threshold=-50, float _restingPotential=-70, float _resetPotential=-70, float _inputResistance=50e9, float _externalCurrent=100, int16_t _xCoordinate=-1, int16_t _yCoordinate=-1, std::vector<LearningRuleHandler*> _learningRuleHandler={}) :
			neuronID(_neuronID),
			rfRow(_rfRow),
			rfCol(_rfCol),
			sublayerID(_sublayerID),
			layerID(_layerID),
			decayCurrent(_decayCurrent),
			decayPotential(_decayPotential),
			refractoryPeriod(_refractoryPeriod),
			synapticEfficacy(1),
			threshold(_threshold),
			restingPotential(_restingPotential),
			resetPotential(_resetPotential),
			inputResistance(_inputResistance),
			externalCurrent(_externalCurrent),
			current(0),
			potential(_restingPotential),
			active(true),
			inhibited(false),
			inhibitionTime(0),
			initialAxon{nullptr, nullptr, 100/_inputResistance, 0, -1},
            lastSpikeTime(0),
            eligibilityTrace(0),
            eligibilityDecay(_eligibilityDecay),
            xCoordinate(_xCoordinate),
            yCoordinate(_yCoordinate),
			learningRuleHandler(_learningRuleHandler),
			plasticityTrace(0),
			burstingActivity(_burstingActivity)
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
		
		Neuron(const Neuron&) = delete; // copy constructor
        Neuron(Neuron&&) = default; // move constructor
        Neuron& operator=(const Neuron&) = delete; // copy assign constructor
        Neuron& operator=(Neuron&&) = default; // move assign constructor
		
		// ----- PUBLIC NEURON METHODS -----
		void addAxon(Neuron* postNeuron, float weight, float delay, bool redundantConnections=true)
        {
            if (postNeuron)
            {
                if (redundantConnections == false)
                {
                    int16_t ID = postNeuron->neuronID;
                    auto result = std::find_if(postAxons.begin(), postAxons.end(), [ID](const std::unique_ptr<axon>& p){return p->postNeuron->neuronID == ID;});
                    
                    if (result == postAxons.end())
                    {
                        postAxons.emplace_back(std::unique_ptr<axon>(new axon{this, postNeuron, weight*(1/inputResistance), delay, -1}));
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
                    postAxons.emplace_back(std::unique_ptr<axon>(new axon{this, postNeuron, weight*(1/inputResistance), delay, -1}));
                    postNeuron->preAxons.push_back(postAxons.back().get());
                }
            }
            else
            {
                throw std::logic_error("Neuron does not exist");
            }
        }
		
		template<typename Network>
		void update(double timestamp, float timestep, spike s, Network* network)
		{
			if (inhibited && timestamp - inhibitionTime >= refractoryPeriod)
			{
				inhibited = false;
			}
			
            if (timestamp - lastSpikeTime >= refractoryPeriod)
            {
                active = true;
            }
			
			// current decay
			current *= std::exp(-timestep/decayCurrent);
			eligibilityTrace *= std::exp(-timestep/eligibilityDecay);
			
			// potential decay
			potential = restingPotential + (potential-restingPotential)*std::exp(-timestep/decayPotential);
			
			// neuron inactive during refractory period
			if (active && !inhibited)
			{
				if (s.axon)
				{
					current += externalCurrent*s.axon->weight;
					activeAxon = *s.axon;
					s.axon->lastInputTime = timestamp;
				}
				potential += (inputResistance*decayCurrent/(decayCurrent - decayPotential)) * current * (std::exp(-timestep/decayCurrent) - std::exp(-timestep/decayPotential));
			}
			
			if (s.axon)
			{
				#ifndef NDEBUG
				std::cout << "t=" << timestamp << " " << (s.axon->preNeuron ? s.axon->preNeuron->getNeuronID() : -1) << "->" << neuronID << " w=" << s.axon->weight << " d=" << s.axon->delay <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << "--> EMITTED" << std::endl;
				#endif
				for (auto addon: network->getStandardAddOns())
				{
					if (potential < threshold)
					{
						addon->incomingSpike(timestamp, s.axon, network);
					}
				}
				if (network->getMainThreadAddOn())
				{
					network->getMainThreadAddOn()->incomingSpike(timestamp, s.axon, network);
				}
			}
			else
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
			
			if (potential >= threshold)
			{
				eligibilityTrace = 1;
				plasticityTrace += 1;
				
				#ifndef NDEBUG
				std::cout << "t=" << timestamp << " " << (activeAxon.preNeuron ? activeAxon.preNeuron->getNeuronID() : -1) << "->" << neuronID << " w=" << activeAxon.weight << " d=" << activeAxon.delay <<" V=" << potential << " Vth=" << threshold << " layer=" << layerID << "--> SPIKED" << std::endl;
				#endif
				
				for (auto addon: network->getStandardAddOns())
				{
					addon->neuronFired(timestamp, &activeAxon, network);
				}
				if (network->getMainThreadAddOn())
				{
					network->getMainThreadAddOn()->neuronFired(timestamp, &activeAxon, network);
				}
				
				for (auto& p : postAxons)
				{
					network->injectGeneratedSpike(spike{timestamp + p->delay, p.get()});
				}
			
				learn(timestamp, network);
				
				lastSpikeTime = timestamp;
				potential = resetPotential;
				if (!burstingActivity)
				{
					current = 0;
				}
				active = false;
			}
		}
		
		spike prepareInitialSpike(double timestamp)
        {
            if (!initialAxon.postNeuron)
            {
                initialAxon.postNeuron = this;
            }
            return spike{timestamp, &initialAxon};
        }
	
    	// ----- SETTERS AND GETTERS -----
		bool getActivity() const
		{
			return active;
		}
		
		int16_t getNeuronID() const
        {
            return neuronID;
        }
		
		int16_t getRfRow() const
		{
			return rfRow;
		}
		
		int16_t getRfCol() const
		{
			return rfCol;
		}
		
		int16_t getSublayerID() const
		{
			return sublayerID;
		}
		
		int16_t getLayerID() const
		{
			return layerID;
		}
		
		float getThreshold() const
        {
            return threshold;
        }
		
        float setThreshold(float _threshold)
        {
            return threshold = _threshold;
        }
		
        float getPotential() const
        {
            return potential;
        }
		
        float setPotential(float newPotential)
        {
            return potential = newPotential;
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
		
		std::vector<axon*>& getPreAxons()
		{
			return preAxons;
		}
		
		std::vector<std::unique_ptr<axon>>& getPostAxons()
		{
			return postAxons;
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
		
		axon* getInitialAxon()
		{
			return &initialAxon;
		}
		
	protected:
		
		// ----- LEARNING RULE -----
		
		// calls a learningRuleHandler to add a learning rule to the neuron
		template<typename Network>
		void learn(double timestamp, Network* network)
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
			lateralInhibition(timestamp, network);
			resetLearning(network);
		}
		
		// ----- NEURON BEHAVIOR -----
		template<typename Network>
		void lateralInhibition(double timestamp, Network* network)
		{
			if (preAxons.size() > 0)
			{
				for (auto& projReset: preAxons[0]->preNeuron->postAxons)
				{
					if (projReset->postNeuron->neuronID != neuronID)
					{
						projReset->postNeuron->inhibited = true;
						projReset->postNeuron->inhibitionTime = timestamp;
						projReset->postNeuron->setCurrent(0);
						projReset->postNeuron->setPotential(restingPotential);
					}
				}
			}
		}
		
		template<typename Network>
		void resetLearning(Network* network)
		{
			// clearing generated spike list
			for (auto i=0; i<network->getGeneratedSpikes().size(); i++)
			{
				if (network->getGeneratedSpikes()[i].axon->postNeuron->layerID == layerID && network->getGeneratedSpikes()[i].axon->postNeuron->sublayerID == sublayerID && network->getGeneratedSpikes()[i].axon->postNeuron->rfRow == rfRow && network->getGeneratedSpikes()[i].axon->postNeuron->rfCol == rfCol)
				{
					network->getGeneratedSpikes().erase(network->getGeneratedSpikes().begin()+i);
				}
			}
			
			// resetting plastic neurons
			for (auto& inputAxon: preAxons)
			{
				inputAxon->preNeuron->eligibilityTrace = 0;
			}
		}
		
		// ----- NEURON PARAMETERS -----
		int16_t                                  neuronID;
		int16_t                                  rfRow;
		int16_t                                  rfCol;
		int16_t                                  sublayerID;
		int16_t                                  layerID;
		float                                    decayCurrent;
		float                                    decayPotential;
        float                                    refractoryPeriod;
        float                                    threshold;
        float                                    restingPotential;
        float                                    resetPotential;
        float                                    inputResistance;
        float                                    current;
        float                                    potential;
		bool                                     active;
		bool                                     inhibited;
		double                                   inhibitionTime;
		
		float                                    synapticEfficacy;
		float                                    externalCurrent;
		float                                    eligibilityTrace;
		float                                    eligibilityDecay;
		int16_t                                  xCoordinate;
		int16_t                                  yCoordinate;
		bool                                     burstingActivity;
		
		// ----- IMPLEMENTATION VARIABLES -----
		axon                                     activeAxon;
        std::vector<std::unique_ptr<axon>> postAxons;
        std::vector<axon*>                 preAxons;
        axon                                     initialAxon;
        double                                   lastSpikeTime;
        std::vector<LearningRuleHandler*>        learningRuleHandler;
		
        // ----- LEARNING RULE VARIABLES -----
        float                                    plasticityTrace;
    };
}
