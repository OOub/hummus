/*
 * network.hpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/11/2017
 *
 * Information:
 */

#pragma once

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <chrono>
#include <deque>

#include "neuron.hpp"

namespace baal
{
	class Network;
	// polymorphic class for add-ons
    class NetworkDelegate
    {
    public:
    	// ----- CONSTRUCTOR AND DESTRUCTOR -----
    	NetworkDelegate() = default;
    	virtual ~NetworkDelegate(){}
		
    	// ----- PURE VIRTUAL METHOD -----
        virtual void getArrivingSpike(float timestamp, projection* p, bool spiked, bool empty, Network* network, Neuron* postNeuron) = 0;
    };
    
    class Network
    {
    public:
		
		// ----- CONSTRUCTOR AND DESTRUCTOR ------
        Network(std::vector<NetworkDelegate*> _delegates = {}) :
            delegates(_delegates),
			layerCounter(0),
			inputSpikeCounter(0)
		{}
		
		// ----- PUBLIC NETWORK METHODS -----
		void addNeurons(int _numberOfNeurons, float _decayCurrent=10, float _decayPotential=20, int _refractoryPeriod=3, float _decaySynapticEfficacy=0, float _synapticEfficacy=1, float _threshold = -50, float  _restingPotential=-70, float _resetPotential=-70, float _inputResistance=50e9, float _externalCurrent=17e-10, float _currentBurnout=3.1e-9)
        {
        	unsigned long shift = 0;
        	if (!neurons.empty())
        	{
				shift = neurons.back().size();
			}
			
        	std::vector<Neuron> temp;
			for (auto i=0+shift; i < _numberOfNeurons+shift; i++)
			{
				temp.emplace_back(i,layerCounter,_decayCurrent,_decayPotential,_refractoryPeriod,_decaySynapticEfficacy,_synapticEfficacy,_threshold,_restingPotential,_resetPotential,_inputResistance, _externalCurrent, _currentBurnout);
			}
			neurons.push_back(std::move(temp));
			layerCounter++;
        }
		
    	void allToallConnectivity(std::vector<Neuron>* presynapticLayer, std::vector<Neuron>* postsynapticLayer, float weight, bool randomDelays, int _delay=0)
    	{
    		int delay = 0;
    		for (auto& pre: *presynapticLayer)
    		{
    			for (auto& post: *postsynapticLayer)
    			{
    				if (randomDelays)
    				{
    					delay = std::rand() % _delay;
					}
					else
					{
    					delay = _delay;
					}
					pre.addProjection(&post, weight, delay);
				}
			}
		}
		
		void injectSpike(spike s)
        {
            initialSpikes.push_back(s);
        }
		
        void injectGeneratedSpike(spike s)
        {
            generatedSpikes.insert(
                std::upper_bound(generatedSpikes.begin(), generatedSpikes.end(), s, [](spike one, spike two){return one.timestamp < two.timestamp;}),
                s);
        }
		
        void run(float _runtime, float _timestep)
        {
        	std::cout << "Running the network...\n";
			#ifndef NDEBUG
            std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
            #endif
			if (!neurons.empty())
			{
				for (float i=0; i<_runtime; i+=_timestep)
				{
					for (auto& pop: neurons)
					{
						for (auto& neuron: pop)
						{
							update(&neuron, i, _timestep);
						}
					}
				}
			}
			else
			{
				throw std::runtime_error("add neurons to the network before running it");
			}
			
			std::cout << "Done.\n";
            #ifndef NDEBUG
            std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
            std::cout << "it took " << elapsed_seconds.count() << "s to run.\n";
            #endif
		}
		
		// ----- SETTERS AND GETTERS -----
		std::vector<std::vector<Neuron>>& getNeuronPopulations()
		{
			return neurons;
		}
		
		std::vector<NetworkDelegate*>& getDelegates()
		{
			return delegates;
		}
		
		std::vector<Neuron*>& getPlasticNeurons()
        {
            return plasticNeurons;
        }
		
        std::vector<float>& getPlasticTime()
        {
            return plasticTime;
        }
		
		std::deque<spike>& getGeneratedSpikes()
        {
            return generatedSpikes;
        }
		
		int getInputSpikeCounter() const
        {
            return inputSpikeCounter;
        }
		
		void setInputSpikeCounter(int resetValue)
        {
            inputSpikeCounter = resetValue;
        }
		
    protected:

		void update(Neuron* neuron, float time, float timestep)
		{
			if (generatedSpikes.empty() && !initialSpikes.empty())
			{
				spike s = initialSpikes.front();
				updateHelper(s, neuron, time, timestep,1);
			}
			else if (initialSpikes.empty() && !generatedSpikes.empty())
			{
				spike s = generatedSpikes.front();
				updateHelper(s, neuron, time, timestep,0);
			}
			else if (!initialSpikes.empty() && !generatedSpikes.empty())
			{
				if (initialSpikes.front().timestamp < generatedSpikes.front().timestamp)
				{
					spike s = initialSpikes.front();
					updateHelper(s, neuron, time, timestep,1);
				}
				else
				{
					spike s = generatedSpikes.front();
					updateHelper(s, neuron, time, timestep,0);
				}
			}
			else
			{
				neuron->update(time, timestep, spike({time, nullptr}), this); //if both are empty
			}
		}
		
		void updateHelper(spike s, Neuron* neuron, float time, float timestep, int listSelector)
		{
			if (s.postProjection->postNeuron->getNeuronID() == neuron->getNeuronID())
			{
				
				if (s.timestamp <= time + (timestep/2))
				{
					if (s.postProjection->isInitial)
					{
						inputSpikeCounter++;
					}
					
					neuron->update(time, timestep, s, this);

					if (listSelector == 0)
					{
						if (!generatedSpikes.empty())
						{
							generatedSpikes.pop_front();
						}
					}
					
					else if (listSelector == 1)
					{
						initialSpikes.pop_front();
					}
				}
				else
				{
					neuron->update(time, timestep, spike({time, nullptr}),this);
				}
			}
			else
			{
				neuron->update(time, timestep, spike({time, nullptr}),this);
			}
		}
		
		// ----- IMPLEMENTATION VARIABLES -----
		std::vector<float>               plasticTime;
        std::vector<Neuron*>             plasticNeurons;
		std::deque<spike>                initialSpikes;
        std::deque<spike>                generatedSpikes;
        std::vector<NetworkDelegate*>    delegates;
		std::vector<std::vector<Neuron>> neurons;
		int                              layerCounter;
		int                              inputSpikeCounter;
    };
}
