/* 
 * network.hpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 6/12/2017
 *
 * Information: the network class acts as a spike manager.
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
		
		enum class Mode
        {
        	display,
        	logger
		};
		
    	// ----- PURE VIRTUAL METHOD -----
        virtual void getArrivingSpike(double timestamp, projection* p, bool spiked, bool empty, Network* network, Neuron* postNeuron) = 0;
        virtual Mode getMode() = 0;
    };
    
    class Network
    {
    public:
		
		// ----- CONSTRUCTOR AND DESTRUCTOR ------
        Network(std::vector<NetworkDelegate*> _delegates = {}) :
            delegates(_delegates),
            teacher(nullptr),
			layerCounter(0),
			inputSpikeCounter(0),
			teachingProgress(false),
			teacherIterator(0)
		{}
		
		// ----- PUBLIC NETWORK METHODS -----
		void addNeurons(int _numberOfNeurons, float _decayCurrent=10, float _decayPotential=20, int _refractoryPeriod=3, float _decaySynapticEfficacy=0, float _synapticEfficacy=1, float _threshold = -50, float  _restingPotential=-70, float _resetPotential=-70, float _inputResistance=50e9, float _externalCurrent=1)
        {
        	unsigned long shift = 0;
        	if (!neurons.empty())
        	{
				shift = neurons.back().size();
			}
			
        	std::vector<Neuron> temp;
			for (auto i=0+shift; i < _numberOfNeurons+shift; i++)
			{
			temp.emplace_back(i,layerCounter,_decayCurrent,_decayPotential,_refractoryPeriod,_decaySynapticEfficacy,_synapticEfficacy,_threshold,_restingPotential,_resetPotential,_inputResistance, _externalCurrent);
			}
			neurons.push_back(std::move(temp));
			layerCounter++;
        }
		
    	void allToallConnectivity(std::vector<Neuron>* presynapticLayer, std::vector<Neuron>* postsynapticLayer, bool randomWeights, float _weight, bool randomDelays, int _delay=0)
    	{
    		int delay = 0;
    		float weight = 0;
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
					
					if (randomWeights)
					{
						weight = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/_weight));
					}
					else
					{
						weight = _weight;
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
		
        void run(double _runtime, float _timestep)
        {
        	std::cout << "Running the network..." << std::endl;
			#ifndef NDEBUG
            std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
            #endif
			if (!neurons.empty())
			{
				for (double i=0; i<_runtime; i+=_timestep)
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
			
			std::cout << "Done." << std::endl;
            #ifndef NDEBUG
            std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
            std::cout << "it took " << elapsed_seconds.count() << "s to run." << std::endl;
            #endif
		}
		
		void learningLogger(std::string filename)
		{
			learningLog.reset(new std::ofstream(filename));
		}
		
		// ----- SETTERS AND GETTERS -----
		std::unique_ptr<std::ofstream>& getLearningLog()
		{
			return learningLog;
		}
		
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
		
        std::vector<double>& getPlasticTime()
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
		
		int getTeacherIterator() const
        {
            return teacherIterator;
        }
		
		void setTeacherIterator(int increment)
        {
            teacherIterator = increment;
        }
		
		int getTeachingProgress() const
        {
            return teachingProgress;
        }
		
		void setTeachingProgress(bool status)
        {
            teachingProgress = status;
        }
		
		// ----- SUPERVISED LEARNING METHOD -----
        void injectTeacher(std::vector<std::vector<double>>* _teacher)
        {
            teacher = _teacher;
            teachingProgress = true;
        }
		
        std::vector<std::vector<double>>* getTeacher() const
        {
            return teacher;
        }
		
    protected:
    
		// -----PROTECTED NETWORK METHODS -----
		void update(Neuron* neuron, double time, float timestep)
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
		
		void updateHelper(spike s, Neuron* neuron, double time, float timestep, int listSelector)
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
		std::vector<double>              plasticTime;
        std::vector<Neuron*>             plasticNeurons;
		std::deque<spike>                initialSpikes;
        std::deque<spike>                generatedSpikes;
        std::vector<NetworkDelegate*>    delegates;
		std::vector<std::vector<Neuron>> neurons;
		int                              layerCounter;
		int                              inputSpikeCounter;
		int                              teacherIterator;
		bool                             teachingProgress;
		std::unique_ptr<std::ofstream>   learningLog;
		
		// ----- SUPERVISED LEARNING VARIABLES -----
        std::vector<std::vector<double>>* teacher;
    };
}
