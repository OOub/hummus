/* 
 * network.hpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 26/02/2018
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
#include "dataParser.hpp"
#include "networkDelegate.hpp"

namespace baal
{
    class Network
    {
    public:
		
		// ----- CONSTRUCTOR AND DESTRUCTOR ------
        Network(std::vector<NetworkDelegate*> _delegates = {}) :
            delegates(_delegates),
            teacher(nullptr),
            layerNumber(0),
			teachingProgress(false),
			teacherIterator(0)
		{}
	
		// ----- PUBLIC NETWORK METHODS -----
		void addNeurons(int _numberOfNeurons, int _layerID, int _xCoordinate=0, int _yCoordinate=0, int _zCoordinate=0, learningMode _learningType=noLearning, float _decayCurrent=10, float _decayPotential=20, int _refractoryPeriod=3, float _eligibilityDecay=100, float _alpha=1, float _lambda=1, float _threshold = -50, float  _restingPotential=-70, float _resetPotential=-70, float _inputResistance=50e9, float _externalCurrent=1)
        {            
        	unsigned long shift = 0;
        	if (!neurons.empty())
        	{
				for (auto& it: neurons)
				{
					shift += it.size();
				}
			}
			
        	std::vector<Neuron> temp;
			for (auto i=0+shift; i < _numberOfNeurons+shift; i++)
			{
				temp.emplace_back(i,_layerID,_decayCurrent,_decayPotential,_refractoryPeriod, _eligibilityDecay,_alpha,_lambda,_threshold,_restingPotential,_resetPotential,_inputResistance, _externalCurrent,_xCoordinate,_yCoordinate,_zCoordinate,_learningType);
			}
			neurons.push_back(std::move(temp));
        }
		
		// standard all to all connectivity
    	void allToallConnectivity(std::vector<Neuron>* presynapticLayer, std::vector<Neuron>* postsynapticLayer, bool randomWeights, float _weight, bool randomDelays, int _delay=0, bool redundantConnections=true)
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
					pre.addProjection(&post, weight, delay, redundantConnections);
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
		
        void run(double _runtime, float _timestep, bool sudokuWeightsSave=false)
        {
        	layerNumber = getNeuronPopulations().size();
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
			
			if (sudokuWeightsSave)
			{
				// saving weights of the initial layer
				std::ofstream myfile;
				myfile.open ("sudoku_output.txt");
				for (auto& n: neurons)
				{
					for (auto& m: n)
					{
						if (m.getLayerID() == 0)
						{
							for (auto& proj: m.getPostProjections())
							{
								myfile << proj->preNeuron->getNeuronID() << " " << proj->postNeuron->getNeuronID() << " " << proj->weight << " " << proj->preNeuron->getLayerID() << " " << proj->postNeuron->getLayerID() << " " << proj->preNeuron->getX() << " " << proj->preNeuron->getY() << "\n";
							}
						}
					}
				}
				myfile.close();
  			}

			std::cout << "Done." << std::endl;
            #ifndef NDEBUG
            std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
            std::cout << "it took " << elapsed_seconds.count() << "s to run." << std::endl;
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
		
		std::deque<spike>& getGeneratedSpikes()
        {
            return generatedSpikes;
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
		
        uint64_t getLayerNumber() const
        {
        	return layerNumber;
        }
		
		// ----- SUPERVISED LEARNING METHODS -----
        void injectTeacher(std::vector<input>* _teacher)
        {
            teacher = _teacher;
            teachingProgress = true;
        }
		
        std::vector<input>* getTeacher() const
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
		std::deque<spike>                initialSpikes;
        std::deque<spike>                generatedSpikes;
        std::vector<NetworkDelegate*>    delegates;
		std::vector<std::vector<Neuron>> neurons;
		uint64_t					     layerNumber;
		int                              teacherIterator;
		bool                             teachingProgress;
		
		// ----- SUPERVISED LEARNING VARIABLES -----
        std::vector<input>* teacher;
    };
}
