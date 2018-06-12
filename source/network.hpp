/* 
 * network.hpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 01/06/2018
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
	struct receptiveField
	{
		std::vector<Neuron> rfNeurons;
		int16_t             rfID;
		int16_t             layerID;
	};
	
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
		{
		
		}
	
		// ----- PUBLIC NETWORK METHODS -----
		// add neurons
		void addNeurons(int16_t _layerID, learningMode _learningType=noLearning, int _numberOfNeurons=1, float _decayCurrent=10, float _decayPotential=20, int _refractoryPeriod=3, float _eligibilityDecay=100, float _alpha=1, float _lambda=1, float _threshold = -50, float  _restingPotential=-70, float _resetPotential=-70, float _inputResistance=50e9, float _externalCurrent=1, int16_t _rfID=0)
        {            
        	unsigned long shift = 0;
        	if (!neurons.empty())
        	{
				for (auto& it: neurons)
				{
					shift += it.rfNeurons.size();
				}
			}
			
        	std::vector<Neuron> temp;
			for (auto i=0+shift; i < _numberOfNeurons+shift; i++)
			{
				temp.emplace_back(i,_layerID,_rfID,_decayCurrent,_decayPotential,_refractoryPeriod, _eligibilityDecay,_alpha,_lambda,_threshold,_restingPotential,_resetPotential,_inputResistance, _externalCurrent,-1,-1,-1,_learningType);
			}
			neurons.push_back(receptiveField{std::move(temp),_rfID,_layerID});
        }
		
		// add neurons within receptive fields // make gridSize optional
		void addReceptiveFields(int rfNumber, int16_t _layerID, learningMode _learningType=noLearning, int gridSize=0, int _numberOfNeurons=-1, float _decayCurrent=10, float _decayPotential=20, int _refractoryPeriod=3, float _eligibilityDecay=100, float _alpha=1, float _lambda=1, float _threshold = -50, float  _restingPotential=-70, float _resetPotential=-70, float _inputResistance=50e9, float _externalCurrent=1)
		{
		    // error handling
		    double d_sqrt = std::sqrt(rfNumber);
            int i_sqrt = d_sqrt;
            if (d_sqrt != i_sqrt)
            {
                throw std::logic_error("the number of receptive fields has to be a perfect square"); 
            }
            
            double d_rfSize = gridSize/std::sqrt(rfNumber);
            int i_rfSize = d_rfSize;
            if (d_rfSize != i_rfSize)
            {
                throw std::logic_error("the size of the square grid does not match with the number of receptive fields");
            }
            
            // receptive field creation
		    if (_numberOfNeurons == -1)
		    { 
		        std::cout << "adding receptive fields with 2D neurons to the network" << std::endl;
		        int16_t x = 0;
		        int16_t y = 0;
		        
		        int ycount = 0;
			    int xcount = 0;
			     
			    int sq_rfNumber = std::sqrt(rfNumber);
		        for (int16_t j=0; j<rfNumber; j++)
		        {   
		            if (j % sq_rfNumber == 0 && j != 0)
		            {
                        ycount = 0;
                        xcount++;
		            }

                	unsigned long shift = 0;
                	if (!neurons.empty())
                	{
				        for (auto& it: neurons)
				        {
					        shift += it.rfNeurons.size();
				        }
			        }
			
                	std::vector<Neuron> temp;
                	int count = 0;
                	x=0+xcount*gridSize/sq_rfNumber;
                	
			        for (auto i=0+shift; i < std::pow(gridSize/sq_rfNumber,2)+shift; i++)
			        { 
			            if (count % (gridSize/sq_rfNumber) == 0 && count != 0)
			            {
			                y = 0+ycount*gridSize/sq_rfNumber;
			            }
			            count++;
				        temp.emplace_back(i,_layerID,j,_decayCurrent,_decayPotential,_refractoryPeriod, _eligibilityDecay,_alpha,_lambda,_threshold,_restingPotential,_resetPotential,_inputResistance, _externalCurrent,x,y,-1,_learningType);
				        y++;
				        if (count % (gridSize/sq_rfNumber) == 0 && count != 0)
			            {
			                x++;
			            }
			        }
			        ycount++;
			        neurons.push_back(receptiveField{std::move(temp),j,_layerID});
		        }
            }
            else if (_numberOfNeurons > 0)
            {
                std::cout << "adding receptive fields with 1D neurons to the network" << std::endl;
                for (auto j=0; j<rfNumber; j++)
		        {   
                    addNeurons(_layerID, _learningType,_numberOfNeurons, _decayCurrent,_decayPotential,_refractoryPeriod,_eligibilityDecay,_alpha, _lambda,_threshold,_restingPotential,_resetPotential,_inputResistance,_externalCurrent,j);
                }
            }
            else
            {
                throw std::logic_error("the number of neurons to add cannot be less than or equal to 0");
            }
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
						weight = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX/_weight));
					}
					else
					{
						weight = _weight;
					}
					pre.addProjection(&post, weight, delay, redundantConnections);
				}
			}
		}
		
		// add spike to the network
		void injectSpike(spike s)
        {
            initialSpikes.push_back(s);
        }
		
		// adding spikes generated by non-input neurons ot the network
        void injectGeneratedSpike(spike s)
        {
            generatedSpikes.insert(
                std::upper_bound(generatedSpikes.begin(), generatedSpikes.end(), s, [](spike one, spike two){return one.timestamp < two.timestamp;}),
                s);
        }
		
		// clock-based running through the network
        void run(double _runtime, float _timestep)
        {
        	layerNumber = getNeuronPopulations().size(); // everything to do with layers is broken
        	std::cout << "Running the network..." << std::endl;
			//#ifndef NDEBUG
            std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
            //#endif
			if (!neurons.empty())
			{
				for (double i=0; i<_runtime; i+=_timestep)
				{
					for (auto& pop: neurons)
					{
						for (auto& neuron: pop.rfNeurons)
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
            //#ifndef NDEBUG
            std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
            std::cout << "it took " << elapsed_seconds.count() << "s to run." << std::endl;
            //#endif
		}
		
		// ----- SETTERS AND GETTERS -----
		std::vector<receptiveField>& getNeuronPopulations()
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
		// add teacher signal for supervised learning
        void injectTeacher(std::vector<input>* _teacher)
        {
            teacher = _teacher;
            teachingProgress = true;
        }
		
		// getter for the teacher signal
        std::vector<input>* getTeacher() const
        {
            return teacher;
        }
		
    protected:
    
		// -----PROTECTED NETWORK METHODS -----
		// update neuron status
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
		
		// helper for the update method
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
		std::vector<receptiveField>      neurons;
		uint64_t					     layerNumber;
		int                              teacherIterator;
		bool                             teachingProgress;
		
		// ----- SUPERVISED LEARNING VARIABLES -----
        std::vector<input>* teacher;
    };
}
