/* 
 * network.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 01/06/2018
 *
 * Information: The Network class acts as a spike manager.
 */

#pragma once

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <string>
#include <deque>

#include "neuron.hpp"
#include "dataParser.hpp"
#include "learningRuleHandler.hpp"
#include "standardNetworkDelegate.hpp"
#include "mainThreadNetworkDelegate.hpp"

namespace adonis_c
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
        Network(std::vector<StandardNetworkDelegate*> _stdDelegates = {}, MainThreadNetworkDelegate* _thDelegate = nullptr) :
            stdDelegates(_stdDelegates),
			thDelegate(_thDelegate),
            teacher(nullptr),
            labels(nullptr),
			teachingProgress(false),
			learningStatus(true),
			learningOffSignal(-1)
		{}
		
		Network(MainThreadNetworkDelegate* _thDelegate) : Network({}, _thDelegate)
		{}
		
		// ----- PUBLIC NETWORK METHODS -----
		// add neurons
		void addNeurons(int16_t _layerID, LearningRuleHandler* _learningRuleHandler=nullptr, int _numberOfNeurons=1, float _decayCurrent=10, float _decayPotential=20, int _refractoryPeriod=3, bool _burstingActivity=false, float _eligibilityDecay=100, float _threshold = -50, float  _restingPotential=-70, float _resetPotential=-70, float _inputResistance=50e9, float _externalCurrent=100, int16_t _rfID=0)
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
				temp.emplace_back(i,_layerID,_rfID,_decayCurrent,_decayPotential,_refractoryPeriod,_burstingActivity, _eligibilityDecay,_threshold,_restingPotential,_resetPotential,_inputResistance, _externalCurrent,-1,-1,-1,_learningRuleHandler);
			}
			neurons.push_back(receptiveField{std::move(temp),_rfID,_layerID});
        }
		
		// add neurons within square non-overlapping receptive fields
		void addReceptiveFields(int rfSize, int gridW, int gridH, int16_t _layerID, LearningRuleHandler* _learningRuleHandler=nullptr, int _numberOfNeurons=-1, float _decayCurrent=10, float _decayPotential=20, int _refractoryPeriod=3, bool _burstingActivity=false, float _eligibilityDecay=100, float _threshold = -50, float  _restingPotential=-70, float _resetPotential=-70, float _inputResistance=50e9, float _externalCurrent=100)
		{
			// error handling
			double dW_check = gridW / rfSize;
			double dH_check = gridW / rfSize;
			
			int iW_check = dW_check;
			int iH_check = dH_check;
			
			if (dW_check != iW_check || dH_check != iH_check)
			{
			 throw std::logic_error("The width and height of the grid need to be divisible by the receptive field size");
			}
			
		    int rfNumber = (gridW/rfSize) * (gridH/rfSize);

            // receptive field creation
		    if (_numberOfNeurons == -1)
		    {
		        std::cout << "adding receptive fields with 2D neurons to the network" << std::endl;
		        int16_t x = 0;
		        int16_t y = 0;

		        int ycount = 0;
			    int xcount = 0;
				int count = 0;

		        for (int16_t j=0; j<rfNumber; j++)
		        {
		            if (j % (gridW/rfSize) == 0 && j != 0)
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

                	x=0+xcount*rfSize;

			        for (auto i=0+shift; i < std::pow(rfSize,2)+shift; i++)
			        {
			            if (count % rfSize == 0 && count != 0)
			            {
							y = 0+ycount*rfSize;
			            }
			            count++;
				        temp.emplace_back(i,_layerID,j,_decayCurrent,_decayPotential,_refractoryPeriod,_burstingActivity, _eligibilityDecay,_threshold,_restingPotential,_resetPotential,_inputResistance, _externalCurrent,x,y,-1,_learningRuleHandler);

						y++;

				        if (count % rfSize == 0 && count != 0)
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
                    addNeurons(_layerID, _learningRuleHandler,_numberOfNeurons, _decayCurrent,_decayPotential,_refractoryPeriod,_burstingActivity,_eligibilityDecay,_threshold,_restingPotential,_resetPotential,_inputResistance,_externalCurrent,j);
                }
            }
            else
            {
                throw std::logic_error("the number of neurons to add cannot be less than or equal to 0");
            }
		}
		
		// standard all to all connectivity
    	void allToAllConnectivity(std::vector<Neuron>* presynapticLayer, std::vector<Neuron>* postsynapticLayer, bool randomWeights, float _weight, bool randomDelays, int _delay=0, bool redundantConnections=true)
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
		
		// connecting two layers according to their receptive fields
		void rfConnectivity(int _preLayer, int _postLayer, bool randomWeights, float _weight, bool randomDelays, int _delay=0, bool flatten=false, bool redundantConnections=true)
		{
			if (!flatten)
			{
				for (auto& receptiveFieldI: neurons)
				{
					if (receptiveFieldI.layerID == _preLayer)
					{
						for (auto& receptiveFieldO: neurons)
						{
							if (receptiveFieldO.rfID == receptiveFieldI.rfID && receptiveFieldO.layerID == _postLayer)
							{
								allToAllConnectivity(&receptiveFieldI.rfNeurons, &receptiveFieldO.rfNeurons, randomWeights, _weight, randomDelays, _delay, redundantConnections);
							}
						}
					}
				}
			}
			else if (flatten)
			{
				for (auto& receptiveFieldI: neurons)
				{
					if (receptiveFieldI.layerID == _preLayer)
					{
						for (auto& receptiveFieldO: neurons)
						{
							if (receptiveFieldO.layerID == _postLayer)
							{
								allToAllConnectivity(&receptiveFieldI.rfNeurons, &receptiveFieldO.rfNeurons, randomWeights, _weight, randomDelays, _delay, redundantConnections);
							}
						}
					}
				}
			}
		}
		
		// add labels that can be displayed on the qtDisplay if it is being used
		void addLabels(std::deque<label>* _labels)
		{
			labels = _labels;
		}
		
		// add spike to the network
		void injectSpike(spike s)
        {
            initialSpikes.push_back(s);
        }
		
		void injectSpikeFromData(std::vector<input>* data)
		{
			if ((*data)[1].x == -1 && (*data)[1].y == -1)
			{
				for (auto idx=0; idx<data->size(); idx++)
				{
					injectSpike(neurons[0].rfNeurons[(*data)[idx].neuronID].prepareInitialSpike((*data)[idx].timestamp));
				}
    		}
    		else
    		{
				int initialLayer = neurons[std::distance(std::begin(neurons),std::min_element(neurons.begin(), neurons.end(), [](const receptiveField& one, const receptiveField& two){return one.layerID < two.layerID;}))].layerID;
				
				for (auto& event: *data)
				{
					for (auto& receptiveField: neurons)
					{
						if (receptiveField.layerID == initialLayer)
						{
							for (auto& neuron: receptiveField.rfNeurons)
							{
								if (neuron.getX() == event.x && neuron.getY() == event.y)
								{
									injectSpike(neuron.prepareInitialSpike(event.timestamp));
									break;
								}
							}
						}
					}
				}
			}
		}
		
		// adding spikes generated by non-input neurons ot the network
        void injectGeneratedSpike(spike s)
        {
            generatedSpikes.insert(
                std::upper_bound(generatedSpikes.begin(), generatedSpikes.end(), s, [](spike one, spike two){return one.timestamp < two.timestamp;}),
                s);
        }
		
		// turn off all learning rules (for cross-validation or test data)
		void turnOffLearning(double timestamp)
		{
			learningOffSignal = timestamp;
		}
		
		// clock-based running through the network
        void run(double _runtime, float _timestep)
        {
			std::thread spikeManager([this, _runtime, _timestep]
			{
				std::cout << "Running the network..." << std::endl;
				std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

				if (!neurons.empty())
				{
					for (double i=0; i<_runtime; i+=_timestep)
					{
					
						if (labels)
						{
							if (labels->size() != 0)
							{
								if (labels->front().onset <= i)
								{
									std::cout << labels->front().name << " at t=" << i << std::endl;
									labels->pop_front();
								}
							}
						}
						
						if (learningOffSignal != -1)
						{
							if (learningStatus==true && i >= learningOffSignal)
							{
								std::cout << "learning turned off at t=" << i << std::endl;
								learningStatus = false;

								if (teachingProgress==true)
								{
									std::cout << "teacher signal stopped at t=" << i << std::endl;
									teachingProgress = false;
									teacher->clear();
								}
							}
						}
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

				std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
				std::cout << "it took " << elapsed_seconds.count() << "s to run." << std::endl;
			});
			
			if (thDelegate)
			{
				// finding the number of layers in the network
				int numberOfLayers = neurons[std::distance(std::begin(neurons),std::max_element(neurons.begin(), neurons.end(), [](const receptiveField& one, const receptiveField& two){return one.layerID < two.layerID;}))].layerID;
			
				// number of neurons in each layer
				std::vector<int> neuronsInLayers(numberOfLayers+1, 0);
				for (auto i=0; i< numberOfLayers+1; i++)
				{
					for (auto& receptiveField: neurons)
					{
						if (receptiveField.layerID == i)
						{
							neuronsInLayers[i] += receptiveField.rfNeurons.size();
						}
					}
					if (i > 0)
					{
						neuronsInLayers[i] += neuronsInLayers[i-1];
					}
				}
				
				thDelegate->begin(numberOfLayers, neuronsInLayers);
			}
			spikeManager.join();
		}
		
		// ----- SETTERS AND GETTERS -----
		std::vector<receptiveField>& getNeuronPopulations()
		{
			return neurons;
		}
		
		std::vector<StandardNetworkDelegate*>& getStandardDelegates()
		{
			return stdDelegates;
		}
		
		MainThreadNetworkDelegate* getMainThreadDelegate()
		{
			return thDelegate;
		}
		
		std::deque<spike>& getGeneratedSpikes()
        {
            return generatedSpikes;
        }
		
		bool getTeachingProgress() const
        {
            return teachingProgress;
        }
		
		bool getLearningStatus() const
        {
            return learningStatus;
        }
		
		void setTeachingProgress(bool status)
        {
            teachingProgress = status;
        }
		
		// ----- SUPERVISED LEARNING METHODS -----
		// add teacher signal for supervised learning
        void injectTeacher(std::deque<double>* _teacher)
        {
            teacher = _teacher;
            teachingProgress = true;
        }
		
		// getter for the teacher signal
        std::deque<double>* getTeacher() const
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
				neuron->update(time, timestep, spike({time, nullptr}), this);
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
		std::deque<spike>                      initialSpikes;
        std::deque<spike>                      generatedSpikes;
        std::vector<StandardNetworkDelegate*>  stdDelegates;
        MainThreadNetworkDelegate*             thDelegate;
		std::vector<receptiveField>            neurons;
		std::deque<label>*                     labels;
		bool                                   teachingProgress;
		bool                                   learningStatus;
		double                                 learningOffSignal;
		
		// ----- SUPERVISED LEARNING VARIABLES -----
        std::deque<double>*                    teacher;
    };
}
