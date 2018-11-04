/* 
 * network.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 29/10/2018
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
		std::vector<std::size_t> neurons;
		int16_t                  row;
		int16_t                  col;
	};
	
	struct sublayer
	{
		std::vector<receptiveField> receptiveFields;
		int16_t                     ID;
	};
	
	struct layer
	{
		std::vector<sublayer> sublayers;
		int16_t               ID;
		int                   width;
		int                   height;
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
		void addLayer(int16_t layerID, LearningRuleHandler* _learningRuleHandler=nullptr, int neuronNumber=1, int rfNumber=1, int _sublayerNumber=1, float _decayCurrent=10, float _decayPotential=20, int _refractoryPeriod=3, bool _burstingActivity=false, float _eligibilityDecay=100, float _threshold = -50, float  _restingPotential=-70, float _resetPotential=-70, float _inputResistance=50e9, float _externalCurrent=100, int16_t _rfID=0)
        {
        	unsigned long shift = 0;
        	if (!layers.empty())
        	{
        		for (auto& l: layers)
        		{
        			for (auto& s: l.sublayers)
        			{
						for (auto& r: s.receptiveFields)
						{
							shift += r.neurons.size();
						}
					}
				}
			}

			// building a layer of one dimensional sublayers with no receptiveFields
			std::vector<sublayer> subTemp;
        	for (int16_t i=0; i<_sublayerNumber; i++)
        	{
        		std::vector<receptiveField> rfTemp;
        		for (int16_t j=0; j<rfNumber; j++)
        		{
					std::vector<std::size_t> neuronTemp;
					for (auto k=0+shift; k<neuronNumber+shift; k++)
					{
						neurons.emplace_back(k, j, 0, i, layerID, _decayCurrent, _decayPotential, _refractoryPeriod, _burstingActivity, _eligibilityDecay, _threshold, _restingPotential, _resetPotential, _inputResistance, _externalCurrent,-1,-1,_learningRuleHandler);
						
						
						neuronTemp.emplace_back(neurons.size()-1);
					}
					rfTemp.emplace_back(receptiveField{neuronTemp, j, 0});
				}
				subTemp.emplace_back(sublayer{rfTemp, i});
			}
			layers.emplace_back(layer{subTemp, layerID, -1, -1});
        }
		
		void add2dLayer(int16_t layerID, int windowSize, int gridW, int gridH, LearningRuleHandler* _learningRuleHandler=nullptr, int _sublayerNumber=1, int _numberOfNeurons=-1, bool overlap=false, float _decayCurrent=10, float _decayPotential=20, int _refractoryPeriod=3, bool _burstingActivity=false, float _eligibilityDecay=100, float _threshold = -50, float  _restingPotential=-70, float _resetPotential=-70, float _inputResistance=50e9, float _externalCurrent=100)
		{
			// error handling
			if (windowSize <= 0 || windowSize > gridW || windowSize > gridH)
			{
				throw std::logic_error("the selected window size is not valid");
			}

			if (_numberOfNeurons != -1 && _numberOfNeurons <= 0)
			{
				throw std::logic_error("the number of neurons selected is wrong");
			}
			
			int overlapSize = 0;
			if (overlap)
			{
				if (windowSize > 1)
				{
					overlapSize = windowSize-1;
				}
				else if (windowSize == 1)
				{
					throw std::logic_error("For a window size equal to 1, consider using a layer with contiguous receptive fields by setting the overlap to false");
				}
			}
			else
			{
				double dW_check = gridW / static_cast<double>(windowSize);
				double dH_check = gridH / static_cast<double>(windowSize);
				
				int iW_check = dW_check;
				int iH_check = dH_check;

				if (dW_check != iW_check || dH_check != iH_check)
				{
					throw std::logic_error("With contiguous receptive fields, the width and height of the grid need to be divisible by the receptive field size");
				}
			}
			
			unsigned long shift = 0;
			if (!layers.empty())
			{
				for (auto& l: layers)
				{
					for (auto& s: l.sublayers)
					{
						for (auto& r: s.receptiveFields)
						{
							shift += r.neurons.size();
						}
					}
				}
			}
			
			std::vector<sublayer> subTemp;
			for (int16_t i=0; i<_sublayerNumber; i++)
			{
				int x = 0;
				int y = 0;
				
				int col = 0;
				int row = 0;
				
				int rowShift = 0;
				int colShift = 0;
				
				int16_t rfCol = 0;
				int16_t rfRow = 0;
				unsigned long neuronCounter = shift;
				
				std::vector<std::size_t> neuronTemp;
				std::vector<receptiveField> rfTemp;
				while (true)
				{
					if (x == gridW-1 && y != gridH-1 && col == 0 && row == 0)
					{
						rfCol = 0;
						rfRow += 1;
						colShift = 0;
						rowShift += windowSize-overlapSize;
					}
					
					x = col+colShift;
					y = row+rowShift;
					
					if (_numberOfNeurons == -1)
					{
						neurons.emplace_back(neuronCounter, rfRow, rfCol, i, layerID, _decayCurrent, _decayPotential, _refractoryPeriod, _burstingActivity, _eligibilityDecay, _threshold, _restingPotential, _resetPotential, _inputResistance, _externalCurrent, x, y, _learningRuleHandler);
					
						neuronCounter += 1;
					
						neuronTemp.emplace_back(neurons.size()-1);
					}
					
					col += 1;
					if (col == windowSize && row != windowSize-1)
					{
						col = 0;
						row += 1;
					}
					else if (col == windowSize && row == windowSize-1)
					{
						col = 0;
						row = 0;
						colShift += windowSize-overlapSize;
						if (_numberOfNeurons > 0)
						{
							for (auto j = 0; j < _numberOfNeurons; j++)
							{
								neurons.emplace_back(neuronCounter, rfRow, rfCol, i, layerID, _decayCurrent, _decayPotential, _refractoryPeriod, _burstingActivity, _eligibilityDecay, _threshold, _restingPotential, _resetPotential, _inputResistance, _externalCurrent, -1, -1, _learningRuleHandler);
								
								neuronCounter += 1;
								
								neuronTemp.emplace_back(neurons.size()-1);
							}
						}
						rfTemp.emplace_back(receptiveField{neuronTemp, rfRow, rfCol});
						rfCol += 1;
						neuronTemp.clear();
					}
					
					if (x == gridW-1 && y == gridH-1)
					{
						break;
					}
					
					if (x > gridW-1 || y > gridH-1)
					{
						throw std::logic_error("the window and the grid do not match. recheck the size parameters");
					}
				}
				subTemp.emplace_back(sublayer{rfTemp, i});
			}
			layers.emplace_back(layer{subTemp, layerID, gridW, gridH});
		}
		
		// all to all connections (for everything including sublayers and receptive fields)
    	void allToAll(layer presynapticLayer, layer postsynapticLayer, bool randomWeights, float _weight, bool randomDelays, int _delay=0, bool redundantConnections=true)
    	{
    		int delay = 0;
    		float weight = 0;
    		for (auto& preSub: presynapticLayer.sublayers)
    		{
    			for (auto& preRF: preSub.receptiveFields)
    			{
					for (auto& pre: preRF.neurons)
					{
						for (auto& postSub: postsynapticLayer.sublayers)
						{
							for (auto& postRF: postSub.receptiveFields)
    						{
    							for (auto& post: postRF.neurons)
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
									neurons[pre].addProjection(&neurons[post], weight, delay, redundantConnections);
								}
							}
						}
					}
				}
			}
		}
		
		// connecting two layers according to their receptive fields
		void convolution(layer presynapticLayer, layer postsynapticLayer, bool randomWeights, float _weight, bool randomDelays, int _delay=0, bool redundantConnections=true)
		{
			// restrict to layers of the same size
			if (presynapticLayer.width != postsynapticLayer.width || presynapticLayer.height != postsynapticLayer.height)
			{
				throw std::logic_error("Convoluting two layers requires them to be the same size");
			}
			
			int delay = 0;
    		float weight = 0;
			
			for (auto& preSub: presynapticLayer.sublayers)
    		{
    			for (auto& preRF: preSub.receptiveFields)
    			{
    				for (auto& postSub: postsynapticLayer.sublayers)
					{
						for (auto& postRF: postSub.receptiveFields)
						{
							if (preRF.row == postRF.row && preRF.col == postRF.col)
							{
								for (auto& pre: preRF.neurons)
								{
									for (auto& post: postRF.neurons)
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
										neurons[pre].addProjection(&neurons[post], weight, delay, redundantConnections);
									}
								}
							}
						}
					}
				}
			}
		}
		
		// subsampling connection of receptive fields
		void pooling(layer presynapticLayer, layer postsynapticLayer, bool randomWeights, float _weight, bool randomDelays, int _delay=0, bool redundantConnections=true)
		{
			auto preMaxRows = presynapticLayer.sublayers[0].receptiveFields.back().row+1;
			auto preMaxColumns = presynapticLayer.sublayers[0].receptiveFields.back().col+1;
			
			auto postMaxRows = postsynapticLayer.sublayers[0].receptiveFields.back().row+1;
			auto postMaxColumns = postsynapticLayer.sublayers[0].receptiveFields.back().col+1;
			
			float fRow_check = preMaxRows / static_cast<float>(postMaxRows);
			float fCol_check = preMaxColumns / static_cast<float>(postMaxColumns);
			
			int rowPoolingFactor = fRow_check;
			int colPoolingFactor = fCol_check;
			
			if (rowPoolingFactor != fRow_check || colPoolingFactor != fCol_check)
			{
				throw std::logic_error("the number of receptive fields in each layer is not proportional. The pooling factor cannot be calculated");
			}
			
			int delay = 0;
    		float weight = 0;
			
			for (auto& preSub: presynapticLayer.sublayers)
			{
				for (auto& postSub: postsynapticLayer.sublayers)
				{
					// each presynaptic sublayer connects to the same postsynaptic sublayer
					if (preSub.ID == postSub.ID)
					{
						int rowShift = 0;
						int colShift = 0;
						for (auto& postRf: postSub.receptiveFields)
						{
							for (auto& preRf: preSub.receptiveFields)
							{
								if ( preRf.row >= 0+rowShift && preRf.row < rowPoolingFactor+rowShift && preRf.col >= 0+colShift && preRf.col < colPoolingFactor+colShift)
								{
									for (auto& pre: preRf.neurons)
									{
										for (auto& post: postRf.neurons)
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
											neurons[pre].addProjection(&neurons[post], weight, delay, redundantConnections);
										}
									}
								}
							}
							colShift += colPoolingFactor;
							if (postRf.col == postMaxColumns-1)
							{
								colShift = 0;
								rowShift += rowPoolingFactor;
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
			// error handling
			if (neurons.empty())
			{
				throw std::logic_error("add neurons before injecting spikes");
			}
			
			if ((*data)[1].x == -1 && (*data)[1].y == -1)
			{
				for (auto& event: *data)
				{
					for (auto& l: layers[0].sublayers)
					{
						if (l.ID == event.sublayerID)
						{
							for (auto& r: l.receptiveFields)
							{
								injectSpike(neurons[r.neurons[event.neuronID]].prepareInitialSpike(event.timestamp));
							}
						}
						else if (l.ID == -1)
						{
							for (auto& r: l.receptiveFields)
							{
								injectSpike(neurons[r.neurons[event.neuronID]].prepareInitialSpike(event.timestamp));
							}
						}
						else
						{
							continue;
						}
					}
				}
    		}
    		else
    		{
				for (auto& event: *data)
				{
					for (auto& l: layers[0].sublayers)
					{
						if (l.ID == event.sublayerID)
						{
							for (auto& r: l.receptiveFields)
							{
								for (auto& n: r.neurons)
								{
									if (neurons[n].getX() == event.x && neurons[n].getY() == event.y)
									{
										injectSpike(neurons[n].prepareInitialSpike(event.timestamp));
										break;
									}
								}
							}
						}
						else if (l.ID == -1)
						{
							for (auto& r: l.receptiveFields)
							{
								for (auto& n: r.neurons)
								{
									if (neurons[n].getX() == event.x && neurons[n].getY() == event.y)
									{
										injectSpike(neurons[n].prepareInitialSpike(event.timestamp));
										break;
									}
								}
							}
						}
						else
						{
							continue;
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
						for (auto& n: neurons)
						{
							update(&n, i, _timestep);
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
				int numberOfLayers = static_cast<int>(layers.size());
			
				// number of neurons in each layer
				std::vector<int> neuronsInLayers;
				for (auto& l: layers)
				{
					int count = 0;
					for (auto& s: l.sublayers)
					{
						for (auto& r: s.receptiveFields)
						{
							count += r.neurons.size();
						}
					}
					neuronsInLayers.emplace_back(count);
				}
				
				thDelegate->begin(numberOfLayers, neuronsInLayers);
			}
			spikeManager.join();
		}
		
		// ----- SETTERS AND GETTERS -----
		std::vector<Neuron>& getNeurons()
		{
			return neurons;
		}
		
		std::vector<layer>& getLayers()
		{
			return layers;
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
        std::vector<layer>                     layers;
		std::vector<Neuron>                    neurons;
		std::deque<label>*                     labels;
		bool                                   teachingProgress;
		bool                                   learningStatus;
		double                                 learningOffSignal;
		
		// ----- SUPERVISED LEARNING VARIABLES -----
        std::deque<double>*                    teacher;
    };
}
