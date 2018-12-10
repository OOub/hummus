/*
 * analysis.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 29/11/2018
 *
 * Information: The Analysis class checks the classification accuracy of the spiking neural network
 */

#pragma once

#include <vector>

#include "neuron.hpp"
#include "network.hpp"
#include "dataParser.hpp"

namespace adonis_c
{
	class Analysis : public StandardNetworkAddOn
	{
	public:
		// ----- CONSTRUCTOR -----
		Analysis(std::string testLabels)
		{
			DataParser parser;
			labels = parser.readLabels(testLabels);
			for (auto label: labels)
			{
				actualLabels.emplace_back(label.name);
			}
		}
		
		// ----- PUBLIC METHODS -----
		void accuracy()
		{
			if (!predictedLabels.empty() && predictedLabels.size() == actualLabels.size())
			{
				std::vector<std::string> correctLabels;
				for (auto i=0; i<actualLabels.size(); i++)
				{
					if (predictedLabels[i] == actualLabels[i])
					{
						correctLabels.emplace_back(predictedLabels[i]);
					}
				}
				
				double accuracy =  (static_cast<double>(correctLabels.size())/actualLabels.size())*100;
				std::cout << "the classification accuracy is: " << accuracy << "%" << std::endl;
			}
			else
			{
				throw std::logic_error("there is a problem with the predicted and actual labels");
			}
		}
		
		void neuronFired(double timestamp, axon* p, Network* network) override
		{
			// logging only after learning is stopped
			if (!network->getLearningStatus())
			{
				// restrict only to the output layer
				if (p->postNeuron->getLayerID() == network->getLayers().back().ID)
				{
					predictedSpikes.emplace_back(spike{timestamp, p});
				}
			}
		}
		
		void simulationComplete(Network* network) override
		{
			for (auto i=1; i<labels.size(); i++)
			{
				auto it = predictedSpikes.end();
				if (i == labels.size()-1)
				{
					it = std::find_if(predictedSpikes.begin(), predictedSpikes.end(), [&](spike a){return a.timestamp >= labels[i-1].onset;});
				}
				else
				{
					it = std::find_if(predictedSpikes.begin(), predictedSpikes.end(), [&](spike a){return a.timestamp >= labels[i-1].onset && a.timestamp < labels[i].onset;});
				}

				if (it != predictedSpikes.end())
				{
					auto idx = std::distance(predictedSpikes.begin(), it);
					for (auto n: network->getSupervisedNeurons())
					{
						if (n.neuron == predictedSpikes[idx].axon->postNeuron->getNeuronID())
						{
							predictedLabels.emplace_back(n.label);
						}
					}
				}
				else
				{
					predictedLabels.emplace_back("NaN");
				}
			}
		}
		
	protected:
		// ----- IMPLEMENTATION VARIABLES -----
		std::vector<spike>       predictedSpikes;
		std::deque<label>        labels;
		std::deque<std::string>  actualLabels;
		std::deque<std::string>  predictedLabels;
	};
}
