/*
 * logger.hpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 21/11/2017
 *
 * Information: Add-on to the Network class, used to write the spiking neural network output into a log binary file.
 */

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>

#include "network.hpp"

namespace baal
{
    class Logger : public NetworkDelegate
    {
    public:
    	// ----- CONSTRUCTOR -----
        Logger(std::string filename) :
            potential(0)
        {
            saveFile.open(filename, std::ios::out | std::ios::binary);
            if (!saveFile.good())
            {
                throw std::runtime_error("the file could not be opened");
            }
        }
		
		// ----- PUBLIC LOGGER METHODS -----
        void getArrivingSpike(double timestamp, projection* p, bool spiked, bool empty, Network* network, Neuron* postNeuron) override
        {
        	if (!empty)
        	{
				potential = p->postNeuron->getPotential();
				
//				float threshold = p->postNeuron->getThreshold();
				int16_t preN = (p->preNeuron ? p->preNeuron->getNeuronID() : -1);
				int16_t postN = p->postNeuron->getNeuronID();
				
				for (auto i=0; i<=4; i++)
				{
//					packet[i] = *(reinterpret_cast<char*>(&timestamp) + i);
//					packet[i+4] = *(reinterpret_cast<char*>(&p->delay) + i);
//					packet[i+8] = *(reinterpret_cast<char*>(&potential) + i);
//					packet[i+12] = *(reinterpret_cast<char*>(&threshold) + i);
					packet[i] = *(reinterpret_cast<char*>(&timestamp) + i);
					packet[i+8] = *(reinterpret_cast<char*>(&p->delay) + i);
					packet[i+12] = *(reinterpret_cast<char*>(&potential) + i);
				}
				
				for (auto i=16, j=0; i<=17; i++,j++)
				{
					packet[i] = *(reinterpret_cast<char*>(&preN) + j);
					packet[i+2] = *(reinterpret_cast<char*>(&postN) + j);
				}

				saveFile.write(packet, sizeof(packet));
            }
        }
    
    protected:
		
    	// ----- IMPLEMENTATION VARIABLES -----
        std::ofstream saveFile;
        char packet[20];
        float potential;
    };
}
