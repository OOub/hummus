/*
 * logger.hpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/06/2018
 *
 * Information: Add-on to the Network class, used to write the spiking neural network output into a log binary file.
 */

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>


#include "network.hpp"

namespace baal
{
    class Logger : public NetworkDelegate
    {
    public:
    	// ----- CONSTRUCTOR -----
        Logger(std::string filename) :
            bytes(32, 0)
        {
            saveFile.open(filename, std::ios::out | std::ios::binary);
            if (!saveFile.good())
            {
                throw std::runtime_error("the file could not be opened");
            }
        }
		
		// ----- PUBLIC LOGGER METHODS -----
		Mode getMode() const override
		{
			return NetworkDelegate::Mode::logger;
		}
		
        void getArrivingSpike(double timestamp, projection* p, bool spiked, bool empty, Network* network, Neuron* postNeuron) override
        {
        	if (!empty)
        	{
        		copy_to(bytes.data() + 0, timestamp);
        		copy_to(bytes.data() + 8, p->delay);
        		copy_to(bytes.data() + 12, p->weight);
				copy_to(bytes.data() + 16, p->postNeuron->getPotential());
				copy_to(bytes.data() + 20, p->preNeuron ? p->preNeuron->getNeuronID() : -1);
				copy_to(bytes.data() + 22, p->postNeuron->getNeuronID());
				copy_to(bytes.data() + 24, p->postNeuron->getLayerID());
				copy_to(bytes.data() + 26, p->postNeuron->getRFID());
				copy_to(bytes.data() + 28, p->postNeuron->getX());
				copy_to(bytes.data() + 30, p->postNeuron->getY());
				saveFile.write(bytes.data(), bytes.size());
            }
        }
    
    protected:
		
		template <typename T>
		static void copy_to(char* target, T t) {
		    *reinterpret_cast<T*>(target) = t;
		}
		
    	// ----- IMPLEMENTATION VARIABLES -----
        std::ofstream saveFile;
        std::vector<char> bytes;
    };
}
