/*
 * dataParser.hpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 17/01/2018
 *
 * Information: The DataParser class is used to input data from files into a vector.
 */

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

namespace baal
{
	struct input
	{
		double timestamp;
		double neuronID;
	};
		
	class DataParser
	{
    public:
    	// ----- CONSTRUCTOR -----
        DataParser(){}
		
		// reading one dimentional data
        std::vector<input> read1D(std::string filename)
        {
        	std::vector<input> data;
            std::vector<double> columns(2);
            
            std::cout << "Reading " << filename << std::endl;
            dataFile.open(filename);
            
            if (dataFile.good())
            {
                while (dataFile >> columns[0] >> columns[1])
                {
                	data.push_back(input{columns[0], columns[1]});
                }
            }
            else
            {
                throw std::runtime_error("the file could not be opened");
            }
            dataFile.close();
			
			std::sort(data.begin(), data.end(), [](input a, input b){
				return a.timestamp < b.timestamp;
			});
			
            std::cout << "Done." << std::endl;
            return data;
        }
    
    protected:
    	// ----- IMPLEMENTATION VARIABLES -----
        std::ifstream dataFile;
	};
}
