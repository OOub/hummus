/*
 * dataParser.hpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/11/2017
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
	class DataParser
	{
    public:
    	// ----- CONSTRUCTOR -----
        DataParser(){}
		
		// reading one dimentional data
        std::vector<std::vector<float>> read1D(std::string filename)
        {
            std::vector<std::vector<float>> data(2);
            std::vector<float> columns(2);
            
            std::cout << "Reading data..." << std::endl;
            dataFile.open(filename);
            
            if (dataFile.good())
            {
                while (dataFile >> columns[0] >> columns[1])
                {
                    data[0].push_back(columns[0]);
                    data[1].push_back(columns[1]);
                }
            }
            else
            {
                throw std::runtime_error("the file could not be opened");
            }
            dataFile.close();
            
            std::cout << "the data contains " << *std::max_element(data[1].begin(),data[1].end())+1 << " input neurons." << std::endl;
            std::cout << "Done." << std::endl;
            return data;
        }
    
    protected:
    	// ----- IMPLEMENTATION VARIABLES -----
        std::ifstream dataFile;
	};
}
