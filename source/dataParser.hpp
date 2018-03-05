/*
 * dataParser.hpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 22/02/2018
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
	
	struct dataPackage
	{
		std::vector<input> data;
		std::vector<std::vector<double>> connectionMatrix;
	};
	
	class DataParser
	{
    public:
    	// ----- CONSTRUCTOR -----
        DataParser(){}
		
		// reading one dimentional data (timestamp, Index)
        std::vector<input> read1D(std::string filename)
        {
            std::cout << "Reading " << filename << std::endl;
            dataFile.open(filename);
            
            if (dataFile.good())
            {
				std::vector<input> data;
				std::vector<double> columns(2);
                while (dataFile >> columns[0] >> columns[1])
                {
                	data.push_back(input{columns[0], columns[1]});
                }
				std::sort(data.begin(), data.end(), [](input a, input b)
				{
					return a.timestamp < b.timestamp;
				});
			
				std::cout << "Done." << std::endl;
				dataFile.close();
				return data;
            }
            else
            {
                throw std::runtime_error("the file could not be opened");
            }
        }
		
		// reading two dimentional data (timestamp, X, Y)
		dataPackage read2D(std::string filename, int width)
		{
			std::cout << "Reading " << filename << std::endl;
			dataFile.open(filename);
			if (dataFile.good())
            {
				dataPackage output;
				std::vector<input> data;
				std::vector<double> columns(3);
				while (dataFile >> columns[0] >> columns[1] >> columns[2])
				{
					data.push_back(input{columns[0],columns[1]+width*columns[2]});
					// step2: do something with columns[1] and columns[2] to get the connectionMatrix
				}
				dataFile.close();
				return output;
            }
			else
            {
                throw std::runtime_error("the file could not be opened");
            }
		}
		
    protected:
    	// ----- IMPLEMENTATION VARIABLES -----
        std::ifstream dataFile;
	};
}
