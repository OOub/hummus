/*
 * dataParser.hpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/05/2018
 *
 * Information: The DataParser class is used to input data from files into a vector.
 * To-Do: implement a way to split into receptive fields by splitting the coordinate system depending on the dimensions and output that into a vector of vectors.
 */

#pragma once

#include <vector>
#include <string>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace baal
{
	struct input
	{
		double timestamp;
		double neuronID;
		double x;
		double y;
	};
	
	class DataParser
	{
    public:
    	// ----- CONSTRUCTOR -----
        DataParser(){}
		
		// reading 1D (timestamp, Index) or 2D data (timestamp, X, Y). For the 2D data, the width of the 2D patch needs to be included as a parameter
		// encode the 2D neurons differently to simply this function because the 2D to 1D mapping is now obsolete
        std::vector<input> readData(std::string filename, int width=24)
        {
            dataFile.open(filename);
            
            if (dataFile.good())
            {
				std::vector<input> data;
				std::string line;
				bool dataType = false;
				
				std::cout << "Reading " << filename << std::endl;
				while (std::getline(dataFile, line))
                {
                	std::vector<std::string> fields;
                	split(fields, line, " ");
                	// 1D data
                	if (fields.size() == 2)
                	{
						data.push_back(input{std::stod(fields[0]), std::stod(fields[1]), -1, -1});
					}
					// 2D data
                	else if (fields.size() == 3)
                	{
                		if (!dataType)
                		{
                			dataType = true;
						}
                		data.push_back(input{std::stod(fields[0]), std::stod(fields[1])+width*std::stod(fields[2]), std::stod(fields[1]), std::stod(fields[2])});
					}
                }
                dataFile.close();
				
				
				if (dataType)
				{
					std::cout << "2D data detected" << std::endl;
				}
				else
				{
					std::cout << "1D data detected" << std::endl;
				}
				
				std::sort(data.begin(), data.end(), [](input a, input b)
				{
					return a.timestamp < b.timestamp;
				});
			
				std::cout << "Done." << std::endl;
				return data;
            }
            else
            {
                throw std::runtime_error("the file could not be opened");
            }
        }
		
		template <typename Container>
		Container& split(Container& result, const typename Container::value_type& s, const typename Container::value_type& delimiters)
		{
			result.clear();
			size_t current;
			size_t next = -1;
			do
			{
				// strip front and back whitespaces
				next = s.find_first_not_of(delimiters, next + 1);
				if (next == Container::value_type::npos)
				{
					break;
				}
				next -= 1;
				
				// split string according to delimiters
				current = next + 1;
				next = s.find_first_of(delimiters, current);
				result.push_back(s.substr(current, next - current));
			}
			while (next != Container::value_type::npos);
			return result;
		}

    protected:
    	// ----- IMPLEMENTATION VARIABLES -----
        std::ifstream dataFile;
	};
}
