/*
 * dataParser.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 06/12/2018
 *
 * Information: The DataParser class is used to read datasets and labels
 */

#pragma once

#include <vector>
#include <string>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <deque>

namespace adonis
{
	struct label
	{
		std::string name;
		double onset;
	};
	
	struct input
	{
		double timestamp;
		double neuronID;
		double x;
		double y;
		double sublayerID;
	};
	
	class DataParser
	{
    public:
    	// ----- CONSTRUCTOR -----
        DataParser() = default;
		
		// reading 1D (timestamp, Index), 2D data (timestamp, X, Y) or 2D data divided into sublayers (timestamp, X, Y, sublayerID)
        std::vector<input> readData(std::string filename)
        {
            dataFile.open(filename);
            
            if (dataFile.good())
            {
				std::vector<input> data;
				std::string line;
				int dataType = 0;
				double neuronCounter = 0;
				std::cout << "Reading " << filename << std::endl;
				while (std::getline(dataFile, line))
                {
                	std::vector<std::string> fields;
                	split(fields, line, " ");
                	// 1D data
                	if (fields.size() == 2)
                	{
                		dataType = 0;
						data.push_back(input{std::stod(fields[0]), std::stod(fields[1]), -1, -1, -1});
					}
					// 2D data
                	else if (fields.size() == 3)
                	{
                		dataType = 1;
                		data.push_back(input{std::stod(fields[0]), neuronCounter, std::stod(fields[1]), std::stod(fields[2]), -1});
                		neuronCounter++;
					}
					else if (fields.size() == 4)
					{
						dataType = 2;
						data.push_back(input{std::stod(fields[0]), neuronCounter, std::stod(fields[1]), std::stod(fields[2]), std::stod(fields[3])});
                		neuronCounter++;
					}
                }
                dataFile.close();
				
				
				if (dataType == 0)
				{
					std::cout << "1D data detected" << std::endl;
				}
				else if (dataType == 1)
				{
					std::cout << "2D data detected" << std::endl;
				}
				else if (dataType == 2)
				{
					std::cout << "2D data divided into sublayers detected" << std::endl;
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
                throw std::runtime_error(filename.append(" could not be opened. Please check that the path is set correctly: if your path for data input is relative to the executable location, please use cd release && ./applicationName instead of ./release/applicationName"));
            }
        }
		
		std::deque<label> readLabels(std::string labels = "")
		{
			if (labels.empty())
			{
				throw std::logic_error("no files were passed to the readLabels() function. There is nothing to do.");
			}
			else
			{
				dataFile.open(labels);
				if (dataFile.good())
				{
					std::deque<label> dataLabels;
					std::string line;
					std::cout << "Reading labels from " << labels << std::endl;
					while (std::getline(dataFile, line))
					{
						std::vector<std::string> fields;
						split(fields, line, " ");
						if (fields.size() == 2)
						{
							dataLabels.push_back(label{fields[0], std::stod(fields[1])});
						}
					}
					dataFile.close();
					std::cout << "Done." << std::endl;
					return dataLabels;
				}
				else
				{
					throw std::runtime_error(labels.append(" could not be opened. Please check that the path is set correctly"));
				}
			}
		}
		
		template <typename Container>
		static Container& split(Container& result, const typename Container::value_type& s, const typename Container::value_type& delimiters)
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
        std::ifstream    dataFile;
	};
}
