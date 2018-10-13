/*
 * dataParser.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/05/2018
 *
 * Information: The DataParser class is used to input data from files into a vector.
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

namespace adonis_c
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
	};
	
	class DataParser
	{
    public:
    	// ----- CONSTRUCTOR -----
        DataParser() :
        	timeShift(nullptr)
        {}
		
		// reading 1D (timestamp, Index) or 2D data (timestamp, X, Y)
        std::vector<input> readTrainingData(std::string filename, bool changeTimeShift=true)
        {
            dataFile.open(filename);
            
            if (dataFile.good())
            {
				std::vector<input> data;
				std::string line;
				bool dataType = false;
				double neuronCounter = 0;
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
                		data.push_back(input{std::stod(fields[0]), neuronCounter, std::stod(fields[1]), std::stod(fields[2])});
                		neuronCounter++;
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
				
				if (changeTimeShift)
				{
					timeShift = &data.back().timestamp;
				}
				
				std::cout << "Done." << std::endl;
				return data;
            }
            else
            {
                throw std::runtime_error(filename.append(" could not be opened. Please check that the path is set correctly: if your path for data input is relative to the executable location, please use cd release && ./applicationName instead of ./release/applicationName"));
            }
        }
		
		// reads test data in the same format as the training data, stops learning at the end of the training data time and shifts the test data timestamps accordingly
		template<typename Network>
		std::vector<input> readTestData(Network* network, std::string filename)
		{
			if (timeShift)
			{
				auto data = readTrainingData(filename, false);
				network->turnOffLearning(*timeShift);
				for (auto& input: data)
				{
					input.timestamp += *timeShift + 1000;
				}
				return data;
			}
			else
			{
				throw std::logic_error("are you sure training data was fed into the network via the readTrainingData() before starting to feed test data using readTestData()?");
			}
				
		}
		
		// reads a teacher signal
		std::deque<double> readTeacherSignal(std::string filename)
		{
		 dataFile.open(filename);
			
            if (dataFile.good())
            {
				std::deque<double> data;
				std::string line;
				
				std::cout << "Reading teacher signal " << filename << std::endl;
				while (std::getline(dataFile, line))
                {
                	std::vector<std::string> fields;
                	split(fields, line, " ");
					
                	if (fields.size() == 1)
                	{
						data.push_back(std::stod(fields[0]));
					}
                }
                dataFile.close();
			
				std::cout << "Done." << std::endl;
				return data;
            }
            else
            {
                throw std::runtime_error(filename.append(" could not be opened. Please check that the path is set correctly: if your path for data input is relative to the executable location, please use cd release && ./applicationName instead of ./release/applicationName"));
            }
		}
		
		std::deque<label> readLabels(std::string trainingLabels = "", std::string testLabels = "")
		{
			if (trainingLabels.empty() && testLabels.empty())
			{
				throw std::logic_error("no files were passed to the readLabels() function. Therefore, there is nothing to do.");
			}
			else if (!trainingLabels.empty() && testLabels.empty())
			{
				return readLabelsHelper(trainingLabels);
			}
			else if (trainingLabels.empty() && !testLabels.empty())
			{
				if (timeShift)
				{
					std::deque<label> dataLabels = readLabelsHelper(testLabels);
					for (auto& lbl: dataLabels)
					{
						lbl.onset += *timeShift + 1000;
					}
					return dataLabels;
				}
				else
				{
					throw std::logic_error("are you sure training data was fed into the network via the readTrainingData() before reading the test data labels?");
				}
				
			}
			else
			{
				if (timeShift)
				{
					std::deque<label> dataLabels = readLabelsHelper(trainingLabels);
					std::deque<label> test = readLabelsHelper(testLabels);
					for (auto& lbl: test)
					{
						lbl.onset += *timeShift + 1000;
					}
					dataLabels.insert(dataLabels.end(), test.begin(), test.end());
					return dataLabels;

				}
				else
				{
					throw std::logic_error("are you sure training data was fed into the network via the readTrainingData() before reading the training and test data labels?");
				}
			}
		}
		
		std::deque<label> readLabelsHelper(std::string filename)
		{
			dataFile.open(filename);
			if (dataFile.good())
            {
            	std::deque<label> dataLabels;
				std::string line;
				std::cout << "Reading labels from " << filename << std::endl;
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
                throw std::runtime_error(filename.append(" could not be opened. Please check that the path is set correctly"));
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
        std::ifstream    dataFile;
        double*          timeShift;
	};
}
