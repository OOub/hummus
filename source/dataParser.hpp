/*
 * dataParser.hpp
 * Hummus - spiking neural network simulator
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

#ifdef POSIX
#include <sys/types.h>
#include <dirent.h>
#endif

#ifdef WINDOWS
#include <windows.h>
#endif


namespace hummus {
    
	struct label {
		std::string name;
		double onset;
	};
	
	struct input {
		double timestamp;
        double neuronID;
		double x;
		double y;
        double polarity;
	};
	
	class DataParser {
        
    public:
    	// ----- CONSTRUCTOR -----
        DataParser() {
            std::random_device device;
            randomEngine = std::mt19937(device());
            gaussian = std::normal_distribution<>(0.0, 1.0);
        };
		
		// reading 1D (timestamp, Index), 2D data (timestamp, X, Y) or 2D data with a polarity used as the sublayerID (timestamp, X, Y, P)
        std::vector<input> readData(std::string filename, bool timeJitter=false, int additiveNoise=0) {
            dataFile.open(filename);
            
            if (dataFile.good()) {
				std::vector<input> data;
				std::string line;
				int dataType = 0;
				double neuronCounter = 0;
                
                double maxID=0; double maxX=0; double maxY=0; double maxPolarity=0;
                
				while (std::getline(dataFile, line)) {
                	std::vector<std::string> fields;
                	split(fields, line, " ,");
                	// 1D data
                	if (fields.size() == 2) {
                		dataType = 0;
						data.push_back(input{std::stod(fields[0]), std::stod(fields[1]), -1, -1, -1});
                        maxID = std::max(maxID, std::stod(fields[1]));
                    // 2D Data
					} else if (fields.size() == 3) {
                		dataType = 1;
                		data.push_back(input{std::stod(fields[0]), neuronCounter, std::stod(fields[1]), std::stod(fields[2]), -1});
                        maxX = std::max(maxX, std::stod(fields[1]));
                        maxY = std::max(maxY, std::stod(fields[2]));
                        
                		neuronCounter++;
					} else if (fields.size() == 4) {
                        dataType = 2;
                        data.push_back(input{std::stod(fields[0]), neuronCounter, std::stod(fields[1]), std::stod(fields[2]), std::stod(fields[3])});
                        maxX = std::max(maxX, std::stod(fields[1]));
                        maxY = std::max(maxY, std::stod(fields[2]));
                        maxPolarity = std::max(maxPolarity, std::stod(fields[3]));
                        neuronCounter++;
                    }
                }
                dataFile.close();
                
                // adding gaussian time jitter
                if (timeJitter) {
                    for (auto& datum: data) {
                        datum.timestamp += gaussian(randomEngine);
                    }
                }
                
                // additive noise
                if (additiveNoise > 0) {
                    // finding maximum timestamp
                    auto it = std::max_element(data.begin(), data.end(), [&](input a, input b){ return a.timestamp < b.timestamp; });
                    double maxTimestamp = data[std::distance(data.begin(), it)].timestamp;
                    
                    // uniform int distribution for the timestamps of spontaneous spikes
                    std::uniform_int_distribution<> uniformTimestamp(0, maxTimestamp);
                    
                    // finding the number of spontaneous spikes to add to the data
                    int additiveSpikes = std::round(data.size() * additiveNoise / 100.);
                    
                    // one-dimensional data
                    if (dataType == 0) {
                        std::uniform_int_distribution<> uniformID(0, maxID);
                        
                        for (auto i=0; i<additiveSpikes; i++) {
                            data.push_back(input{static_cast<double>(uniformTimestamp(randomEngine)), static_cast<double>(uniformID(randomEngine)), -1, -1});
                        }
                    // two-dimensional data
                    } else if (dataType == 1){
                        std::uniform_int_distribution<> uniformX(0, maxX);
                        std::uniform_int_distribution<> uniformY(0, maxY);
                        
                        for (auto i=0; i<additiveSpikes; i++) {
                            data.push_back(input{static_cast<double>(uniformTimestamp(randomEngine)), 0, static_cast<double>(uniformX(randomEngine)), static_cast<double>(uniformY(randomEngine)), -1});
                        }
                    } else if (dataType == 2){
                        std::uniform_int_distribution<> uniformX(0, maxX);
                        std::uniform_int_distribution<> uniformY(0, maxY);
                        std::uniform_int_distribution<> uniformPolarity(0, maxPolarity);
                        
                        for (auto i=0; i<additiveSpikes; i++) {
                            data.push_back(input{static_cast<double>(uniformTimestamp(randomEngine)), 0, static_cast<double>(uniformX(randomEngine)), static_cast<double>(uniformY(randomEngine)), static_cast<double>(uniformPolarity(randomEngine))});
                        }
                    }
                }
                
                // sorting data according to timestamps
				std::sort(data.begin(), data.end(), [](input a, input b) {
					return a.timestamp < b.timestamp;
				});
				
				return data;
            } else {
                throw std::runtime_error(filename.append(" could not be opened. Please check that the path is set correctly: if your path for data input is relative to the executable location, please use cd release && ./applicationName instead of ./release/applicationName"));
            }
        }
		
        // read a weight matrix file delimited by a space or a comma, where the inputs are the columns and the outputs are the rows
        std::vector<std::vector<double>> readWeightMatrix(std::string filename) {
            dataFile.open(filename);
            
            if (dataFile.good()) {
                std::string line;
                std::vector<std::vector<double>> data;
                
                while (std::getline(dataFile, line)) {
                    std::vector<std::string> fields;
                    split(fields, line, " ,");
                    
                    std::vector<double> postSynapticWeights;
                    
                    // filling temporary vector by each field of the line read, then convert the field to double
                    for (auto& f: fields) {
                        postSynapticWeights.push_back(std::stod(f));
                    }
                    
                    // filling vector of vectors to build 2D weight matrix
                    data.push_back(postSynapticWeights);
                }
                dataFile.close();
                return data;
            } else {
                throw std::runtime_error(filename.append(" could not be opened. Please check that the path is set correctly: if your path for data input is relative to the executable location, please use cd release && ./applicationName instead of ./release/applicationName"));
            }
        }
        
		std::deque<label> readLabels(std::string labels = "") {
			if (labels.empty()) {
				throw std::logic_error("no files were passed to the readLabels() function. There is nothing to do.");
			} else {
				dataFile.open(labels);
				if (dataFile.good()) {
					std::deque<label> dataLabels;
					std::string line;
					
					while (std::getline(dataFile, line)) {
						std::vector<std::string> fields;
						split(fields, line, " ,");
						if (fields.size() == 2) {
							dataLabels.push_back(label{fields[0], std::stod(fields[1])});
						}
					}
					dataFile.close();
					
					return dataLabels;
				} else {
					throw std::runtime_error(labels.append(" could not be opened. Please check that the path is set correctly"));
				}
			}
		}
        
		template <typename Container>
		static Container& split(Container& result, const typename Container::value_type& s, const typename Container::value_type& delimiters) {
			result.clear();
			size_t current;
			size_t next = -1;
			do {
				// strip front and back whitespaces
				next = s.find_first_not_of(delimiters, next + 1);
				if (next == Container::value_type::npos) {
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
        std::ifstream                   dataFile;
        std::mt19937                    randomEngine;
        std::normal_distribution<>      gaussian;
	};
}
