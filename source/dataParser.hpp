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
#include <random>
#include <deque>

#include "../third_party/filesystem.hpp"

namespace fs = ghc::filesystem;

namespace hummus {
    
	struct label {
		std::string name;
		double      onset;
	};
	
	struct event {
		double   timestamp;
        int      neuron_id;
		int      x;
		int      y;
	};
	
	class DataParser {
        
    public:
    	// ----- CONSTRUCTOR -----
        DataParser() {
            std::random_device device;
            random_engine = std::mt19937(device());
            gaussian = std::normal_distribution<>(0.0, 1.0);
        };
        
        // saves the files from a database of files formatted to eventstream format into a vector of strings
        std::vector<std::string> generate_database(const std::string directory_path, int sample_percentage=100) {
            std::vector<std::string> database;
            fs::path current_dir(directory_path);
            // save all files containing the .es extension in the database vector
            for (auto &file : fs::recursive_directory_iterator(current_dir)) {
                if (file.path().extension() == ".es") {
                    database.emplace_back(file.path());
                }
            }
            
            // shuffle the database vector
            std::random_device r;
            std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
            
            // create a random engine
            std::mt19937 random_engine(seed);
            
            std::shuffle(begin(database), end(database), random_engine);
            
            // get the number of samples from the percentage
            if (sample_percentage < 100) {
                size_t number_of_samples = static_cast<size_t>(std::ceil(database.size() * sample_percentage / 100));
                return std::vector<std::string>(database.begin(), database.begin()+number_of_samples);
            } else {
                return database;
            }
        }
        
        // saves the files from the N-MNIST database - formatted to eventstream format - into a a pair of vector: a vector of strings for the full paths to files, and a vector of labels
        // The N-MNIST database needs to have the same structure as the original folder otherwise the labels will be messed up. For example: ~/N-MNIST/Train/0
        std::pair<std::vector<std::string>, std::deque<label>> generate_nmnist_database(const std::string directory_path, int sample_percentage=100) {
            std::vector<std::string> database;
            std::deque<label> labels;
            
            fs::path current_dir(directory_path);
            // save all files containing the .es extension in the database vector
            for (auto &file : fs::recursive_directory_iterator(current_dir)) {
                if (file.path().extension() == ".es") {
                    labels.emplace_back(label{std::string(1, file.path().parent_path().string().back()), -1});
                    database.emplace_back(file.path());
                }
            }
            
            // shuffle the database and labels vectors
            std::random_device r;
            std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
            
            // create two random engines with the same state
            std::mt19937 random_engine1(seed);
            auto random_engine2 = random_engine1;
            
            std::shuffle(begin(database), end(database), random_engine1);
            std::shuffle(begin(labels), end(labels), random_engine2);
            
            // get the number of samples from the percentage
            if (sample_percentage < 100) {
                size_t number_of_samples = static_cast<size_t>(std::ceil(database.size() * sample_percentage / 100));
                return std::make_pair(std::vector<std::string>(database.begin(), database.begin()+number_of_samples), std::deque<label>(labels.begin(), labels.begin()+number_of_samples));
            } else {
                return std::make_pair(database, labels);
            }
        }
        
		// reading 1D (timestamp, Index), 2D data (timestamp, X, Y) or 2D data with a polarity (timestamp, X, Y, P)
        std::vector<event> read_txt_data(std::string filename, double shift_timestamps=0, bool time_jitter=false, int additive_noise=0) {
            data_file.open(filename);
            
            if (data_file.good()) {
				std::vector<event> data;
				std::string line;
				bool one_dimentional = false;
				int neuron_counter = 0;
                int max_id=0;
                int max_x=0;
                int max_y=0;
                
                while (std::getline(data_file, line)) {
                	std::vector<std::string> fields;
                	split(fields, line, " ,");
                	// 1D data
                	if (fields.size() == 2) {
						data.emplace_back(event{std::stod(fields[0]), std::stoi(fields[1]), -1, -1});
                        max_id = std::max(max_id, std::stoi(fields[1]));
                    // 2D Data
					} else if (fields.size() == 3) {
                		one_dimentional = true;
                		data.emplace_back(event{std::stod(fields[0]), neuron_counter, std::stoi(fields[1]), std::stoi(fields[2])});
                        max_x = std::max(max_x, std::stoi(fields[1]));
                        max_y = std::max(max_y, std::stoi(fields[2]));
                        
                		neuron_counter++;
					}
                }
                data_file.close();
                
                // adding gaussian time jitter + shiting the timestamp
                if (time_jitter) {
                    for (auto& datum: data) {
                        datum.timestamp += gaussian(random_engine);
                    }
                }
                
                // shiting the timestamps
                if (shift_timestamps != 0) {
                    for (auto& datum: data) {
                        datum.timestamp += shift_timestamps;
                    }
                }
                
                // additive noise
                if (additive_noise > 0) {
                    // finding maximum timestamp
                    auto it = std::max_element(data.begin(), data.end(), [&](event a, event b){ return a.timestamp < b.timestamp; });
                    double max_timestamp = data[std::distance(data.begin(), it)].timestamp;
                    
                    // uniform int distribution for the timestamps of spontaneous spikes
                    std::uniform_int_distribution<double> uniform_timestamp(0, max_timestamp);
                    
                    // finding the number of spontaneous spikes to add to the data
                    int additive_spikes = std::round(data.size() * additive_noise / 100.);
                    
                    // one-dimensional data
                    if (one_dimentional) {
                        std::uniform_int_distribution<> uniform_id(0, max_id);
                        
                        for (auto i=0; i<additive_spikes; i++) {
                            data.emplace_back(event{uniform_timestamp(random_engine), uniform_id(random_engine), UINT16_MAX, UINT16_MAX});
                        }
                    // two-dimensional data
                    } else {
                        std::uniform_int_distribution<> uniform_x(0, max_x);
                        std::uniform_int_distribution<> uniform_y(0, max_y);
                        
                        for (auto i=0; i<additive_spikes; i++) {
                            data.emplace_back(event{uniform_timestamp(random_engine), 0, uniform_x(random_engine), uniform_y(random_engine)});
                        }
                    }
                }
                
                // sorting data according to timestamps
				std::sort(data.begin(), data.end(), [](event a, event b) {
					return a.timestamp < b.timestamp;
				});
				
				return data;
            } else {
                throw std::runtime_error(filename.append(" could not be opened. Please check that the path is set correctly: if your path for data input is relative to the executable location, please use cd release && ./applicationName instead of ./release/applicationName"));
            }
        }
		
        // read a weight matrix file delimited by a space or a comma, where the inputs are the columns and the outputs are the rows
        std::vector<std::vector<double>> read_connectivity_matrix(std::string filename) {
            data_file.open(filename);
            
            if (data_file.good()) {
                std::string line;
                std::vector<std::vector<double>> data;
                
                while (std::getline(data_file, line)) {
                    std::vector<std::string> fields;
                    split(fields, line, " ,");
                    
                    std::vector<double> postsynaptic_weights;
                    
                    // filling temporary vector by each field of the line read, then convert the field to double
                    for (auto& f: fields) {
                        postsynaptic_weights.emplace_back(std::stod(f));
                    }
                    
                    // filling vector of vectors to build 2D weight matrix
                    data.emplace_back(postsynaptic_weights);
                }
                data_file.close();
                return data;
            } else {
                throw std::runtime_error(filename.append(" could not be opened. Please check that the path is set correctly: if your path for data input is relative to the executable location, please use cd release && ./applicationName instead of ./release/applicationName"));
            }
        }
        
		std::deque<label> read_txt_labels(std::string labels = "") {
			if (labels.empty()) {
				throw std::logic_error("no files were passed to the readLabels() function. There is nothing to do.");
			} else {
				data_file.open(labels);
				if (data_file.good()) {
					std::deque<label> dataLabels;
					std::string line;
					
					while (std::getline(data_file, line)) {
						std::vector<std::string> fields;
						split(fields, line, " ,");
						if (fields.size() == 2) {
							dataLabels.emplace_back(label{fields[0], std::stod(fields[1])});
						}
					}
					data_file.close();
					
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
				result.emplace_back(s.substr(current, next - current));
			}
			while (next != Container::value_type::npos);
			return result;
		}
        
    protected:
        
    	// ----- IMPLEMENTATION VARIABLES -----
        std::ifstream                   data_file;
        std::mt19937                    random_engine;
        std::normal_distribution<>      gaussian;
	};
}
