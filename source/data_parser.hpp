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
#include <filesystem>

namespace hummus {
    
	struct label {
		int    id;
		double timestamp;
	};
	
    struct event {
        double   timestamp;
        int      neuron_id;
        int      x = -1;
        int      y = -1;
    };

    struct dataset {
        std::vector<event>                   spikes;
        std::vector<std::string>             files;
        std::deque<label>                    labels;
        std::unordered_map<std::string, int> class_map;
    };
	
	class DataParser {
        
    public:
        
    	// ----- CONSTRUCTOR -----
        DataParser(bool seed_parser=false) {
            std::random_device device;
            if (seed_parser) {
                std::seed_seq seed{device(), device(), device(), device(), device(), device(), device(), device()};
                random_engine = std::mt19937(seed);
            } else {
                random_engine = std::mt19937(device());
            }
            gaussian = std::normal_distribution<double>(0.0, 1.0);
        };
        
        // loads data from a database of files
        //     * saves file paths into files string vector
        //     * saves labels into a vector of struct - label struct = (id, timestamp)
        //     * saves an unordered set that maps the class string labels to their corresponding ids
        dataset load_data(const std::string path, int sample_percentage=100, const std::vector<std::string> classes={}) {
            if (sample_percentage > 100 || sample_percentage <= 0) {
                throw std::logic_error("your sample is a percentage that needs to be between 1 and 100");
            }
            
            if (!std::filesystem::exists(path) || !std::filesystem::is_directory(path)) {
                throw std::logic_error("path doesn't exist or not a directory");
            }
            
            // if there's already a class map initialised we'll use it (so that label ids are the same across datasets)
            bool new_class_map = false;
            if (class_map.empty()) {
                new_class_map = true;
            }
            
            // initialise containers
            std::vector<event> spikes = {};
            std::vector<std::string> files = {};
            std::deque<label> labels = {};
            
            // save all files containing the .es or the .npy extension in the database vector
            int label_id = 0;
            std::filesystem::path current_dir(path);
            for (auto &file : std::filesystem::recursive_directory_iterator(current_dir)) {
                if (file.path().extension() == ".es" || file.path().extension() == ".npy") {
                    // only use specific classes
                    if (!classes.empty()) {
                        if (std::find(classes.begin(), classes.end(),file.path().parent_path().filename().string()) != classes.end()) {
                            // get label and convert to int
                            std::string str_label = file.path().parent_path().filename().string();
                            if (new_class_map) {
                                auto return_val = class_map.try_emplace(str_label, label_id);
                                if (return_val.second) {
                                    label_id++;
                                }
                            }
                            
                            // fill containers
                            labels.emplace_back(label{class_map[str_label], -1});
                            files.emplace_back(file.path());
                        }
                    // use all classes
                    } else {
                        // get label and convert to int
                        std::string str_label = file.path().parent_path().filename().string();
                        if (new_class_map) {
                            auto return_val = class_map.try_emplace(str_label, label_id);
                            if (return_val.second) {
                                label_id++;
                            }
                        }
                        
                        // fill containers
                        labels.emplace_back(label{class_map[str_label], -1});
                        files.emplace_back(file.path());
                    }
                }
            }

            // shuffle the database and labels vectors
            auto random_engine2 = random_engine;
            std::shuffle(files.begin(), files.end(), random_engine);
            std::shuffle(labels.begin(), labels.end(), random_engine2);

            // get the number of samples from the percentage
            if (sample_percentage < 100) {
                size_t number_of_samples = static_cast<size_t>(std::ceil(files.size() * sample_percentage / 100));
                return dataset{
                    {},
                    std::vector<std::string>(files.begin(), files.begin()+number_of_samples),
                    std::deque<label>(labels.begin(), labels.begin()+number_of_samples),
                    class_map};
                
            } else {
                return dataset{{}, files, labels, class_map};
            }
        }
        
        // loads data from a single npy file and optionally a txt file containing labels
        //     - data:
        //         * 2D data = (timestamp, neuron_id)
        //         * 3D data = (timestamp, x, y)
        //     - label format (string, timestamp)
        dataset load_data(const std::string data_path, const std::string label_path, double shift_timestamps=0.0, bool time_jitter=false) {
            
            std::filesystem::path given_path = data_path;
            if (!std::filesystem::exists(data_path) || !(given_path.extension() == ".npy")) {
                throw std::logic_error("path doesn't exist or not an npy file");
            }
            
            // initialise containers
            std::vector<event> spikes = {};
            std::vector<std::string> files = {};
            std::deque<label> labels = {};

            // load spikes into vector
            std::vector<int> npy_shape;
            std::vector<double> npy_data;
            aoba::LoadArrayFromNumpy<double>(data_path, npy_shape, npy_data);
            
            // shaping the spikes vector
            for (int i=0; i<static_cast<int>(npy_data.size()); i+=npy_shape[1]) {
                
                
                if (npy_shape[1] == 2) {
                    event new_spike = event{npy_data[i], static_cast<int>(npy_data[i+1])};
                    spikes.emplace_back(event{new_spike});
                } else if (npy_shape[1] == 3){
                    event new_spike = event{npy_data[i], static_cast<int>(npy_data[i+1]), static_cast<int>(npy_data[i+2])};
                    spikes.emplace_back(event{new_spike});
                } else {
                    throw std::logic_error("npy file is not formatted correctly");
                }
            }
            
            // adding gaussian time jitter + shiting the timestamp
            if (time_jitter) {
                std::transform(spikes.begin(), spikes.end(), spikes.begin(), [&](event& s){return event{
                    s.timestamp+gaussian(random_engine),
                    s.neuron_id,
                    s.x,
                    s.y};
                });
            }

            // shiting the timestamps
            if (shift_timestamps != 0) {
                std::transform(spikes.begin(), spikes.end(), spikes.begin(), [&](event& s){ return event{
                    s.timestamp+shift_timestamps,
                    s.neuron_id,
                    s.x,
                    s.y};
                });
            }
            
            // sort the vector according to timestamps
            std::sort(spikes.begin(), spikes.end(), [](event a, event b) {
                return a.timestamp < b.timestamp;
            });
            
            // load labels
            if (!label_path.empty()) {
                labels = read_txt_labels(label_path);
            }
            
            return dataset{spikes, files, labels, class_map};
        }

        // reads labels in a txt file. each line is : (string, timestamp)
        std::deque<label> read_txt_labels(std::string labels = "") {
			if (labels.empty()) {
				throw std::logic_error("no files were passed to the read_txt_labels() function. There is nothing to do.");
			} else {
				data_file.open(labels);
				if (data_file.good()) {
                    
                    // if there's already a class map initialised we'll use it (so that label ids are the same across datasets)
                    bool new_class_map = false;
                    if (class_map.empty()) {
                        new_class_map = true;
                    }
                    
                    // inialise containers
					std::deque<label> dataLabels;
					std::string line;
                    std::unordered_map<std::string, int> class_map;
                    
                    // looping through the txt file
                    int label_id = 0;
					while (std::getline(data_file, line)) {
						std::vector<std::string> fields;
						split(fields, line, " ,");
						if (fields.size() == 2) {
                            if (new_class_map) {
                                // get label and convert to int
                                auto return_val = class_map.try_emplace(fields[0], label_id);
                                if (return_val.second) {
                                    label_id++;
                                }
                            }
                            dataLabels.emplace_back(label{class_map[fields[0]], std::stod(fields[1])});
						}
					}
					data_file.close();

                    return dataLabels;
				} else {
					throw std::runtime_error(labels.append(" could not be opened. Please check that the path is set correctly"));
				}
			}
		}

        // read a weight matrix file delimited by a space or a comma, where the inputs are the columns and the outputs are the rows
        std::vector<std::vector<float>> read_connectivity_matrix(std::string filename) {
            data_file.open(filename);

            if (data_file.good()) {
                std::string line;
                std::vector<std::vector<float>> data;

                while (std::getline(data_file, line)) {
                    std::vector<std::string> fields;
                    split(fields, line, " ,");

                    std::vector<float> postsynaptic_weights;

                    // filling temporary vector by each field of the line read, then convert the field to float
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
        std::ifstream                        data_file;
        std::mt19937                         random_engine;
        std::normal_distribution<double>     gaussian;
        std::unordered_map<std::string, int> class_map;
	};
}
