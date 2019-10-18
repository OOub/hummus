/*
 * display.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/07/2018
 *
 * Information: Add-on used to display a GUI of the spiking neural network output using Qt (Qt5 dependency)
 */

#pragma once

#include <numeric>
#include <vector>
#include <thread>
#include <chrono>
#include <string>

// QT5 Dependency
#include <QtWidgets/QApplication>
#include <QQmlApplicationEngine>
#include <QtQuick/QQuickView>

#include "input_viewer.hpp"
#include "output_viewer.hpp"
#include "dynamics_viewer.hpp"

namespace hummus {
    
    class Synapse;
    class Neuron;
    class Network;
    
    class Display : public MainAddon {
        
    public:

    	// ----- CONSTRUCTOR AND DESTRUCTOR-----
        Display() :
                neuronToTrack(-1),
                inputSublayerToTrack(0),
                outputLayerToTrack(1),
                outputSublayerToTrack(0) {
        	static int argc = 1;
			static char* argv[1] = {NULL};

			app.reset(new QApplication(argc, argv));

            qmlRegisterType<InputViewer>("InputViewer", 1, 0, "InputViewer");
            qmlRegisterType<OutputViewer>("OutputViewer", 1, 0, "OutputViewer");
            qmlRegisterType<DynamicsViewer>("DynamicsViewer", 1, 0, "DynamicsViewer");

			engine.reset(new QQmlApplicationEngine());

			engine->rootContext()->setContextProperty("layers", 1);
			engine->rootContext()->setContextProperty("inputSublayer", 1);
			engine->rootContext()->setContextProperty("sublayers", 1);
			engine->rootContext()->setContextProperty("numberOfNeurons", 1);
			engine->rootContext()->setContextProperty("displayCurrents", false);

            engine->load(QUrl(QStringLiteral("qrc:/gui.qml")));
                    
            auto window = qobject_cast<QQuickWindow*>(engine->rootObjects().first());
                    
			QSurfaceFormat format;
            format.setDepthBufferSize(24);
            format.setStencilBufferSize(8);
            format.setVersion(3, 3);
            format.setProfile(QSurfaceFormat::CompatibilityProfile);
            window->setFormat(format);
            window->show();

            input_viewer = window->findChild<InputViewer*>("inputViewer");
            output_viewer = window->findChild<OutputViewer*>("outputViewer");
            dynamics_viewer = window->findChild<DynamicsViewer*>("dynamicsViewer");
        }
        
        virtual ~Display(){}
        
    	// ----- PUBLIC DISPLAY METHODS -----
		void incoming_spike(double timestamp, Synapse* s, Neuron* postsynaptic_neuron, Network* network) override {
            dynamics_viewer->handle_data(timestamp, postsynaptic_neuron->get_neuron_id(), postsynaptic_neuron->get_potential(), postsynaptic_neuron->get_current(), postsynaptic_neuron->get_threshold());
            
            if (output_viewer->get_layer_changed()) {
                engine->rootContext()->setContextProperty("sublayers", static_cast<int>(output_viewer->get_y_lookup()[output_viewer->get_layer_tracker()].size()-1));
                output_viewer->set_layer_changed(false);
            }
		}

        void neuron_fired(double timestamp, Synapse* s, Neuron* postsynaptic_neuron, Network* network) override {
            // so decision-making neurons which do not pass synapses don't crash
            if (s) {
                input_viewer->handle_data(timestamp, s->get_presynaptic_neuron_id(), postsynaptic_neuron->get_neuron_id(), postsynaptic_neuron->get_sublayer_id());
            }
			output_viewer->handle_data(timestamp, postsynaptic_neuron->get_neuron_id(), postsynaptic_neuron->get_layer_id(), postsynaptic_neuron->get_sublayer_id());
            dynamics_viewer->handle_data(timestamp, postsynaptic_neuron->get_neuron_id(), postsynaptic_neuron->get_potential(), postsynaptic_neuron->get_current(), postsynaptic_neuron->get_threshold());
		}

		void status_update(double timestamp, Neuron* postsynaptic_neuron, Network* network) override {
            input_viewer->handle_update(timestamp);
            output_viewer->handle_update(timestamp);
            dynamics_viewer->handle_data(timestamp, postsynaptic_neuron->get_neuron_id(), postsynaptic_neuron->get_potential(), postsynaptic_neuron->get_current(), postsynaptic_neuron->get_threshold());
		}

		void begin(Network* network, std::mutex* sync) override {
            // finding the number of layers in the network
            int numberOfLayers = static_cast<int>(network->get_layers().size());

            // number of sublayers in each layer
            std::vector<int> sublayerInLayers;
            for (auto& l: network->get_layers()) {
                sublayerInLayers.emplace_back(l.sublayers.size());
            }

            // number of neurons in each layer
            std::vector<int> neuronsInLayers;
            for (auto& l: network->get_layers()) {
                neuronsInLayers.emplace_back(l.neurons.size());
            }

            // number of neurons in each sublayer
            std::vector<std::vector<int>> neuronsInSublayers(numberOfLayers);
            int idx = 0;
            for (auto& l: network->get_layers()) {
                for (auto& s: l.sublayers) {
                    neuronsInSublayers[idx].emplace_back(s.neurons.size());
                }
                idx += 1;
            }

            int neuronNumber = static_cast<int>(network->get_neurons().size());

            engine->rootContext()->setContextProperty("numberOfNeurons", neuronNumber);
            engine->rootContext()->setContextProperty("inputSublayer", sublayerInLayers[0]-1);
            engine->rootContext()->setContextProperty("layers", numberOfLayers-1);

            input_viewer->set_y_lookup(neuronsInSublayers[0]);
            output_viewer->set_y_lookup(neuronsInSublayers, neuronsInLayers);

            input_viewer->change_sublayer(inputSublayerToTrack);
            output_viewer->change_layer(outputLayerToTrack);
            output_viewer->change_sublayer(outputSublayerToTrack);
            dynamics_viewer->track_neuron(neuronToTrack);

            sync->unlock();

			app->exec();
		}

        void reset() override {
            input_viewer->reset();
            output_viewer->reset();
            dynamics_viewer->reset();
        }

		// ----- SETTERS -----
		void hardware_acceleration(bool accelerate=true) {
            input_viewer->hardware_acceleration(accelerate);
            output_viewer->hardware_acceleration(accelerate);
            dynamics_viewer->hardware_acceleration(accelerate);
        }

		void track_layer(int layerToTrack) {
			outputLayerToTrack = layerToTrack;
		}

		void track_input_sublayer(int sublayerToTrack) {
			inputSublayerToTrack = sublayerToTrack;
		}

		void track_output_sublayer(int sublayerToTrack) {
			outputSublayerToTrack = sublayerToTrack;
		}

        void track_neuron(size_t _neuron_to_track) {
            neuronToTrack = static_cast<int>(_neuron_to_track);
        }

        void track_neuron(int _neuron_to_track) {
        	neuronToTrack = _neuron_to_track;
        }

		void set_time_window(double new_window) {
            input_viewer->set_time_window(new_window);
            output_viewer->set_time_window(new_window);
            dynamics_viewer->set_time_window(new_window);
        }

        void plot_currents(bool current_plot=true) {
            engine->rootContext()->setContextProperty("displayCurrents", current_plot);
            dynamics_viewer->plot_currents(current_plot);
        }

    protected:

		// ----- IMPLEMENTATION VARIABLES -----
        std::unique_ptr<QApplication>          app;
        std::unique_ptr<QQmlApplicationEngine> engine;
        InputViewer*                           input_viewer;
        OutputViewer*                          output_viewer;
        DynamicsViewer*                        dynamics_viewer;
        int                                    neuronToTrack;
        int                                    inputSublayerToTrack;
        int                                    outputLayerToTrack;
        int                                    outputSublayerToTrack;
    };
}
