/*
 * qtDisplay.hpp
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

#include "../../core.hpp"
#include "inputViewer.hpp"
#include "outputViewer.hpp"
#include "dynamicsViewer.hpp"

namespace hummus {
    class QtDisplay : public MainThreadAddon {
        
    public:

    	// ----- CONSTRUCTOR AND DESTRUCTOR-----
        QtDisplay() :
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
			
			engine = new QQmlApplicationEngine();
			
			engine->rootContext()->setContextProperty("layers", 1);
			engine->rootContext()->setContextProperty("inputSublayer", 1);
			engine->rootContext()->setContextProperty("sublayers", 1);
			engine->rootContext()->setContextProperty("numberOfNeurons", 1);
			engine->rootContext()->setContextProperty("displayCurrents", false);
                    
            engine->loadData(
				#include "gui.qml"
            );
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
        
        virtual ~QtDisplay(){}
        
    	// ----- PUBLIC DISPLAY METHODS -----
		void incomingSpike(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
			dynamics_viewer->handleData(timestamp, s, postsynapticNeuron, network);
		}

        void neuronFired(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
			input_viewer->handleData(timestamp, s, postsynapticNeuron, network);
			output_viewer->handleData(timestamp, s, postsynapticNeuron, network);
			dynamics_viewer->handleData(timestamp, s, postsynapticNeuron, network);
		}

		void timestep(double timestamp, Neuron* postsynapticNeuron, Network* network) override {
			input_viewer->handleTimestep(timestamp);
			output_viewer->handleTimestep(timestamp);
			dynamics_viewer->handleTimestep(timestamp, postsynapticNeuron, network);
		}

        void statusUpdate(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            input_viewer->handleTimestep(timestamp);
            output_viewer->handleTimestep(timestamp);
            dynamics_viewer->handleData(timestamp, s, postsynapticNeuron, network);
        }
        
		void begin(Network* network, std::mutex* sync) override {
            // finding the number of layers in the network
            int numberOfLayers = static_cast<int>(network->getLayers().size());

            // number of sublayers in each layer
            std::vector<int> sublayerInLayers;
            for (auto& l: network->getLayers()) {
                sublayerInLayers.emplace_back(l.sublayers.size());
            }

            // number of neurons in each layer
            std::vector<int> neuronsInLayers;
            for (auto& l: network->getLayers()) {
                neuronsInLayers.emplace_back(l.neurons.size());
            }

            // number of neurons in each sublayer
            std::vector<std::vector<int>> neuronsInSublayers(numberOfLayers);
            int idx = 0;
            for (auto& l: network->getLayers()) {
                for (auto& s: l.sublayers) {
                    neuronsInSublayers[idx].emplace_back(s.neurons.size());
                }
                idx += 1;
            }

            int neuronNumber = static_cast<int>(network->getNeurons().size());
            
            engine->rootContext()->setContextProperty("numberOfNeurons", neuronNumber);
            engine->rootContext()->setContextProperty("inputSublayer", sublayerInLayers[0]-1);
            engine->rootContext()->setContextProperty("layers", numberOfLayers-1);

            input_viewer->setYLookup(neuronsInSublayers[0]);
            output_viewer->setEngine(engine);
            output_viewer->setYLookup(neuronsInSublayers, neuronsInLayers);

            input_viewer->changeSublayer(inputSublayerToTrack);
            output_viewer->changeLayer(outputLayerToTrack);
            output_viewer->changeSublayer(outputSublayerToTrack);
            dynamics_viewer->trackNeuron(neuronToTrack);
			
            sync->unlock();

			app->exec();
		}
		
		// ----- SETTERS -----
		void useHardwareAcceleration(bool accelerate) {
            input_viewer->useHardwareAcceleration(accelerate);
            output_viewer->useHardwareAcceleration(accelerate);
            dynamics_viewer->useHardwareAcceleration(accelerate);
        }

		void trackLayer(int layerToTrack) {
			outputLayerToTrack = layerToTrack;
		}
		
		void trackInputSublayer(int sublayerToTrack) {
			inputSublayerToTrack = sublayerToTrack;
		}
		
		void trackOutputSublayer(int sublayerToTrack) {
			outputSublayerToTrack = sublayerToTrack;
		}
		
        void trackNeuron(size_t _neuronToTrack) {
        	neuronToTrack = _neuronToTrack;
        }

		void setTimeWindow(double newWindow) {
            input_viewer->setTimeWindow(newWindow);
            output_viewer->setTimeWindow(newWindow);
            dynamics_viewer->setTimeWindow(newWindow);
        }
		
        void plotCurrents(bool current_plot) {
            engine->rootContext()->setContextProperty("displayCurrents", current_plot);
            dynamics_viewer->plotCurrents(current_plot);
        }
        
    protected:

		// ----- IMPLEMENTATION VARIABLES -----
        std::unique_ptr<QApplication>          app;
        QQmlApplicationEngine*                 engine;
        InputViewer*                           input_viewer;
        OutputViewer*                          output_viewer;
        DynamicsViewer*                        dynamics_viewer;
        size_t                                 neuronToTrack;
        int                                    inputSublayerToTrack;
        int                                    outputLayerToTrack;
        int                                    outputSublayerToTrack;
    };
}
