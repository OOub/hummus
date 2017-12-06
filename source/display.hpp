/*
 * display.hpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 21/11/2017
 *
 * Information: Add-on to the Network class, used to display a GUI of the spiking neural network output
 */

#pragma once

#include <vector>
#include <thread>
#include <chrono>

// QT5 Dependency
#include <QtWidgets/QApplication>
#include <QQmlApplicationEngine>
#include <QtQuick/QQuickView>

#include "network.hpp"
#include "inputViewer.hpp"
#include "outputViewer.hpp"
#include "potentialViewer.hpp"

namespace baal
{
    class Display : public NetworkDelegate
    {
    public:
		
    	// ----- CONSTRUCTOR -----
        Display(std::vector<NetworkDelegate*> nd = {}, float threshold=-50)
        {
        	static int argc = 1;
			static char* argv[1] = {NULL};
			
			app.reset(new QApplication(argc, argv));
			
            nd.push_back(this);
            network = Network(nd);
			
            qmlRegisterType<InputViewer>("InputViewer", 1, 0, "InputViewer");
            qmlRegisterType<OutputViewer>("OutputViewer", 1, 0, "OutputViewer");
            qmlRegisterType<PotentialViewer>("PotentialViewer", 1, 0, "PotentialViewer");
			
            engine.reset(new QQmlApplicationEngine("../../source/gui.qml"));
            auto window = (QQuickWindow*) engine->rootObjects().first();
			
            format.setDepthBufferSize(24);
            format.setStencilBufferSize(8);
            format.setVersion(3, 3);
            format.setProfile(QSurfaceFormat::CompatibilityProfile);
            window->setFormat(format);
            
            window->show();
            
            inputviewer = window->findChild<InputViewer*>("inputViewer");
            outputviewer = window->findChild<OutputViewer*>("outputViewer");
            potentialviewer = window->findChild<PotentialViewer*>("potentialViewer");
        }
		
    	// ----- PUBLIC DISPLAY METHODS -----
        void getArrivingSpike(float timestamp, projection* p, bool spiked, bool empty, Network* network, Neuron* postNeuron) override
        {
            inputviewer->handleData(timestamp, p, spiked, empty, network, postNeuron);
            outputviewer->handleData(timestamp, p, spiked, empty, network, postNeuron);
            potentialviewer->handleData(timestamp, p, spiked, empty, network, postNeuron);
        }
		
		int run(float _runtime, float _timestep)
        {
            std::thread spikeManager([this, _runtime, _timestep]{
                network.run(_runtime, _timestep);
            });
            int errorCode = app->exec();
            spikeManager.join();
            return errorCode;
        }
		
		// ----- NETWORK CLASS WRAPPERS -----
		void addNeurons(int _numberOfNeurons, float _decayCurrent=10, float _decayPotential=20, int _refractoryPeriod=3, float _decaySynapticEfficacy=0, float _synapticEfficacy=1, float _threshold = -50, float  _restingPotential=-70, float _resetPotential=-70, float _inputResistance=50e9, float _externalCurrent=40e-10)
		{
			network.addNeurons(_numberOfNeurons,_decayCurrent,_decayPotential,_refractoryPeriod,_decaySynapticEfficacy,
				_synapticEfficacy,_threshold,_restingPotential,_resetPotential,_inputResistance, _externalCurrent);
		}
		
		void allToallConnectivity(std::vector<Neuron>* presynapticLayer, std::vector<Neuron>* postsynapticLayer, float weight, bool randomDelays, int _delay=0)
		{
			network.allToallConnectivity(presynapticLayer,postsynapticLayer,weight,randomDelays,_delay);
		}
		
		void injectSpike(spike s)
        {
            network.injectSpike(s);
        }
		
		std::vector<std::vector<Neuron>>& getNeuronPopulations()
		{
			return network.getNeuronPopulations();
		}
		
		void injectTeacher(std::vector<std::vector<float>>* _teacher)
        {
            network.injectTeacher(_teacher);
        }
		
		// ----- SETTERS -----
		void useHardwareAcceleration(bool accelerate)
        {
            inputviewer->useHardwareAcceleration(accelerate);
            outputviewer->useHardwareAcceleration(accelerate);
            potentialviewer->useHardwareAcceleration(accelerate);
        }
		
        void trackNeuron(int neuronToTrack)
        {
            potentialviewer->trackNeuron(neuronToTrack);
        }
		
		void setTimeWindow(float newWindow)
        {
            inputviewer->setTimeWindow(newWindow);
            outputviewer->setTimeWindow(newWindow);
            potentialviewer->setTimeWindow(newWindow);
        }
		
        void setInputMinY(float minInputY)
        {
            inputviewer->setMinY(minInputY);
        }
		
        void setOutputMinY(float minOutputY)
        {
            outputviewer->setMinY(minOutputY);
        }
		
    protected:
		
		// ----- IMPLEMENTATION VARIABLES -----
        std::unique_ptr<QApplication>          app;
        std::unique_ptr<QQmlApplicationEngine> engine;
        QSurfaceFormat                         format;
        Network                                network;
        InputViewer*                           inputviewer;
        OutputViewer*                          outputviewer;
        PotentialViewer*                       potentialviewer;
    };
}
