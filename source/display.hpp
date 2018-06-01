/*
 * display.hpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/05/2018
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
#include <QQmlContext>

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
        Display(std::vector<NetworkDelegate*> nd = {})
        {
        	static int argc = 1;
			static char* argv[1] = {NULL};
			
			app.reset(new QApplication(argc, argv));
			
            nd.push_back(this);
            network = Network(nd);
			
            qmlRegisterType<InputViewer>("InputViewer", 1, 0, "InputViewer");
            qmlRegisterType<OutputViewer>("OutputViewer", 1, 0, "OutputViewer");
            qmlRegisterType<PotentialViewer>("PotentialViewer", 1, 0, "PotentialViewer");
			
			engine = new QQmlApplicationEngine();
			engine->rootContext()->setContextProperty("layers", 1);
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
            
            inputviewer = window->findChild<InputViewer*>("inputViewer");
            outputviewer = window->findChild<OutputViewer*>("outputViewer");
            potentialviewer = window->findChild<PotentialViewer*>("potentialViewer");
        }
		
    	// ----- PUBLIC DISPLAY METHODS -----
		Mode getMode() const override
		{
			return NetworkDelegate::Mode::display;
		}
		
        void getArrivingSpike(double timestamp, projection* p, bool spiked, bool empty, Network* network, Neuron* postNeuron) override
        {
            inputviewer->handleData(timestamp, p, spiked, empty, network, postNeuron);
            outputviewer->handleData(timestamp, p, spiked, empty, network, postNeuron);
            potentialviewer->handleData(timestamp, p, spiked, empty, network, postNeuron);
        }
		
		int run(double _runtime, float _timestep)
        {	
            auto maxLayerNumber = std::max_element(network.getNeuronPopulations().begin(), network.getNeuronPopulations().end(), [](const receptiveField& one, const receptiveField& two){return one.layerID < two.layerID;});
            
        	engine->rootContext()->setContextProperty("layers", network.getNeuronPopulations()[std::distance(std::begin(network.getNeuronPopulations()),maxLayerNumber)].layerID);
        	
            std::thread spikeManager([this, _runtime, _timestep]{
                network.run(_runtime, _timestep);
            });
            int errorCode = app->exec();
            spikeManager.join();
            return errorCode;
        }
		
		// ----- NETWORK CLASS WRAPPERS -----
		void addNeurons(int16_t _layerID, learningMode _learningType=noLearning, int _numberOfNeurons=1, float _decayCurrent=10, float _decayPotential=20, int _refractoryPeriod=3, float _eligibilityDecay=100, float _alpha=1, float _lambda=1, float _threshold = -50, float  _restingPotential=-70, float _resetPotential=-70, float _inputResistance=50e9, float _externalCurrent=1, int16_t _rfID=0)
		{   
			network.addNeurons(_layerID, _learningType,_numberOfNeurons, _decayCurrent,_decayPotential,_refractoryPeriod,_eligibilityDecay,_alpha, _lambda,_threshold,_restingPotential,_resetPotential,_inputResistance,_externalCurrent,_rfID);
		}
		
		void addReceptiveFields(int gridSize, int rfNumber, int16_t _layerID, learningMode _learningType=noLearning, int _numberOfNeurons=-1, float _decayCurrent=10, float _decayPotential=20, int _refractoryPeriod=3, float _eligibilityDecay=100, float _alpha=1, float _lambda=1, float _threshold = -50, float  _restingPotential=-70, float _resetPotential=-70, float _inputResistance=50e9, float _externalCurrent=1)
		{
		    network.addReceptiveFields(gridSize,rfNumber,_layerID,_learningType,_numberOfNeurons,_decayCurrent,_decayPotential,_refractoryPeriod,_eligibilityDecay,_alpha,_lambda,_threshold,_restingPotential,_resetPotential,_inputResistance,_externalCurrent);
		}
		
		void allToallConnectivity(std::vector<Neuron>* presynapticLayer, std::vector<Neuron>* postsynapticLayer, bool randomWeights, float _weight, bool randomDelays, int _delay=0, bool redundantConnections=true)
		{
			network.allToallConnectivity(presynapticLayer,postsynapticLayer,randomWeights,_weight,randomDelays,_delay,redundantConnections);
		}
		
		void injectSpike(spike s)
        {
            network.injectSpike(s);
        }
		
		std::vector<receptiveField>& getNeuronPopulations()
		{
			return network.getNeuronPopulations();
		}
		
		template<typename input>
		void injectTeacher(std::vector<input>* _teacher)
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
		
		void trackLayer(int layerToTrack)
		{
			outputviewer->changeLayer(layerToTrack);
		}
		
        void trackNeuron(int neuronToTrack)
        {
            potentialviewer->trackNeuron(neuronToTrack);
        }
		
		void setTimeWindow(double newWindow)
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
        QQmlApplicationEngine*                 engine; // this should be a unique pointer but there's a double free error because of the usage of qml and the engine
        Network                                network;
        InputViewer*                           inputviewer;
        OutputViewer*                          outputviewer;
        PotentialViewer*                       potentialviewer;
    };
}
