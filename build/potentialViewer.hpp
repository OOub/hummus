/*
 * potentialViewer.hpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 21/11/2017
 *
 * Information: The PotentialViewer class is used by the Display class to show a specified neuron's approximate potential. It is only an approximation because this GUI element works in an even-based fashion.
 */

#pragma once

#include <algorithm>
#include <atomic>

// QT5 and QT Charts Dependency
#include <QtCore/QObject>
#include <QtCore/QtMath>

#include <QtQuick/QQuickView>
#include <QtQuick/QQuickItem>
#include <QtQuick/QtQuick>

#include <QtCharts/QAbstractSeries>
#include <QtCharts/QValueAxis>
#include <QtCharts/QAreaSeries>
#include <QtCharts/QXYSeries>
#include <QtCharts/QChart>

#include "network.hpp"

namespace baal
{
    class PotentialViewer : public QObject
    {
    Q_OBJECT
    public:
		
        // ----- CONSTRUCTOR AND DESTRUCTOR
        PotentialViewer(QObject *parent = 0) :
            QObject(parent),
            timeWindow(100),
            openGL(false),
            isClosed(false),
            maxX(1),
//            minY(-70),
//            maxY(-50),
			minY(20),
            maxY(-70),
            potential(0),
            neuronTracker(-1)
        {
            atomicGuard.clear(std::memory_order_release);
        }
        
        virtual ~PotentialViewer(){}
		
    	// ----- PUBLIC POTENTIALVIEWER METHODS -----
        void handleData(float timestamp, projection* p, bool spiked, bool empty, Network* network, Neuron* postNeuron)
        {
        	if (!empty)
        	{
				if (p->postNeuron->getNeuronID() == neuronTracker)
				{
					while (atomicGuard.test_and_set(std::memory_order_acquire)) {}
					if (!isClosed)
					{
						potential = p->postNeuron->getPotential();
						points.append(QPointF(timestamp, potential));
						minY = std::min(minY, static_cast<float>(potential));
						maxY = std::max(maxY, static_cast<float>(potential));
					}
					else
					{
						points.clear();
					}
					atomicGuard.clear(std::memory_order_release);
				}
			}
			else
			{
				if (postNeuron->getNeuronID() == neuronTracker)
				{
					while (atomicGuard.test_and_set(std::memory_order_acquire)) {}
					if (!isClosed)
					{
						potential = postNeuron->getPotential();
						points.append(QPointF(timestamp, potential));
						minY = std::min(minY, static_cast<float>(potential));
						maxY = std::max(maxY, static_cast<float>(potential));
					}
					else
					{
						points.clear();
					}
					atomicGuard.clear(std::memory_order_release);
				}
			}
			maxX = timestamp;
        }
		
		// ----- SETTERS -----
		void setTimeWindow(float newWindow)
        {
            timeWindow = newWindow;
        }
		
		void useHardwareAcceleration(bool accelerate)
        {
            openGL = accelerate;
        }
		
        void trackNeuron(int neuronToTrack)
        {
            neuronTracker = neuronToTrack;
        }
		
    Q_SIGNALS:
    public slots:
		
        // ----- QT-RELATED METHODS -----
        void changeTrackedNeuron(int newNeuron)
        {
            neuronTracker = newNeuron;
        }
    
        void disable()
        {
            while (atomicGuard.test_and_set(std::memory_order_acquire)) {}
            isClosed = true;
            atomicGuard.clear(std::memory_order_release);
        }
    
        void update(QtCharts::QValueAxis *axisX, QtCharts::QValueAxis *axisY, QtCharts::QAbstractSeries *series)
        {
            if (!isClosed)
            {
                if (series)
                {
                    while (atomicGuard.test_and_set(std::memory_order_acquire)) {}
                    if (openGL)
                    {
                        series->setUseOpenGL(true);
                    }
                    
                    axisX->setRange(maxX - timeWindow, maxX+1);
                    if (!points.isEmpty())
                    {
                        auto firstToKeep = std::upper_bound(points.begin(), points.end(), points.back().x() - timeWindow, [](float timestamp, const QPointF& point) {
                            return timestamp < point.x();
                        });
                        points.remove(0, static_cast<int>(std::distance(points.begin(), firstToKeep)));
            
                        static_cast<QtCharts::QXYSeries *>(series)->replace(points);
                        axisY->setRange(minY-1,maxY+1);
                    }
                    atomicGuard.clear(std::memory_order_release);
                }
            }
        }
        
    protected:
		
        // ----- IMPLEMENTATION VARIABLES -----
        bool                  isClosed;
        bool                  openGL;
        float                 timeWindow;
        QVector<QPointF>      points;
        float                 maxX;
        float                 minY;
        float                 maxY;
        std::atomic_flag      atomicGuard;
        float                 potential;
        int                   neuronTracker;
    };
}
