/*
 * outputViewer.hpp
 * Adonis_t - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 02/03/2018
 *
 * Information: The OutputViewer class is used by the Display class to show the output neurons. Depends on Qt5
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
#include <QtWidgets/QSpinBox>

#include "network.hpp"

namespace adonis_t
{
    class OutputViewer : public QObject
    {
    Q_OBJECT
    public:
		
    	// ----- CONSTRUCTOR AND DESTRUCTOR -----
        OutputViewer(QObject *parent = 0) :
            QObject(parent),
            timeWindow(100),
            openGL(false),
            isClosed(false),
            input(0),
            minY(0),
            maxY(1),
            layerTracker(1)
    
        {
            atomicGuard.clear(std::memory_order_release);
        }
        
        virtual ~OutputViewer(){}
		
    	// ----- PUBLIC OUTPUTVIEWER METHODS -----
        void handleData(double timestamp, projection* p, bool spiked, bool empty, Network* network, Neuron* postNeuron)
        {
        	if (!empty)
        	{
				if (p->postNeuron->getLayerID() == layerTracker && spiked)
				{
					while (atomicGuard.test_and_set(std::memory_order_acquire)) {}
					if (!isClosed)
					{
						points.append(QPointF(timestamp, p->postNeuron->getNeuronID()));
						maxY = std::max(static_cast<float>(maxY), static_cast<float>(p->postNeuron->getNeuronID()));
						minY = yLookupTable[layerTracker-1];
					}
					else
					{
						points.clear();
					}
					atomicGuard.clear(std::memory_order_release);
				}
            }
            input = timestamp;
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
		
		void setYLookup(std::vector<int> newLookup)
		{
		    yLookupTable = newLookup;
		}
		
    Q_SIGNALS:
    public slots:
		
    	// ----- QT-RELATED METHODS -----
		void changeLayer(int newLayer)
		{
		    if (layerTracker != newLayer)
		    {
			    layerTracker = newLayer;
			    minY = 0;
			    maxY = 1;
			}
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
                    axisX->setRange(input - timeWindow, input+1);
                    if (!points.isEmpty())
                    {
                        auto firstToKeep = std::upper_bound(points.begin(), points.end(), points.back().x() - timeWindow, [](double timestamp, const QPointF& point) {
                            return timestamp < point.x();
                        });
                        points.remove(0, static_cast<int>(std::distance(points.begin(), firstToKeep)));
            
                        static_cast<QtCharts::QXYSeries *>(series)->replace(points);
                        axisY->setRange(minY-1,maxY+1);
                    }
                }
                atomicGuard.clear(std::memory_order_release);
            }
        }
		
    protected:
		
    	// ----- IMPLEMENTATION VARIABLES -----
        bool                  openGL;
        bool                  isClosed;
        double                timeWindow;
        QVector<QPointF>      points;
        float                 input;
        int                   minY;
        int                   maxY;
        std::atomic_flag      atomicGuard;
        int                   layerTracker;
        std::vector<int>      yLookupTable;
    };
}
