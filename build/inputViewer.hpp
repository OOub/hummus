/*
 * inputViewer.hpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 21/11/2017
 *
 * Information: The InputViewer class is used by the Display class to show the input neurons.
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

Q_DECLARE_METATYPE(QtCharts::QAbstractSeries *)
Q_DECLARE_METATYPE(QtCharts::QValueAxis *)

namespace baal
{
    class InputViewer : public QObject
    {
    Q_OBJECT
    public:
		
    	// ----- OCNSTRUCTOR AND DESTRUCTOR -----
        InputViewer(QObject *parent = 0) :
            QObject(parent),
            timeWindow(100),
            openGL(false),
            isClosed(false),
            maxX(1),
            minY(0),
            maxY(1)
    
        {
            qRegisterMetaType<QtCharts::QAbstractSeries*>();
            qRegisterMetaType<QtCharts::QValueAxis*>();
            atomicGuard.clear(std::memory_order_release);
        }
        
        virtual ~InputViewer(){}
		
    	// ----- PUBLIC INPUTVIEWER METHODS -----
        void handleData(float timestamp, projection* p, bool spiked, bool empty, Network* network, Neuron* postNeuron)
        {
        	if (!empty)
        	{
				if (p->isInitial && spiked)
				{
					while (atomicGuard.test_and_set(std::memory_order_acquire)) {}
					if (!isClosed)
					{
						points.append(QPointF(timestamp, p->postNeuron->getNeuronID()));
						maxY = std::max(static_cast<float>(maxY), static_cast<float>(p->postNeuron->getNeuronID()));
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
		
        void setMinY(float y)
        {
            minY = y;
        }
		
		void useHardwareAcceleration(bool accelerate)
        {
            openGL = accelerate;
        }
        
    Q_SIGNALS:
    public slots:
		
    	// ----- QT-RELATED METHODS -----
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
        int                   minY;
        int                   maxY;
        std::atomic_flag      atomicGuard;
    };
}
