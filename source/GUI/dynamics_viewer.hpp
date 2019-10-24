/*
 * dynamicsViewer.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 17/01/2018
 *
 * Information: The DynamicsViewer class is used by the Display class to show a specified neuron's potential and current. Depends on Qt5
 */

#pragma once

#include <algorithm>
#include <atomic>

// QT5 and QT Charts Dependency
#include <QtCore/QObject>

#include <QtQuick/QQuickView>
#include <QtQuick/QQuickItem>
#include <QtQuick/QtQuick>

#include <QtCharts/QAbstractSeries>
#include <QtCharts/QValueAxis>
#include <QtCharts/QAreaSeries>
#include <QtCharts/QXYSeries>
#include <QtCharts/QChart>

namespace hummus {
    
    class DynamicsViewer : public QObject {
        
    Q_OBJECT
    public:
		
        // ----- CONSTRUCTOR AND DESTRUCTOR
        DynamicsViewer(QObject *parent = 0) :
                QObject(parent),
                is_closed(false),
                openGL(true),
                time_window(100),
                max_x(0),
                neuron_tracker(-1),
                current_plot(false),
                y_n_lim(-70),
                y_p_lim(-50),
                yr_n_lim(0),
                yr_p_lim(1) {
            atomic_guard.clear(std::memory_order_release);
            min_y = y_n_lim;
            max_y = y_p_lim;
            min_y_right = yr_n_lim;
            max_y_right = yr_p_lim;
        }
        
        virtual ~DynamicsViewer(){}
		
    	// ----- PUBLIC DYNAMICSVIEWER METHODS -----
		void handle_data(double timestamp, int postsynapticNeuronID, double _potential, double _current, double _threshold) {
			if (postsynapticNeuronID == neuron_tracker) {
                while (atomic_guard.test_and_set(std::memory_order_acquire)) {}
				if (!is_closed) {
                    // saving data points to plot
                    if (current_plot) {
                        current_points.append(QPointF(timestamp,_current));
                    }
                    
					points.append(QPointF(timestamp, _potential));
					thres_points.append(QPointF(timestamp, _threshold));
                    // membrane potential axis
					min_y = std::min(min_y, _potential);
					max_y = std::max(max_y, _potential);
                    // injected current axis
                    min_y_right = std::min(min_y_right, _current);
                    max_y_right = std::max(max_y_right, _current);
				} else {
					points.clear();
                    thres_points.clear();
                    current_points.clear();
				}
				atomic_guard.clear(std::memory_order_release);
			}
            
            // time axis
            max_x = timestamp;
		}
		
		// ----- SETTERS -----
        void set_potential_limits(double _y_n_lim, double _y_p_lim) {
            y_n_lim = _y_n_lim;
            y_p_lim = _y_p_lim;
            min_y = _y_n_lim;
            max_y = _y_p_lim;
        }
        
        void set_current_limits(double _yr_n_lim, double _yr_p_lim) {
            yr_n_lim = _yr_n_lim;
            yr_p_lim = _yr_p_lim;
            min_y_right = _yr_n_lim;
            max_y_right = _yr_p_lim;
        }
        
		void set_time_window(double new_window) {
            time_window = new_window;
        }
		
		void hardware_acceleration(bool accelerate) {
            openGL = accelerate;
        }
		
        void track_neuron(int neuronToTrack) {
            neuron_tracker = neuronToTrack;
        }
		
        void plot_currents(bool _current_plot) {
            current_plot = _current_plot;
        }
        
        void reset() {
            points.clear();
            thres_points.clear();
            current_points.clear();
        }
        
    Q_SIGNALS:
    public Q_SLOTS:
		
        // ----- QT-RELATED METHODS -----
        void change_tracked_neuron(int new_neuron) {
            if (neuron_tracker != new_neuron) {
                neuron_tracker = new_neuron;
                min_y = y_n_lim;
                max_y = y_p_lim;
                min_y_right = yr_n_lim;
                max_y_right = yr_p_lim;
            }
        }
    
        void disable() {
            while (atomic_guard.test_and_set(std::memory_order_acquire)) {}
            is_closed = true;
            atomic_guard.clear(std::memory_order_release);
        }
    
        void update(QtCharts::QValueAxis *axisX, QtCharts::QValueAxis *axisY, QtCharts::QAbstractSeries *series,  int seriesType) {
            if (!is_closed) {
                if (series) {
                    while (atomic_guard.test_and_set(std::memory_order_acquire)) {}
                    if (openGL) {
                        series->setUseOpenGL(true);
                    }
					
                    switch (seriesType) {
                        case 0:
                            axisX->setRange(max_x - time_window, max_x+1);
                            if (!points.isEmpty()) {
                                auto firstToKeep = std::upper_bound(points.begin(), points.end(), points.back().x() - time_window, [](double timestamp, const QPointF& point) {
                                    return timestamp < point.x();
                                });
                                points.remove(0, static_cast<int>(std::distance(points.begin(), firstToKeep)));
                    
                                static_cast<QtCharts::QXYSeries *>(series)->replace(points);
                                axisY->setRange(min_y,max_y);
                            }
                            break;
                        case 1:
                            if (!thres_points.isEmpty()) {
                                auto firstToKeep = std::upper_bound(thres_points.begin(), thres_points.end(), thres_points.back().x() - time_window, [](double timestamp, const QPointF& thres_points) {
                                    return timestamp < thres_points.x();
                                });
                                thres_points.remove(0, static_cast<int>(std::distance(thres_points.begin(), firstToKeep)));
                    
                                static_cast<QtCharts::QXYSeries *>(series)->replace(thres_points);
                            }
                            break;
                        case 2:
                            if (current_plot) {
                                if (!current_points.isEmpty()) {
                                    auto firstToKeep = std::upper_bound(current_points.begin(), current_points.end(), current_points.back().x() - time_window, [](double timestamp, const QPointF& current_points) {
                                        return timestamp < current_points.x();
                                    });
                                    current_points.remove(0, static_cast<int>(std::distance(current_points.begin(), firstToKeep)));
                                    
                                    static_cast<QtCharts::QXYSeries *>(series)->replace(current_points);
                                    axisY->setRange(min_y_right,max_y_right);
                                }
                                break;
                            }
                    }
					
                    atomic_guard.clear(std::memory_order_release);
                }
            }
        }
        
    protected:
		
        // ----- IMPLEMENTATION VARIABLES -----
        bool                  is_closed;
        bool                  openGL;
        double                time_window;
        QVector<QPointF>      points;
        QVector<QPointF>      thres_points;
        QVector<QPointF>      current_points;
        double                max_x;
        double                min_y;
        double                max_y;
        double                min_y_right;
        double                max_y_right;
        std::atomic_flag      atomic_guard;
        int                   neuron_tracker;
        bool                  current_plot;
        double                y_n_lim;
        double                y_p_lim;
        double                yr_n_lim;
        double                yr_p_lim;
    };
}
