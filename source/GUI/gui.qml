R""(

/*
 * gui.qml
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 16/01/2018
 *
 * Information: QML file that defines the GUI.
 */

import QtQuick 2.1
import QtQuick.Controls 2.1
import QtQuick.Controls 1.4 as OldCtrl
import QtQuick.Layouts 1.1
import QtQuick.Window 2.1
import QtCharts 2.1

import InputViewer 1.0
import OutputViewer 1.0
import DynamicsViewer 1.0

ApplicationWindow {
	id: mainWindow
	title: qsTr("Hummus")
	height: 1000
	width: 920
	minimumHeight: 650
	minimumWidth: 450
	property int refresh: 200
	property int a: 0
	property int b: 1
	property int c: 2
	property int xScaleZoom: 0
	property int yScaleZoom: 0
	property bool pp: true
	property bool timer: true
	visible: true
	color: "#FFFFFF"
	flags: Qt.Window | Qt.WindowFullscreenButtonHint

	ColumnLayout {
		id: mainGrid
		anchors.fill: parent

		Rectangle {
			id: menu
    	color: "#FFFFFF"
    	Layout.alignment: Qt.AlignCenter
			Layout.topMargin: 5
			Layout.leftMargin: 5
			Layout.rightMargin: 5
			Layout.minimumWidth: mainGrid.width-10
			Layout.minimumHeight: 27
    	radius: 2

			RoundButton {
				id: play
				text: qsTr("\u25B6")
				anchors.centerIn: parent

				contentItem: Text {
					text: play.text
					color: play.down ? "#000000" : "#363636"
					horizontalAlignment: Text.AlignHCenter
        			verticalAlignment: Text.AlignVCenter
					elide: Text.ElideRight
				}

				background: Rectangle {
					color: play.down ? "#bdbebf" : "#FFFFFF"
					implicitWidth: 35
					implicitHeight: 35
					border.width: 1
					border.color: "#bdbebf"
					radius: 17.5
				}

				onClicked: {
					if (pp == true) {
						pp = false
						play.text = qsTr("\u2759\u2759")
					}
					else {
						pp = true
						play.text = qsTr("\u25B6")
					}
				}
			}

		}

/// INPUT VIEWER

		Rectangle {
			color: "#bdbebf"
			Layout.minimumWidth: mainGrid.width
			Layout.minimumHeight: 1
			Layout.alignment: Qt.AlignCenter
		}

		Rectangle {
			id: inputRec
			color: '#FFFFFF'
			Layout.alignment: Qt.AlignCenter
			Layout.topMargin: 5
			Layout.leftMargin: 5
			Layout.rightMargin: 5
			Layout.minimumWidth: mainGrid.width-10
			Layout.minimumHeight: mainGrid.height/3-19
			radius: 2

			Text {
				id: sublayerLegend1
				text: "sublayers"
			}

			OldCtrl.SpinBox {
				id: sublayerbox1
				minimumValue: 0
				maximumValue: inputSublayer
				anchors.left: sublayerLegend1.right
				anchors.leftMargin: 5

				onEditingFinished: {
					inputViewer.change_sublayer(value)
				}
			}

			Rectangle {
				id: inputInnerRec
				color: '#FFFFFF'
				width: parent.width
        height: parent.height - 20
				anchors.top: inputRec.top
				anchors.topMargin: 20

				ChartView {
					id: inputChart
					title: "Input Neurons"
					titleFont : Qt.font({bold: true})
					anchors.fill: parent
					antialiasing: true
					backgroundColor: '#FFFFFF'
					legend.visible: false
					dropShadowEnabled: false

					ValueAxis {
						id:inputX
						tickCount: inputInnerRec.width/75
						titleText: "Time (ms)"
						labelsFont:Qt.font({pointSize: 11})
					}

					ValueAxis {
						id:inputY
						tickCount: inputInnerRec.height/50
						titleText: "Input Neurons"
						labelsFont:Qt.font({pointSize: 11})
					}

					ScatterSeries {
						id: input
						name: "Input Neurons"
						markerSize: 5
						markerShape: ScatterSeries.MarkerShapeCircle
						axisX: inputX
						axisY: inputY
						borderColor: 'transparent'
					}

					Rectangle{
					    id: recZoom
					    border.color: "steelblue"
					    border.width: 1
					    color: "steelblue"
					    opacity: 0.3
					    visible: false
					    transform: Scale { origin.x: 0; origin.y: 0; xScale: xScaleZoom; yScale: yScaleZoom}
					}

					MouseArea {
					    anchors.fill: parent
					    hoverEnabled: true
					    onPressed: {
					        recZoom.x = mouseX;
					        recZoom.y = mouseY;
					        recZoom.visible = true;
					    }
					    onMouseXChanged: {
					        if (mouseX - recZoom.x >= 0) {
					            xScaleZoom = 1;
					            recZoom.width = mouseX - recZoom.x;
					        } else {
					            xScaleZoom = -1;
					            recZoom.width = recZoom.x - mouseX;
					        }
					    }
					    onMouseYChanged: {
					        if (mouseY - recZoom.y >= 0) {
					            yScaleZoom = 1;
					            recZoom.height = mouseY - recZoom.y;
					        } else {
					            yScaleZoom = -1;
					            recZoom.height = recZoom.y - mouseY;
					        }
					    }
					    onReleased: {
					        var x = (mouseX >= recZoom.x) ? recZoom.x : mouseX
					        var y = (mouseY >= recZoom.y) ? recZoom.y : mouseY
					        inputChart.zoomIn(Qt.rect(x, y, recZoom.width, recZoom.height));
					        recZoom.visible = false;
					    }
					    onDoubleClicked: inputChart.zoomReset();
					}

					Timer {
						id: refreshTimer
						interval: refresh
						running: pp
						repeat: timer
						onTriggered: {
							inputViewer.update(inputX, inputY, inputChart.series(0));
						}
					}

					InputViewer {
						objectName: "inputViewer"
						id: inputViewer
					}
				}
			}
		}

/// OUTPUT VIEWER
		Rectangle {
			color: "#bdbebf"
			Layout.minimumWidth: mainGrid.width
			Layout.minimumHeight: 1
			Layout.alignment: Qt.AlignCenter
		}

		Rectangle {
			id: outputRec
			color: '#FFFFFF'
			Layout.alignment: Qt.AlignCenter
			Layout.leftMargin: 5
			Layout.rightMargin: 5
			Layout.minimumWidth: mainGrid.width-10
			Layout.minimumHeight: mainGrid.height/3-19
			radius: 2

			Text {
				id: layerLegend
				text: "layer"
			}

			OldCtrl.SpinBox {
				id: layerbox
				minimumValue: 1
				maximumValue: layers
				anchors.left: layerLegend.right
				anchors.leftMargin: 5
				onEditingFinished: {
					outputViewer.change_layer(value)
				}
			}

			Text {
				id: sublayerLegend
				text: "sublayers"
				anchors.left: layerbox.right
				anchors.leftMargin: 20
			}

			OldCtrl.SpinBox {
				id: sublayerbox
				minimumValue: 0
				maximumValue: sublayers
				anchors.left: sublayerLegend.right
				anchors.leftMargin: 5
				onEditingFinished: {
					outputViewer.change_sublayer(value)
				}
			}

			Rectangle {
				id: outputInnerRec
				color: '#FFFFFF'
				width: parent.width
        height: parent.height - 20
				anchors.top: outputRec.top
				anchors.topMargin: 20

				ChartView {
					id: outputChart
					title: "Downstream Neurons"
					titleFont : Qt.font({bold: true})
					anchors.fill: parent
					antialiasing: true
					backgroundColor: '#FFFFFF'
					legend.visible: false
					dropShadowEnabled: false

					ValueAxis {
						id:outputX
						tickCount: outputInnerRec.width/75
						titleText: "Time (ms)"
						labelsFont:Qt.font({pointSize: 11})
					}

					ValueAxis {
						id:outputY
						tickCount: outputInnerRec.height/50
						titleText: "Downstream Neurons"
						labelsFont:Qt.font({pointSize: 11})

					}

					ScatterSeries {
						id: output
						name: "Output Neurons"
						markerSize: 5
						markerShape: ScatterSeries.MarkerShapeCircle
						axisX: outputX
						axisY: outputY
						borderColor: 'transparent'
					}

					Rectangle {
					    id: recZoom2
					    border.color: "steelblue"
					    border.width: 1
					    color: "steelblue"
					    opacity: 0.3
					    visible: false
					    transform: Scale { origin.x: 0; origin.y: 0; xScale: xScaleZoom; yScale: yScaleZoom}
					}

					MouseArea {
					    anchors.fill: parent
					    hoverEnabled: true
					    onPressed: {
					        recZoom2.x = mouseX;
					        recZoom2.y = mouseY;
					        recZoom2.visible = true;
					    }
					    onMouseXChanged: {
					        if (mouseX - recZoom2.x >= 0) {
					            xScaleZoom = 1;
					            recZoom2.width = mouseX - recZoom2.x;
					        } else {
					            xScaleZoom = -1;
					            recZoom2.width = recZoom2.x - mouseX;
					        }
					    }
					    onMouseYChanged: {
					        if (mouseY - recZoom2.y >= 0) {
					            yScaleZoom = 1;
					            recZoom2.height = mouseY - recZoom2.y;
					        } else {
					            yScaleZoom = -1;
					            recZoom2.height = recZoom2.y - mouseY;
					        }
					    }
					    onReleased: {
					        var x = (mouseX >= recZoom2.x) ? recZoom2.x : mouseX
					        var y = (mouseY >= recZoom2.y) ? recZoom2.y : mouseY
					        outputChart.zoomIn(Qt.rect(x, y, recZoom2.width, recZoom2.height));
					        recZoom2.visible = false;
					    }
					    onDoubleClicked: outputChart.zoomReset();
					}

					Timer {
						id: refreshTimer2
						interval: refresh
						running: pp
						repeat: timer
						onTriggered: {
							outputViewer.update(outputX, outputY, outputChart.series(0));
						}
					}

					OutputViewer {
						objectName: "outputViewer"
						id: outputViewer
					}

				}
			}
		}

/// POTENTIAL VIEWER

		Rectangle {
			color: "#bdbebf"
			Layout.minimumWidth: mainGrid.width
			Layout.minimumHeight: 1
			Layout.alignment: Qt.AlignCenter
		}

		Rectangle {
			id: potentialRec
			color: '#FFFFFF'
			Layout.alignment: Qt.AlignCenter
			Layout.bottomMargin: 5
			Layout.leftMargin: 5
			Layout.rightMargin: 5
			Layout.minimumWidth: mainGrid.width-10
			Layout.minimumHeight: mainGrid.height/3-19
			radius: 2

			Text {
				id: neuronLegend
				text: "neuron"
			}

			OldCtrl.SpinBox {
				id: spinbox
				maximumValue: numberOfNeurons
				anchors.left: neuronLegend.right
				anchors.leftMargin: 5
				onEditingFinished: {
					dynamicsViewer.change_tracked_neuron(value)
				}
			}

			Rectangle {
				id: potentialInnerRec
				color: '#FFFFFF'
				width: parent.width
        height: parent.height - 20
				anchors.top: potentialRec.top
				anchors.topMargin: 20

				ChartView {
					id: membraneChart
					title: "Neuron Dynamics"
					titleFont : Qt.font({bold: true})
					anchors.fill: parent
					antialiasing: true
					backgroundColor: '#FFFFFF'
					legend.visible: false
					dropShadowEnabled: false

					ValueAxis {
						id:mX
						tickCount: potentialInnerRec.width/75
						titleText: "Time (ms)"
						labelsFont:Qt.font({pointSize: 11})
					}

					ValueAxis {
						id:mY
						tickCount: potentialInnerRec.height/50
						titleText: "Membrane Potential (mV)"
						labelsFont:Qt.font({pointSize: 11})
					}

					ValueAxis {
						id:mY_right
						tickCount: potentialInnerRec.height/50
						titleText: "Injected Current (A)"
						labelsFont:Qt.font({pointSize: 11})
					}

					LineSeries {
						id: membranePotential
						name: "Neuron Dynamics"
						axisX: mX
						axisY: mY
					}

					LineSeries {
						id: threshold
						axisX: mX
						axisY: mY
						color: "#ED6A56"
					}

					LineSeries {
						id: injectedCurrent
						axisX: mX
						axisYRight: mY_right
					}

	        Rectangle {
					    id: recZoom3
					    border.color: "steelblue"
					    border.width: 1
					    color: "steelblue"
					    opacity: 0.3
					    visible: false
					    transform: Scale { origin.x: 0; origin.y: 0; xScale: xScaleZoom; yScale: yScaleZoom}
					}

					MouseArea {
					    anchors.fill: parent
					    hoverEnabled: true
					    onPressed: {
					        recZoom3.x = mouseX;
					        recZoom3.y = mouseY;
					        recZoom3.visible = true;
					    }
					    onMouseXChanged: {
					        if (mouseX - recZoom3.x >= 0) {
					            xScaleZoom = 1;
					            recZoom3.width = mouseX - recZoom3.x;
					        } else {
					            xScaleZoom = -1;
					            recZoom3.width = recZoom3.x - mouseX;
					        }
					    }
					    onMouseYChanged: {
					        if (mouseY - recZoom3.y >= 0) {
					            yScaleZoom = 1;
					            recZoom3.height = mouseY - recZoom3.y;
					        } else {
					            yScaleZoom = -1;
					            recZoom3.height = recZoom3.y - mouseY;
					        }
					    }
					    onReleased: {
					        var x = (mouseX >= recZoom3.x) ? recZoom3.x : mouseX
					        var y = (mouseY >= recZoom3.y) ? recZoom3.y : mouseY
					        membraneChart.zoomIn(Qt.rect(x, y, recZoom3.width, recZoom3.height));
					        recZoom3.visible = false;
					    }
					    onDoubleClicked: membraneChart.zoomReset();
					}

					Timer {
						id: refreshTimer3
						interval: refresh
						running: pp
						repeat: timer
						onTriggered: {
							dynamicsViewer.update(mX, mY, membraneChart.series(0),a);
							dynamicsViewer.update(mX, mY, membraneChart.series(1),b);

							if (displayCurrents == true) {
								dynamicsViewer.update(mX, mY_right, membraneChart.series(2),c);
							}
						}
					}

					DynamicsViewer {
						objectName: "dynamicsViewer"
						id: dynamicsViewer
					}
				}
			}
		}
	}
	onClosing: {
		inputViewer.disable();
    	outputViewer.disable();
    	dynamicsViewer.disable();
	}
}
)""
