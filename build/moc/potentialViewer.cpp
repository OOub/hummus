/****************************************************************************
** Meta object code from reading C++ file 'potentialViewer.hpp'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.9.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../source/potentialViewer.hpp"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'potentialViewer.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.9.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_baal__PotentialViewer_t {
    QByteArrayData data[11];
    char stringdata0[136];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_baal__PotentialViewer_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_baal__PotentialViewer_t qt_meta_stringdata_baal__PotentialViewer = {
    {
QT_MOC_LITERAL(0, 0, 21), // "baal::PotentialViewer"
QT_MOC_LITERAL(1, 22, 19), // "changeTrackedNeuron"
QT_MOC_LITERAL(2, 42, 0), // ""
QT_MOC_LITERAL(3, 43, 9), // "newNeuron"
QT_MOC_LITERAL(4, 53, 7), // "disable"
QT_MOC_LITERAL(5, 61, 6), // "update"
QT_MOC_LITERAL(6, 68, 21), // "QtCharts::QValueAxis*"
QT_MOC_LITERAL(7, 90, 5), // "axisX"
QT_MOC_LITERAL(8, 96, 5), // "axisY"
QT_MOC_LITERAL(9, 102, 26), // "QtCharts::QAbstractSeries*"
QT_MOC_LITERAL(10, 129, 6) // "series"

    },
    "baal::PotentialViewer\0changeTrackedNeuron\0"
    "\0newNeuron\0disable\0update\0"
    "QtCharts::QValueAxis*\0axisX\0axisY\0"
    "QtCharts::QAbstractSeries*\0series"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_baal__PotentialViewer[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   29,    2, 0x0a /* Public */,
       4,    0,   32,    2, 0x0a /* Public */,
       5,    3,   33,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 6, 0x80000000 | 6, 0x80000000 | 9,    7,    8,   10,

       0        // eod
};

void baal::PotentialViewer::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        PotentialViewer *_t = static_cast<PotentialViewer *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->changeTrackedNeuron((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->disable(); break;
        case 2: _t->update((*reinterpret_cast< QtCharts::QValueAxis*(*)>(_a[1])),(*reinterpret_cast< QtCharts::QValueAxis*(*)>(_a[2])),(*reinterpret_cast< QtCharts::QAbstractSeries*(*)>(_a[3]))); break;
        default: ;
        }
    }
}

const QMetaObject baal::PotentialViewer::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_baal__PotentialViewer.data,
      qt_meta_data_baal__PotentialViewer,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *baal::PotentialViewer::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *baal::PotentialViewer::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_baal__PotentialViewer.stringdata0))
        return static_cast<void*>(const_cast< PotentialViewer*>(this));
    return QObject::qt_metacast(_clname);
}

int baal::PotentialViewer::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 3)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 3;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
