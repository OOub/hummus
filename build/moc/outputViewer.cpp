/****************************************************************************
** Meta object code from reading C++ file 'outputViewer.hpp'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.9.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../source/outputViewer.hpp"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'outputViewer.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.9.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_baal__OutputViewer_t {
    QByteArrayData data[9];
    char stringdata0[103];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_baal__OutputViewer_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_baal__OutputViewer_t qt_meta_stringdata_baal__OutputViewer = {
    {
QT_MOC_LITERAL(0, 0, 18), // "baal::OutputViewer"
QT_MOC_LITERAL(1, 19, 7), // "disable"
QT_MOC_LITERAL(2, 27, 0), // ""
QT_MOC_LITERAL(3, 28, 6), // "update"
QT_MOC_LITERAL(4, 35, 21), // "QtCharts::QValueAxis*"
QT_MOC_LITERAL(5, 57, 5), // "axisX"
QT_MOC_LITERAL(6, 63, 5), // "axisY"
QT_MOC_LITERAL(7, 69, 26), // "QtCharts::QAbstractSeries*"
QT_MOC_LITERAL(8, 96, 6) // "series"

    },
    "baal::OutputViewer\0disable\0\0update\0"
    "QtCharts::QValueAxis*\0axisX\0axisY\0"
    "QtCharts::QAbstractSeries*\0series"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_baal__OutputViewer[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   24,    2, 0x0a /* Public */,
       3,    3,   25,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 4, 0x80000000 | 4, 0x80000000 | 7,    5,    6,    8,

       0        // eod
};

void baal::OutputViewer::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        OutputViewer *_t = static_cast<OutputViewer *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->disable(); break;
        case 1: _t->update((*reinterpret_cast< QtCharts::QValueAxis*(*)>(_a[1])),(*reinterpret_cast< QtCharts::QValueAxis*(*)>(_a[2])),(*reinterpret_cast< QtCharts::QAbstractSeries*(*)>(_a[3]))); break;
        default: ;
        }
    }
}

const QMetaObject baal::OutputViewer::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_baal__OutputViewer.data,
      qt_meta_data_baal__OutputViewer,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *baal::OutputViewer::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *baal::OutputViewer::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_baal__OutputViewer.stringdata0))
        return static_cast<void*>(const_cast< OutputViewer*>(this));
    return QObject::qt_metacast(_clname);
}

int baal::OutputViewer::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 2)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 2;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
