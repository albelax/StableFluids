/****************************************************************************
** Meta object code from reading C++ file 'GLWindow.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.9.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../include/GLWindow.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'GLWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.9.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_GLWindow_t {
    QByteArrayData data[14];
    char stringdata0[128];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_GLWindow_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_GLWindow_t qt_meta_stringdata_GLWindow = {
    {
QT_MOC_LITERAL(0, 0, 8), // "GLWindow"
QT_MOC_LITERAL(1, 9, 4), // "init"
QT_MOC_LITERAL(2, 14, 0), // ""
QT_MOC_LITERAL(3, 15, 5), // "reset"
QT_MOC_LITERAL(4, 21, 11), // "setTimestep"
QT_MOC_LITERAL(5, 33, 9), // "_timeStep"
QT_MOC_LITERAL(6, 43, 12), // "setDiffusion"
QT_MOC_LITERAL(7, 56, 10), // "_diffusion"
QT_MOC_LITERAL(8, 67, 12), // "setViscosity"
QT_MOC_LITERAL(9, 80, 10), // "_viscosity"
QT_MOC_LITERAL(10, 91, 10), // "setDensity"
QT_MOC_LITERAL(11, 102, 8), // "_density"
QT_MOC_LITERAL(12, 111, 10), // "SaveFrames"
QT_MOC_LITERAL(13, 122, 5) // "_save"

    },
    "GLWindow\0init\0\0reset\0setTimestep\0"
    "_timeStep\0setDiffusion\0_diffusion\0"
    "setViscosity\0_viscosity\0setDensity\0"
    "_density\0SaveFrames\0_save"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_GLWindow[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   49,    2, 0x0a /* Public */,
       3,    0,   50,    2, 0x0a /* Public */,
       4,    1,   51,    2, 0x0a /* Public */,
       6,    1,   54,    2, 0x0a /* Public */,
       8,    1,   57,    2, 0x0a /* Public */,
      10,    1,   60,    2, 0x0a /* Public */,
      12,    1,   63,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Double,    5,
    QMetaType::Void, QMetaType::Double,    7,
    QMetaType::Void, QMetaType::Double,    9,
    QMetaType::Void, QMetaType::Double,   11,
    QMetaType::Void, QMetaType::Bool,   13,

       0        // eod
};

void GLWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        GLWindow *_t = static_cast<GLWindow *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->init(); break;
        case 1: _t->reset(); break;
        case 2: _t->setTimestep((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 3: _t->setDiffusion((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 4: _t->setViscosity((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 5: _t->setDensity((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 6: _t->SaveFrames((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject GLWindow::staticMetaObject = {
    { &QOpenGLWidget::staticMetaObject, qt_meta_stringdata_GLWindow.data,
      qt_meta_data_GLWindow,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *GLWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *GLWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_GLWindow.stringdata0))
        return static_cast<void*>(this);
    return QOpenGLWidget::qt_metacast(_clname);
}

int GLWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QOpenGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 7;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
