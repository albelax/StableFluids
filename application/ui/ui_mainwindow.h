/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.6.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QGridLayout *s_mainWindowGridLayout;
    QSpacerItem *horizontalSpacer_6;
    QSpacerItem *horizontalSpacer_5;
    QSpacerItem *horizontalSpacer_2;
    QSpacerItem *horizontalSpacer;
    QGroupBox *s_drawGB;
    QGridLayout *gridLayout_2;
    QSpacerItem *verticalSpacer;
    QSpacerItem *horizontalSpacer_3;
    QGroupBox *s_transformGB;
    QGridLayout *gridLayout;
    QDoubleSpinBox *Viscosity;
    QLabel *ViscosityLabel;
    QDoubleSpinBox *timestep;
    QLabel *DensityLabel;
    QLabel *TimestepLabel;
    QDoubleSpinBox *Diffusion;
    QLabel *label;
    QPushButton *reset;
    QDoubleSpinBox *Density;
    QLabel *DiffusionLabel;
    QSpacerItem *horizontalSpacer_4;
    QCheckBox *saveFrames;
    QMenuBar *menubar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(800, 600);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QStringLiteral("centralwidget"));
        s_mainWindowGridLayout = new QGridLayout(centralwidget);
        s_mainWindowGridLayout->setObjectName(QStringLiteral("s_mainWindowGridLayout"));
        horizontalSpacer_6 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        s_mainWindowGridLayout->addItem(horizontalSpacer_6, 0, 1, 1, 1);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        s_mainWindowGridLayout->addItem(horizontalSpacer_5, 0, 3, 1, 1);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        s_mainWindowGridLayout->addItem(horizontalSpacer_2, 0, 2, 1, 1);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        s_mainWindowGridLayout->addItem(horizontalSpacer, 0, 4, 1, 1);

        s_drawGB = new QGroupBox(centralwidget);
        s_drawGB->setObjectName(QStringLiteral("s_drawGB"));
        gridLayout_2 = new QGridLayout(s_drawGB);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_2->addItem(verticalSpacer, 9, 1, 1, 1);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_2->addItem(horizontalSpacer_3, 0, 2, 1, 1);


        s_mainWindowGridLayout->addWidget(s_drawGB, 2, 5, 1, 1);

        s_transformGB = new QGroupBox(centralwidget);
        s_transformGB->setObjectName(QStringLiteral("s_transformGB"));
        gridLayout = new QGridLayout(s_transformGB);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        Viscosity = new QDoubleSpinBox(s_transformGB);
        Viscosity->setObjectName(QStringLiteral("Viscosity"));
        Viscosity->setMaximum(10);
        Viscosity->setSingleStep(0.1);

        gridLayout->addWidget(Viscosity, 6, 2, 1, 1);

        ViscosityLabel = new QLabel(s_transformGB);
        ViscosityLabel->setObjectName(QStringLiteral("ViscosityLabel"));

        gridLayout->addWidget(ViscosityLabel, 5, 2, 1, 1);

        timestep = new QDoubleSpinBox(s_transformGB);
        timestep->setObjectName(QStringLiteral("timestep"));
        timestep->setMaximum(10);
        timestep->setSingleStep(0.1);
        timestep->setValue(1);

        gridLayout->addWidget(timestep, 2, 2, 1, 1);

        DensityLabel = new QLabel(s_transformGB);
        DensityLabel->setObjectName(QStringLiteral("DensityLabel"));

        gridLayout->addWidget(DensityLabel, 7, 2, 1, 1);

        TimestepLabel = new QLabel(s_transformGB);
        TimestepLabel->setObjectName(QStringLiteral("TimestepLabel"));

        gridLayout->addWidget(TimestepLabel, 1, 2, 1, 1);

        Diffusion = new QDoubleSpinBox(s_transformGB);
        Diffusion->setObjectName(QStringLiteral("Diffusion"));
        Diffusion->setMaximum(100);
        Diffusion->setSingleStep(0.1);

        gridLayout->addWidget(Diffusion, 4, 2, 1, 1);

        label = new QLabel(s_transformGB);
        label->setObjectName(QStringLiteral("label"));

        gridLayout->addWidget(label, 0, 3, 1, 1);

        reset = new QPushButton(s_transformGB);
        reset->setObjectName(QStringLiteral("reset"));

        gridLayout->addWidget(reset, 0, 2, 1, 1);

        Density = new QDoubleSpinBox(s_transformGB);
        Density->setObjectName(QStringLiteral("Density"));
        Density->setMaximum(1000);
        Density->setSingleStep(10);
        Density->setValue(200);

        gridLayout->addWidget(Density, 8, 2, 1, 1);

        DiffusionLabel = new QLabel(s_transformGB);
        DiffusionLabel->setObjectName(QStringLiteral("DiffusionLabel"));

        gridLayout->addWidget(DiffusionLabel, 3, 2, 1, 1);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_4, 0, 4, 1, 1);

        saveFrames = new QCheckBox(s_transformGB);
        saveFrames->setObjectName(QStringLiteral("saveFrames"));

        gridLayout->addWidget(saveFrames, 9, 2, 1, 1);


        s_mainWindowGridLayout->addWidget(s_transformGB, 0, 5, 1, 1);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QStringLiteral("menubar"));
        menubar->setGeometry(QRect(0, 0, 800, 25));
        MainWindow->setMenuBar(menubar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Stable Fluids", 0));
        s_drawGB->setTitle(QString());
        s_transformGB->setTitle(QString());
        ViscosityLabel->setText(QApplication::translate("MainWindow", "Viscosity", 0));
        DensityLabel->setText(QApplication::translate("MainWindow", "Density", 0));
        TimestepLabel->setText(QApplication::translate("MainWindow", "TimeStep", 0));
        label->setText(QString());
        reset->setText(QApplication::translate("MainWindow", "Reset", 0));
        DiffusionLabel->setText(QApplication::translate("MainWindow", "Diffusion", 0));
        saveFrames->setText(QApplication::translate("MainWindow", "Save Frames", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
