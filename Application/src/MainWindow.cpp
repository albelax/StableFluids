#include "MainWindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
  QMainWindow(parent),
  m_ui(new Ui::MainWindow)
{
  m_ui -> setupUi(this);
  m_ui -> setupUi(this);
  m_gl = new GLWindow(this);
  m_ui -> s_mainWindowGridLayout -> addWidget(m_gl,0,0,3,5);
  connect( m_ui->reset, SIGNAL( clicked(bool)), m_gl, SLOT(reset()));
  connect( m_ui->timestep, SIGNAL(valueChanged( double )), m_gl, SLOT(setTimestep(double)));
  connect( m_ui->Diffusion, SIGNAL(valueChanged( double )), m_gl, SLOT(setDiffusion(double)));
  connect( m_ui->Viscosity, SIGNAL(valueChanged( double )), m_gl, SLOT(setViscosity(double)));
  connect( m_ui->Density, SIGNAL(valueChanged( double )), m_gl, SLOT(setDensity(double)));
  connect( m_ui->saveFrames, SIGNAL(toggled(bool)), m_gl, SLOT(SaveFrames(bool)));
}

MainWindow::~MainWindow()
{
  delete m_ui;
}

//----------------------------------------------------------------------------------------------------------------------

void MainWindow::keyPressEvent(QKeyEvent *_event)
{
  // this method is called every time the main window recives a key event.
  // we then switch on the key value and set the camera in the GLWindow
  switch ( _event->key() )
  {
    case Qt::Key_Escape : QApplication::exit(EXIT_SUCCESS); break;
    default : break;
  }
}

//----------------------------------------------------------------------------------------------------------------------

void MainWindow::mouseMoveEvent(QMouseEvent * _event)
{
  m_gl->mouseMove(_event);
}

//----------------------------------------------------------------------------------------------------------------------

void MainWindow::mousePressEvent(QMouseEvent * _event)
{
  m_gl->mouseClick(_event);
}

//----------------------------------------------------------------------------------------------------------------------

void MainWindow::mouseReleaseEvent(QMouseEvent * _event)
{
  m_gl->mouseClick(_event);
}

//----------------------------------------------------------------------------------------------------------------------
