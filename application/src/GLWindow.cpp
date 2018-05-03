#include "GLWindow.h"

#include <iostream>
#include <QColorDialog>
#include <QGLWidget>
#include <QImage>
#include <QScreen>
#include <unistd.h>
#include "parameters.h"

const std::string address = "../application/"; // if the program is fired from the bin folder

//----------------------------------------------------------------------------------------------------------------------

GLWindow::GLWindow( QWidget *_parent ) : QOpenGLWidget( _parent )
{
  // set this widget to have the initial keyboard focus
  // re-size the widget to that of the parent (in this case the GLFrame passed in on construction)
  this->resize( _parent->size() );
  m_camera.setInitialMousePos(0,0);
  m_camera.setTarget(0.0f, 0.0f, -2.0f);
  m_camera.setEye(0.0f, 0.0f, 0.0f);

  m_image = QPixmap( Common::gridWidth, Common::gridHeight ).toImage();
  m_image.fill(Qt::white);

  m_solverType = solverType::CPU;

  if ( m_solverType == solverType::CPU )
    m_activeSolver = std::unique_ptr<Solver>( &m_solver );
  else if ( m_solverType == solverType::GPU )
    m_activeSolver = std::unique_ptr<Solver>( &m_solverGpu );
  m_activeSolver->activate();

  m_active = true;
}

//----------------------------------------------------------------------------------------------------------------------

void GLWindow::initializeGL()
{
#ifdef linux
  glewExperimental = GL_TRUE;
  glewInit();
#endif

  glEnable( GL_DEPTH_TEST );
  glEnable( GL_MULTISAMPLE );
  glEnable( GL_TEXTURE_2D );
  glClearColor( 0.5f, 0.5f, 0.5f, 1.0f );
  glViewport( 0, 0, devicePixelRatio(), devicePixelRatio() );

  m_meshes[0] = Mesh( address + "models/plane.obj", "plane" );
  m_mesh = & m_meshes[0];

  init();

  m_MV = glm::translate( m_MV, glm::vec3( 0.0f, 0.0f, -2.0f) );
}

//----------------------------------------------------------------------------------------------------------------------

GLWindow::~GLWindow()
{
  m_activeSolver.release();
}

//----------------------------------------------------------------------------------------------------------------------

void GLWindow::mouseMove( QMouseEvent * _event )
{
  if ( m_active )
    m_activeSolver->cleanBuffer();
  m_camera.handleMouseMove( _event->pos().x() * Common::multiplier, _event->pos().y() * Common::multiplier );

  float posx = _event->pos().x() / static_cast<float>(width()) *  Common::gridWidth-1 * Common::multiplier;
  float posy = _event->pos().y() / static_cast<float>(height()) * Common::gridHeight-1 * Common::multiplier;

  int x = static_cast<int>( posx ) > Common::gridWidth-1 ? Common::gridWidth-1 : static_cast<int>(posx);
  int y = static_cast<int>( posy ) > Common::gridHeight-1 ? Common::gridHeight-1 : static_cast<int>(posy);

  if ( x < 1 ) x = 1;
  if ( y < 1 ) y = 1;

  if ( _event->buttons() == Qt::RightButton && m_active )
    m_activeSolver->setVel0(x, y, _event->pos().x() - prevX, _event->pos().y() - prevY );

  else if ( _event->buttons() == Qt::LeftButton && m_active )
    m_activeSolver->setD0(x, y);
  if ( m_active )
    m_activeSolver->addSource();

  update();
}

//----------------------------------------------------------------------------------------------------------------------

void GLWindow::mouseClick(QMouseEvent * _event)
{
  if ( m_active )
    m_activeSolver->cleanBuffer();
  prevX = _event->pos().x();
  prevY = _event->pos().y();

  if ( _event->buttons() == Qt::LeftButton && m_active )
    m_activeSolver->setD0(prevX, prevY);

  update();
}

//----------------------------------------------------------------------------------------------------------------------

void GLWindow::init()
{
  std::string shadersAddress = address + "shaders/";
  m_shader = Shader( "m_shader", shadersAddress + "renderedVert.glsl", shadersAddress + "renderedFrag.glsl" );

  glLinkProgram( m_shader.getShaderProgram() );
  glUseProgram( m_shader.getShaderProgram() );

  glGenVertexArrays( 1, &m_vao );
  glBindVertexArray( m_vao );
  glGenBuffers( 1, &m_vbo );
  glGenBuffers( 1, &m_nbo );
  glGenBuffers( 1, &m_tbo );

  m_mesh->setBufferIndex( 0 );
  m_amountVertexData = m_mesh->getAmountVertexData();

  // load vertices
  glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
  glBufferData( GL_ARRAY_BUFFER, m_amountVertexData * sizeof(float), 0, GL_STATIC_DRAW );
  glBufferSubData( GL_ARRAY_BUFFER, 0, m_mesh->getAmountVertexData() * sizeof(float), &m_mesh->getVertexData() );

  // pass vertices to shader
  GLint pos = glGetAttribLocation( m_shader.getShaderProgram(), "VertexPosition" );
  glEnableVertexAttribArray( pos );
  glVertexAttribPointer( pos, 3, GL_FLOAT, GL_FALSE, 0, 0 );

  // load normals
  glBindBuffer( GL_ARRAY_BUFFER,	m_nbo );
  glBufferData( GL_ARRAY_BUFFER, m_amountVertexData * sizeof(float), 0, GL_STATIC_DRAW );
  glBufferSubData( GL_ARRAY_BUFFER, 0, m_mesh->getAmountVertexData() * sizeof(float), &m_mesh->getNormalsData() );

  // pass normals to shader
  GLint n = glGetAttribLocation( m_shader.getShaderProgram(), "VertexNormal" );
  glEnableVertexAttribArray( n );
  glVertexAttribPointer( n, 3, GL_FLOAT, GL_FALSE, 0, 0 );


  glBindBuffer( GL_ARRAY_BUFFER,	m_tbo );
  glBufferData( GL_ARRAY_BUFFER, m_amountVertexData * sizeof(float), 0, GL_STATIC_DRAW) ;
  glBufferSubData( GL_ARRAY_BUFFER, 0, m_mesh->getAmountVertexData() * sizeof(float), &m_mesh->getUVsData() );

  // pass texture coords to shader
  GLint t = glGetAttribLocation( m_shader.getShaderProgram(), "TexCoord" );
  glEnableVertexAttribArray( t );
  glVertexAttribPointer( t, 2, GL_FLOAT, GL_FALSE, 0, (void*) 0 );


  // link matrices with shader locations
  m_MVAddress = glGetUniformLocation( m_shader.getShaderProgram(), "MV" );
  m_MVPAddress = glGetUniformLocation( m_shader.getShaderProgram(), "MVP" );
  m_NAddress = glGetUniformLocation( m_shader.getShaderProgram(), "N" );

  m_colourTextureAddress = glGetUniformLocation( m_shader.getShaderProgram(), "ColourTexture" );

  addTexture();
  glUniform1i( m_colourTextureAddress, 0 );

  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
  glGenerateMipmap( GL_TEXTURE_2D );
}

//------------------------------------------------------------------------------------------------------------------------------

void GLWindow::paintGL()
{
  glClearColor( 1, 1, 1, 1.0f );
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

  if ( m_active )
  {
    m_activeSolver->animVel();
    m_activeSolver->animDen();
    draw( m_activeSolver->getDens(), m_activeSolver->getRowCell() );
  }

  auto m_glImage = QGLWidget::convertToGLFormat( m_image );
  if(m_glImage.isNull())
    qWarning("IMAGE IS NULL");
  glBindTexture( GL_TEXTURE_2D, m_textures[m_textures.size()-1] );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, m_glImage.width(), m_glImage.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, m_glImage.bits() );

  renderTexture();

  update();
}

//------------------------------------------------------------------------------------------------------------------------------

void GLWindow::renderScene()
{
  glViewport( 0, 0, width()*devicePixelRatio(), height()*devicePixelRatio() ); //fix for retina screens
  glClearColor( 1, 1, 1, 1.0f );
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

  m_camera.update();
  m_projection = glm::perspective( glm::radians( 60.0f ),
                                   static_cast<float>( width() ) / static_cast<float>( height() ), 0.1f, 100.0f );
  m_view = glm::lookAt( glm::vec3( 0.0f, 0.0f, 5.0f ), glm::vec3( 0.0f, 0.0f, 0.0f ), glm::vec3( 0.0f, 1.0f, 0.0f ) );

  m_MVP = m_projection * m_camera.viewMatrix() * m_MV;
  glm::mat3 N = glm::mat3( glm::inverse( glm::transpose( m_MV ) ) );

  glUniformMatrix4fv( m_MVPAddress, 1, GL_FALSE, glm::value_ptr( m_MVP ) );
  glUniformMatrix4fv( m_MVAddress, 1, GL_FALSE, glm::value_ptr( m_MV ) );

  glUniformMatrix3fv( m_NAddress, 1, GL_FALSE, glm::value_ptr( N ) );

  glDrawArrays( GL_TRIANGLES, 0 , ( m_amountVertexData / 3 ) );
}

//------------------------------------------------------------------------------------------------------------------------------

void GLWindow::reset()
{
  if ( m_active )
    m_activeSolver->reset();
}
//------------------------------------------------------------------------------------------------------------------------------

void GLWindow::draw( const real * _density, int _size )
{
  for ( int i = 1; i < m_image.height(); ++i )
  {
    for ( int j = 1; j < m_image.width(); ++j )
    {
      float density = (_density[(j - 1) * _size + (i - 1)] +
          _density[(j - 1) * _size + i] +
          _density[j * _size + (i - 1)] +
          _density[j * _size + i])/4.0f;

      float r = 255 - ( density > 255 ? 255 : density );
      if ( i == 0 || j == 0 || i == m_image.height()-1 || j == m_image.width()-1 )
        r = 255;
      m_image.setPixel(i, j, qRgb(r, r, r) );
    }
  }
}

//------------------------------------------------------------------------------------------------------------------------------

void GLWindow::renderTexture()
{
  glViewport( 0, 0, width(), height() );
  glClearColor( 1, 1, 1, 1.0f );
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

  glBindTexture( GL_TEXTURE_2D, m_textures[0]);

  glUniform1i( m_colourTextureAddress, 0 );

  glActiveTexture( GL_TEXTURE0 );

  glUseProgram( m_shader.getShaderProgram() );

  glBindVertexArray( m_vao );
  glBindBuffer( GL_ARRAY_BUFFER, m_vbo );
  glEnableVertexAttribArray( glGetAttribLocation( m_shader.getShaderProgram(), "VertexPosition" ) );
  glVertexAttribPointer( glGetAttribLocation( m_shader.getShaderProgram(), "VertexPosition" ), 3, GL_FLOAT, GL_FALSE, 0, 0 );

  glBindBuffer( GL_ARRAY_BUFFER, m_tbo );
  glEnableVertexAttribArray( glGetAttribLocation( m_shader.getShaderProgram(), "TexCoord" ) );
  glVertexAttribPointer( glGetAttribLocation( m_shader.getShaderProgram(), "TexCoord" ), 2, GL_FLOAT, GL_FALSE, 0, (void*) 0 );

  glDrawArrays( GL_TRIANGLES, m_mesh->getBufferIndex() / 3, ( m_mesh->getAmountVertexData() / 3 ) );
}

//------------------------------------------------------------------------------------------------------------------------------

void GLWindow::addTexture()
{
  GLuint tmp;
  m_textures.push_back(tmp);
  glActiveTexture( GL_TEXTURE0 + ( m_textures.size() - 1 ) );
  glGenTextures( 1, &m_textures[ m_textures.size() - 1 ] );

  auto m_glImage = QGLWidget::convertToGLFormat( m_image );
  if(m_glImage.isNull())
    qWarning("IMAGE IS NULL");
  glBindTexture( GL_TEXTURE_2D, m_textures[m_textures.size()-1] );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, m_glImage.width(), m_glImage.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, m_glImage.bits() );
}
