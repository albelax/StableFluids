#include "GLWindow.h"

#include <iostream>
#include <QColorDialog>
#include <QGLWidget>
#include <QImage>
#include <QScreen>
#include <unistd.h>


// testing gpu functions with the gpu solver
#define CROSS_TESTING 1

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

  m_image = QPixmap( 128, 128 ).toImage();
  m_image.fill(Qt::white);
  m_solver.activate();

  m_solverGpu.activate();
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

  m_MV = glm::translate( m_MV, glm::vec3(0.0f, 0.0f, -2.0f) );
}

//----------------------------------------------------------------------------------------------------------------------

GLWindow::~GLWindow()
{

}

//----------------------------------------------------------------------------------------------------------------------

void GLWindow::mouseMove(QMouseEvent * _event)
{
  m_solver.cleanBuffer();

  m_camera.handleMouseMove( _event->pos().x(), _event->pos().y() );

  float posx = _event->pos().x() / static_cast<float>(width()) * 127.0f;
  float posy = _event->pos().y() / static_cast<float>(height()) * 127.0f;

  int x = static_cast<int>(posx) > 127 ? 127 : static_cast<int>(posx);
  int y = static_cast<int>(posy) > 127 ? 127 : static_cast<int>(posy);

  if ( x < 1 ) x = 1;
  if ( y < 1 ) y = 1;

  if ( _event->buttons() == Qt::RightButton )
    m_solver.setVel0(x,y, _event->pos().x() - prevX, _event->pos().y() - prevY );
  else if ( _event->buttons() == Qt::LeftButton )
    m_solver.setD0(x, y);

  m_solver.addSource();

  update();
}

//----------------------------------------------------------------------------------------------------------------------

void GLWindow::mouseClick(QMouseEvent * _event)
{
  m_solver.cleanBuffer();

  prevX = _event->pos().x();
  prevY = _event->pos().y();


  if ( _event->buttons() == Qt::LeftButton )
    m_solver.setD0(prevX, prevY);
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
  //  GLenum error = glGetError();
  //  if ( error !=  GL_NO_ERROR )
  //    std::cout << "GL Error " << error << '\n';
}

//------------------------------------------------------------------------------------------------------------------------------

void GLWindow::paintGL()
{
#if CROSS_TESTING

  //---------------------------------------- animVel
  m_solverGpu.copyToDevice( m_solver.m_pressure, m_solverGpu.m_pressure, m_solver.m_totCell );
  m_solverGpu.copyToDevice( m_solver.m_divergence, m_solverGpu.m_divergence, m_solver.m_totCell );
  m_solverGpu.copyToDevice( m_solver.m_velocity.x, m_solverGpu.m_velocity.x, m_solver.m_totVelX );
  m_solverGpu.copyToDevice( m_solver.m_velocity.y, m_solverGpu.m_velocity.y, m_solver.m_totVelY);

  m_solverGpu.projection();


  m_solverGpu.copy( m_solverGpu.m_pressure,   m_solver.m_pressure,   m_solver.m_totCell );
  m_solverGpu.copy( m_solverGpu.m_divergence, m_solver.m_divergence, m_solver.m_totCell );
  m_solverGpu.copy( m_solverGpu.m_velocity.x, m_solver.m_velocity.x, m_solver.m_totVelX );
  m_solverGpu.copy( m_solverGpu.m_velocity.y, m_solver.m_velocity.y, m_solver.m_totVelY);
  //    projection();

  if( m_solver.m_diffusion > 0.0f)
  {
    SWAP(m_solver.m_previousVelocity.x, m_solver.m_velocity.x);
    SWAP(m_solver.m_previousVelocity.y, m_solver.m_velocity.y);
    m_solver.diffuseVel();
  }

  SWAP(m_solver.m_previousVelocity.x, m_solver.m_velocity.x);
  SWAP(m_solver.m_previousVelocity.y, m_solver.m_velocity.y);
  m_solver.advectVel();

  m_solverGpu.copyToDevice( m_solver.m_pressure, m_solverGpu.m_pressure, m_solver.m_totCell );
  m_solverGpu.copyToDevice( m_solver.m_divergence, m_solverGpu.m_divergence, m_solver.m_totCell );
  m_solverGpu.copyToDevice( m_solver.m_velocity.x, m_solverGpu.m_velocity.x, m_solver.m_totVelX );
  m_solverGpu.copyToDevice( m_solver.m_velocity.y, m_solverGpu.m_velocity.y, m_solver.m_totVelY);

  m_solverGpu.projection();


  m_solverGpu.copy( m_solverGpu.m_pressure,   m_solver.m_pressure,   m_solver.m_totCell );
  m_solverGpu.copy( m_solverGpu.m_divergence, m_solver.m_divergence, m_solver.m_totCell );
  m_solverGpu.copy( m_solverGpu.m_velocity.x, m_solver.m_velocity.x, m_solver.m_totVelX );
  m_solverGpu.copy( m_solverGpu.m_velocity.y, m_solver.m_velocity.y, m_solver.m_totVelY);
  //----------------------------------------
#else
  m_solver.animVel();
#endif
  m_solver.animDen();

  glClearColor( 1, 1, 1, 1.0f );
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );


  for ( int i = 0; i < m_image.height(); ++i )
  {
    for ( int j = 0; j < m_image.width(); ++j )
    {
      float r,g,b = 0;

      r = 255 - ( m_solver.getDens(i, j) > 255 ? 255 : m_solver.getDens(i, j) );
      g = 255 - ( m_solver.getDens(i, j) > 255 ? 255 : m_solver.getDens(i, j) );
      b = 255 - ( m_solver.getDens(i, j) > 255 ? 255 : m_solver.getDens(i, j) );

      m_image.setPixel(i,j,qRgb(r,g,b) );
    }
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
  m_solver.reset();
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

  //  m_image.load(_image.c_str());

  auto m_glImage = QGLWidget::convertToGLFormat( m_image );
  if(m_glImage.isNull())
    qWarning("IMAGE IS NULL");
  glBindTexture( GL_TEXTURE_2D, m_textures[m_textures.size()-1] );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, m_glImage.width(), m_glImage.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, m_glImage.bits() );
}
