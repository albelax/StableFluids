#ifndef NGLSCENE_H_
#define NGLSCENE_H_


#include "Shader.h"
#include "TrackballCamera.h"
#include "Mesh.h"

#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <ext.hpp>
#include <glm.hpp>
#include <QOpenGLWidget>
#include <QResizeEvent>
#include <QEvent>
#include <memory>
#include "MacStableSolver.h"
#include <QImage>
#include <GpuSolver.h>


class GLWindow : public QOpenGLWidget
{
  Q_OBJECT // must include this if you use Qt signals/slots
public :
  /// @brief Constructor for GLWindow
  //----------------------------------------------------------------------------------------------------------------------
  /// @brief Constructor for GLWindow
  /// @param [in] _parent the parent window to create the GL context in
  //----------------------------------------------------------------------------------------------------------------------
  GLWindow( QWidget *_parent );

  /// @brief dtor
  ~GLWindow();
  void mouseMove( QMouseEvent * _event );
  void mouseClick( QMouseEvent * _event );

public slots:
  //  void rotating( const bool _rotating ) { m_rotating = _rotating; }
  void init();
  void reset();
  void setTimestep( double _timeStep ) { m_solver.setTimestep( static_cast<float>( _timeStep ) ); }
  void setDiffusion( double _diffusion ) { m_solver.setDiffusion( static_cast<float>( _diffusion ) ); }
  void setViscosity( double _viscosity ) { m_solver.setViscosity( static_cast<float>( _viscosity ) ); }
  void setDensity( double _density ) { m_solver.setDensity( static_cast<float>( _density ) ); }

protected:
  /// @brief  The following methods must be implimented in the sub class
  /// this is called when the window is created
  void initializeGL();

  /// @brief this is the main gl drawing routine which is called whenever the window needs to be re-drawn
  void paintGL();
  void renderScene();
  void renderTexture();

private:
  int prevX;
  int prevY;
  int m_amountVertexData;
  GLuint m_vao;
  GLuint m_vbo;
  GLuint m_nbo;
  GLuint m_tbo;
  GLint m_vertexPositionAddress;
  GLint m_vertexNormalAddress;
  GLint m_MVAddress;
  GLint m_MVPAddress;
  GLint m_NAddress;
  GLuint m_colourTextureAddress;
  Shader m_shader;
  TrackballCamera m_camera;
  Mesh * m_mesh;
  std::array<Mesh, 1> m_meshes;

  glm::mat4 m_projection;
  glm::mat4 m_view;
  glm::mat4 m_MV; 
  glm::mat4 m_MVP; 
  StableSolverCpu m_solver;  
  GpuSolver m_solverGpu;
  QImage m_image; 
  std::vector<GLuint> m_textures; 
  void addTexture();  

};

#endif
