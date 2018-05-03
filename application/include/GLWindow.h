#ifndef OPENGLSCENE_H_
#define OPENGLSCENE_H_


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
#include <QImage>
#include "Solver.h"
#include "MacStableSolver.h"
#include <GpuSolver.h>

enum solverType { CPU, GPU };

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
  void setTimestep( double _timeStep ) { m_activeSolver->setTimestep( static_cast<float>( _timeStep ) ); }
  void setDiffusion( double _diffusion ) { m_activeSolver->setDiffusion( static_cast<float>( _diffusion ) ); }
  void setViscosity( double _viscosity ) { m_activeSolver->setViscosity( static_cast<float>( _viscosity ) ); }
  void setDensity( double _density ) { m_activeSolver->setDensity( static_cast<float>( _density ) ); }

protected:
  /// @brief  The following methods must be implimented in the sub class
  /// this is called when the window is created
  void initializeGL();

  /// @brief this is the main gl drawing routine which is called whenever the window needs to be re-drawn
  void paintGL();
  void renderScene();
  void renderTexture();
  void draw(const real * _density, int _size , bool _save);

private:
  enum solverType m_solverType;
  bool m_active = false;
  int m_prevX;
  int m_prevY;
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

  /// \brief m_solver, cpu solver
  StableSolverCpu m_solver;

  /// \brief m_solverGpu, gpu solver
  GpuSolver m_solverGpu;

  /// \brief m_activeSolver, pointer to the active solver, it will be set in the constructor
  std::unique_ptr<Solver> m_activeSolver;

  /// \brief m_image, image used to draw on the screen
  QImage m_image; 
  std::vector<GLuint> m_textures; 
  void addTexture();  
};

#endif
