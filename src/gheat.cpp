// gheat.cpp
// Heat Method (Geodesics in Heat) viewer with glm-based camera & vertex picking.
// Compute: Eigen (sparse). Rendering & camera: OpenGL + GLUT/freeglut + glm.
// Mouse:
//   - Left-drag : rotate (yaw/pitch)
//   - Right-drag: pan
//   - Left-click (no drag): pick nearest projected vertex -> recompute distances
//   - Mouse wheel: zoom (ONLY when FREEGLUT is defined)
// Keys:
//   - r : randomize source vertex
//   - +/- : zoom in/out
//   - q or ESC : quit

#if defined(WIN32)
  #pragma warning(disable:4996)
  #include <GL/freeglut.h>   // defines FREEGLUT
#elif defined(__APPLE__) || defined(MACOSX)
  #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  #define GL_SILENCE_DEPRECATION
  #include <GLUT/glut.h>
#else
  #include <GL/freeglut.h>   // defines FREEGLUT
#endif

#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp> // yawPitchRoll

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/Geometry> // Vector3d::cross, normalized

#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

// ---------------- Types ----------------
using Vec3 = Eigen::Vector3d;
using Sparse = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

struct Mesh {
  std::vector<Vec3> V;
  std::vector<Eigen::Vector3i> F;
  std::vector<double> Avert;   // lumped mass per vertex
  std::vector<double> phi;     // distance field
  std::vector<double> uheat;   // heat solution
  Sparse Lc; // cotan stiffness
  Sparse A;  // lumped mass (diagonal)
  double meanEdge = 1.0;
  Vec3 bbmin = Vec3(DBL_MAX, DBL_MAX, DBL_MAX);
  Vec3 bbmax = Vec3(-DBL_MAX, -DBL_MAX, -DBL_MAX);
};

static Mesh gM;
static std::mt19937_64 gRng(12345);
static Eigen::SimplicialLDLT<Sparse> gSolverHeat;
static Eigen::SimplicialLDLT<Sparse> gSolverPoisson;

// ---------------- Window/Camera ----------------
static int gWinW = 1280, gWinH = 800;
static double camDist = 3.0;     // distance from target
static double yawRad = 0.4;      // yaw around +Y
static double pitchRad = 0.2;    // pitch around +X
static double panX = 0.0, panY = 0.0; // pan in view plane
static Vec3 sceneCenter(0,0,0);

// for picking & drag detection
static int lastX = -1, lastY = -1;
static int downX = -1, downY = -1;
static bool lbtn = false, rbtn = false;
static bool moved = false;
static const int PICK_RADIUS_PX = 12;   // max pixel distance for picking
static const int CLICK_MOVE_TOL = 3;    // pixels: within -> treated as click

// keep glm matrices in sync with OpenGL fixed pipeline
static glm::mat4 gProj(1.0f);
static glm::mat4 gMV(1.0f);
static glm::mat4 gMVP(1.0f);
static const float kFovY = glm::radians(45.0f);
static const float kZNear = 0.001f;
static const float kZFar  = 1000000.0f;

// ---------------- Utils ----------------
static inline double clamp01(double x){ return x<0?0:(x>1?1:x); }

static inline double triArea(const Vec3& a, const Vec3& b, const Vec3& c) {
  return 0.5 * ((b-a).cross(c-a)).norm();
}

static inline double cotan_at(const Vec3& a, const Vec3& b, const Vec3& c) {
  Vec3 u = b - a, v = c - a;
  double num = u.dot(v);
  double den = u.cross(v).norm();
  if(den < 1e-20) return 0.0;
  return num / den;
}

// ---------------- OBJ loader (v + triangular f) ----------------
bool loadOBJ(const std::string& path, Mesh& M) {
  std::ifstream in(path);
  if(!in) { std::fprintf(stderr,"[ERR] cannot open: %s\n", path.c_str()); return false; }
  std::string line;
  std::vector<Vec3> V;
  std::vector<Eigen::Vector3i> F;
  Vec3 bbmin(DBL_MAX,DBL_MAX,DBL_MAX), bbmax(-DBL_MAX,-DBL_MAX,-DBL_MAX);

  while(std::getline(in,line)) {
    if(line.empty() || line[0]=='#') continue;
    std::istringstream ss(line);
    std::string tag; ss >> tag;
    if(tag=="v") {
      double x,y,z; ss>>x>>y>>z;
      V.emplace_back(x,y,z);
      bbmin = bbmin.cwiseMin(Vec3(x,y,z));
      bbmax = bbmax.cwiseMax(Vec3(x,y,z));
    } else if(tag=="f") {
      auto parseIndex = [&](const std::string& tok)->int{
        size_t p = tok.find('/');
        int idx = 0;
        if(p==std::string::npos) idx = std::stoi(tok);
        else idx = std::stoi(tok.substr(0,p));
        if(idx<0) idx = (int)V.size()+idx+1;
        return idx-1;
      };
      std::string a,b,c; ss>>a>>b>>c;
      if(a.empty()||b.empty()||c.empty()) continue;
      int i=parseIndex(a), j=parseIndex(b), k=parseIndex(c);
      if(i>=0 && j>=0 && k>=0) F.emplace_back(i,j,k);
    }
  }
  if(V.empty()||F.empty()){ std::fprintf(stderr,"[ERR] empty mesh\n"); return false; }
  M.V.swap(V);
  M.F.swap(F);
  M.bbmin = bbmin;
  M.bbmax = bbmax;
  return true;
}

// ---------------- Discrete operators ----------------
void buildLaplacianAndMass(Mesh& M) {
  const int n = (int)M.V.size();
  std::vector<double> Avert(n,0.0);
  std::vector<Triplet> Ltrip;
  std::vector<double> diag(n, 0.0);

  double elSum = 0.0; size_t elCount = 0;

  for(const auto& f : M.F) {
    int i=f[0], j=f[1], k=f[2];
    const Vec3 &vi=M.V[i], &vj=M.V[j], &vk=M.V[k];

    double Af = triArea(vi,vj,vk);
    if(!(Af>0)) continue;
    Avert[i] += Af/3.0; Avert[j] += Af/3.0; Avert[k] += Af/3.0;

    double coti = cotan_at(vi, vj, vk);
    double cotj = cotan_at(vj, vk, vi);
    double cotk = cotan_at(vk, vi, vj);

    auto add_edge = [&](int a,int b,double w){
      Ltrip.emplace_back(a,b,-w);
      Ltrip.emplace_back(b,a,-w);
      diag[a] += w; diag[b] += w;
    };
    add_edge(i,j,0.5*cotk);
    add_edge(j,k,0.5*coti);
    add_edge(k,i,0.5*cotj);

    elSum += (vi - vj).norm(); elCount++;
    elSum += (vj - vk).norm(); elCount++;
    elSum += (vk - vi).norm(); elCount++;
  }
  for(int i=0;i<n;++i) Ltrip.emplace_back(i,i, diag[i]);

  M.Lc.resize(n,n); M.Lc.setFromTriplets(Ltrip.begin(), Ltrip.end()); M.Lc.makeCompressed();

  std::vector<Triplet> Atrip; Atrip.reserve(n);
  for(int i=0;i<n;++i) Atrip.emplace_back(i,i, std::max(1e-16, Avert[i]));
  M.A.resize(n,n); M.A.setFromTriplets(Atrip.begin(), Atrip.end()); M.A.makeCompressed();

  M.Avert.swap(Avert);
  M.meanEdge = (elCount>0) ? (elSum / (double)elCount) : 1.0;
}

// ---------------- Heat Method ----------------
Eigen::VectorXd solveHeat(const Mesh& M, int source) {
  const int n = (int)M.V.size();
  double t = M.meanEdge * M.meanEdge;           // t ? h^2
  Sparse H = M.A - t * M.Lc;

  gSolverHeat.compute(H);
  if(gSolverHeat.info()!=Eigen::Success)
    std::fprintf(stderr,"[ERR] Heat factorization failed\n");

  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
  rhs[source] = 1.0;

  Eigen::VectorXd u = gSolverHeat.solve(rhs);
  if(gSolverHeat.info()!=Eigen::Success)
    std::fprintf(stderr,"[ERR] Heat solve failed\n");
  return u;
}

static inline Vec3 faceGradient(const Vec3& vi, const Vec3& vj, const Vec3& vk,
                                double ui, double uj, double uk) {
  Vec3 e0 = vj - vi;
  Vec3 N = e0.cross(vk - vi); // not normalized
  double Af2 = N.norm();      // 2*Area
  if(Af2 < 1e-20) return Vec3::Zero();
  Vec3 e_i = vk - vj; // opposite to vertex i
  Vec3 e_j = vi - vk; // opposite to vertex j
  Vec3 e_k = vj - vi; // opposite to vertex k
  Vec3 nrm = N.normalized();
  Vec3 grad = (ui * (nrm.cross(e_i))
             + uj * (nrm.cross(e_j))
             + uk * (nrm.cross(e_k))) / (Af2);
  return grad;
}

Eigen::VectorXd buildDivergenceRHS(const Mesh& M, const Eigen::VectorXd& u) {
  const int n = (int)M.V.size();
  Eigen::VectorXd b = Eigen::VectorXd::Zero(n);

  for(const auto& f : M.F) {
    int i=f[0], j=f[1], k=f[2];
    const Vec3 &vi=M.V[i], &vj=M.V[j], &vk=M.V[k];
    double ui=u[i], uj=u[j], uk=u[k];

    Vec3 gu = faceGradient(vi,vj,vk, ui,uj,uk);
    double gu_norm = gu.norm();
    if(gu_norm < 1e-20) continue;
    Vec3 X = -gu / gu_norm;

    double coti = cotan_at(vi, vj, vk);
    double cotj = cotan_at(vj, vk, vi);
    double cotk = cotan_at(vk, vi, vj);

    Vec3 e1 = vj - vi; Vec3 e2 = vk - vi;
    double div_i = 0.5 * ( cotk * e1.dot(X) + cotj * e2.dot(X) );
    Vec3 e1j = vk - vj; Vec3 e2j = vi - vj;
    double div_j = 0.5 * ( coti * e1j.dot(X) + cotk * e2j.dot(X) );
    Vec3 e1k = vi - vk; Vec3 e2k = vj - vk;
    double div_k = 0.5 * ( cotj * e1k.dot(X) + coti * e2k.dot(X) );

    b[i] += div_i; b[j] += div_j; b[k] += div_k;
  }
  return b;
}

Eigen::VectorXd solvePoisson(const Mesh& M, const Eigen::VectorXd& b) {
  gSolverPoisson.compute(M.Lc);
  if(gSolverPoisson.info()!=Eigen::Success)
    std::fprintf(stderr,"[ERR] Poisson factorization failed\n");

  Eigen::VectorXd phi = gSolverPoisson.solve(b);
  if(gSolverPoisson.info()!=Eigen::Success)
    std::fprintf(stderr,"[ERR] Poisson solve failed\n");
  double mn = phi.minCoeff();
  phi.array() -= mn;
  return phi;
}

void computeDistance(Mesh& M, int source) {
  if(source<0 || source >= (int)M.V.size()) return;
  Eigen::VectorXd u = solveHeat(M, source);
  Eigen::VectorXd b = buildDivergenceRHS(M, u);
  Eigen::VectorXd phi = solvePoisson(M, b);
  M.uheat.assign(u.data(), u.data()+u.size());
  M.phi.assign(phi.data(), phi.data()+phi.size());
}

// ---------------- Camera ----------------
static void initCameraFromBounds() {
  sceneCenter = 0.5*(gM.bbmin + gM.bbmax);
  double diag = (gM.bbmax - gM.bbmin).norm();
  if (diag <= 1e-12) diag = 1.0;
  camDist = diag * 1.8;
  yawRad = 0.4; pitchRad = 0.2;
  panX = panY = 0.0;
}

static void setProjectionAndView() {
  // OpenGL fixed-pipeline (for drawing)
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  double aspect = (double)gWinW / (double)gWinH;
  gluPerspective(45.0, aspect, kZNear, kZFar);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // glm matrices (for math & picking)
  gProj = glm::perspective(kFovY, (float)aspect, kZNear, kZFar);

  glm::mat4 R = glm::yawPitchRoll((float)yawRad, (float)pitchRad, 0.0f);
  glm::vec3 sc((float)sceneCenter.x(), (float)sceneCenter.y(), (float)sceneCenter.z());
  glm::vec3 eye = sc + glm::vec3(R * glm::vec4(0, 0, (float)camDist, 1));
  glm::mat4 V = glm::lookAt(eye, sc, glm::vec3(0,1,0));
  glm::mat4 Tpan = glm::translate(glm::mat4(1.0f), glm::vec3((float)panX, (float)panY, 0.0f));
  gMV = V * Tpan;
  gMVP = gProj * gMV;

  // Load MV into OpenGL
  glLoadMatrixf(&gMV[0][0]);
}

// ---------------- Coloring ----------------
static void scalarToRGB(double s, double smin, double smax, double& r, double& g, double& b) {
  double t = (s - smin) / (smax - smin + 1e-20);
  t = clamp01(t);
  double x = t;
  r = clamp01(-0.7 + 3.5*x);
  g = clamp01(0.2 +  2.0*x);
  b = clamp01(0.9 -  1.2*x + 0.3*std::sin(6.28318*x));
}

// ---------------- Rendering ----------------
static int gSource = -1;

static void renderMeshColored() {
  if(gM.V.empty() || gM.F.empty() || gM.phi.empty()) return;

  std::vector<double> tmp = gM.phi;
  std::nth_element(tmp.begin(), tmp.begin() + (int)(0.95*tmp.size()), tmp.end());
  double p95 = tmp[(int)(0.95*tmp.size())];
  double smin = 0.0, smax = std::max(1e-9, p95);

  glBegin(GL_TRIANGLES);
  for(const auto& f : gM.F) {
    for(int k=0;k<3;++k){
      int vid = f[k];
      double r,g,b;
      scalarToRGB(gM.phi[vid], smin, smax, r,g,b);
      glColor3d(r,g,b);
      const Vec3& p = gM.V[vid];
      glVertex3d(p.x(), p.y(), p.z());
    }
  }
  glEnd();

  if(gSource>=0){
    glPointSize(6.f);
    glBegin(GL_POINTS);
    glColor3d(1,1,1);
    const Vec3& p = gM.V[gSource];
    glVertex3d(p.x(), p.y(), p.z());
    glEnd();
  }
}

// ---------------- Picking ----------------
// Screen (x,y) in window coords -> pick nearest projected vertex within radius.
static int pickNearestVertex(int mouseX, int mouseY) {
  if(gM.V.empty()) return -1;
  // flip Y to OpenGL window coords
  int winX = mouseX;
  int winY = gWinH - mouseY - 1;

  int best = -1;
  double bestD2 = (double)PICK_RADIUS_PX * (double)PICK_RADIUS_PX;

  // Project each vertex via MVP (glm)
  for (int i=0;i<(int)gM.V.size();++i) {
    const Vec3& v = gM.V[i];
    glm::vec4 clip = gMVP * glm::vec4((float)v.x(), (float)v.y(), (float)v.z(), 1.0f);
    if (clip.w == 0.0f) continue;
    glm::vec3 ndc = glm::vec3(clip) / clip.w; // [-1,1]
    // Cull if behind near/far (optional)
    if (ndc.z < -1.0f || ndc.z > 1.0f) continue;

    // NDC -> window
    float sx = (ndc.x * 0.5f + 0.5f) * (float)gWinW;
    float sy = (ndc.y * 0.5f + 0.5f) * (float)gWinH;
    double dx = (double)sx - (double)winX;
    double dy = (double)sy - (double)winY;
    double d2 = dx*dx + dy*dy;
    if (d2 < bestD2) { bestD2 = d2; best = i; }
  }
  return best;
}

// ---------------- GLUT callbacks ----------------
static void display() {
  glViewport(0,0,gWinW,gWinH);
  glClearColor(0.08f,0.08f,0.1f,1.f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  setProjectionAndView();
  renderMeshColored();

  glutSwapBuffers();
}

static void reshape(int w, int h) {
  gWinW = (w>1)?w:1;
  gWinH = (h>1)?h:1;
  glutPostRedisplay();
}

static void keyboard(unsigned char key, int, int) {
  if(key==27 || key=='q') std::exit(0);
  else if(key=='+') { camDist *= 0.9; if(camDist<1e-3) camDist=1e-3; }
  else if(key=='-') { camDist *= 1.1; }
  else if(key=='r') {
    std::uniform_int_distribution<int> dist(0, (int)gM.V.size()-1);
    gSource = dist(gRng);
    computeDistance(gM, gSource);
  }
  glutPostRedisplay();
}

static void mouse(int button,int state,int x,int y){
  if (button == GLUT_LEFT_BUTTON)  {
    if (state == GLUT_DOWN) {
      lbtn = true; moved = false; downX = lastX = x; downY = lastY = y;
    } else {
      // release: if not moved -> treat as click -> pick
      lbtn = false;
      int dx = x - downX, dy = y - downY;
      if (std::abs(dx) <= CLICK_MOVE_TOL && std::abs(dy) <= CLICK_MOVE_TOL) {
        int vid = pickNearestVertex(x,y);
        if (vid >= 0) {
          gSource = vid;
          computeDistance(gM, gSource);
        }
      }
    }
  }
  if (button == GLUT_RIGHT_BUTTON) {
    if (state == GLUT_DOWN) { rbtn = true; lastX = x; lastY = y; }
    else rbtn = false;
  }
}

static void motion(int x,int y){
  int dx = x - lastX;
  int dy = y - lastY;
  lastX = x; lastY = y;

  if (lbtn) { // rotate
    if (std::abs(x - downX) > CLICK_MOVE_TOL || std::abs(y - downY) > CLICK_MOVE_TOL) moved = true;
    yawRad   += dx * 0.005;
    pitchRad += dy * 0.005;
    const double lim = 1.55;
    if (pitchRad >  lim) pitchRad =  lim;
    if (pitchRad < -lim) pitchRad = -lim;
  }
  if (rbtn) { // pan
    double s = 0.002 * camDist;
    panX += dx * s;
    panY -= dy * s;
  }
  glutPostRedisplay();
}

#if defined(FREEGLUT)
static void wheel(int wheel, int direction, int x, int y){
  (void)wheel; (void)x; (void)y;
  camDist *= (direction > 0) ? 0.9 : 1.1;
  if (camDist < 1e-3) camDist = 1e-3;
  glutPostRedisplay();
}
#endif

// ---------------- Main ----------------
int main(int argc, char** argv) {
  if(argc < 2) {
    std::fprintf(stderr,"Usage: %s mesh.obj\n", argv[0]);
    return 1;
  }
  if(!loadOBJ(argv[1], gM)) return 1;

  buildLaplacianAndMass(gM);

  // initial source & distance
  std::uniform_int_distribution<int> dist(0, (int)gM.V.size()-1);
  gSource = dist(gRng);
  computeDistance(gM, gSource);

  initCameraFromBounds();

  glutInit(&argc, argv);
#if defined(FREEGLUT)
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
  glutInitWindowSize(gWinW, gWinH);
  glutCreateWindow("Geodesics in Heat: A New Approach to Computing Distance Based on Heat Flow");

  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
#if defined(FREEGLUT)
  glutMouseWheelFunc(wheel); // wheel only with freeglut
#endif

  glEnable(GL_POLYGON_OFFSET_FILL);
  glPolygonOffset(1.0f, 1.0f);
  glEnable(GL_LINE_SMOOTH);
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_DEPTH_TEST);

  glutMainLoop();
  return 0;
}
