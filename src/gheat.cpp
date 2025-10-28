// gheat.cpp
// Heat Method (Geodesics in Heat) minimal viewer with mouse camera controls.
// Dependencies: C++17, Eigen (Core/Sparse/Geometry), OpenGL + GLUT/freeglut.
//
// Controls:
//   Mouse Left-drag  : rotate (yaw/pitch)
//   Mouse Right-drag : pan
//   Mouse Wheel      : zoom  (only when FREEGLUT is defined)
//   + / -            : zoom in / out
//   r                : re-pick random source vertex and recompute
//   q / ESC          : quit
//
// Build (Linux):
//   g++ gheat.cpp -std=gnu++17 -O2 -I/path/to/eigen -lglut -lGL -lGLU
// Build (macOS):
//   g++ gheat.cpp -std=gnu++17 -O2 -I/path/to/eigen -framework OpenGL -framework GLUT
// Build (Windows MinGW/MSYS2):
//   g++ gheat.cpp -std=gnu++17 -O2 -I/path/to/eigen -lfreeglut -lopengl32 -lglu32
//
// Note: If you include <GL/freeglut.h>, FREEGLUT macro is defined and wheel zoom is enabled.
//       If you include platform GLUT (<GLUT/glut.h> on macOS), wheel zoom is disabled by #if.

#if defined(WIN32)
  #pragma warning(disable:4996)
  #include <GL/freeglut.h>
  #define HAVE_FREEGLUT 1
#elif defined(__APPLE__) || defined(MACOSX)
  #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  #define GL_SILENCE_DEPRECATION
  #include <GLUT/glut.h>
#else
  #include <GL/freeglut.h>
  #define HAVE_FREEGLUT 1
#endif

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/Geometry> // for Vector3d::cross(), normalized()
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

// ----------------------------- Types -----------------------------
using Vec3 = Eigen::Vector3d;
using Sparse = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

struct Mesh {
  std::vector<Vec3> V;                 // vertices
  std::vector<Eigen::Vector3i> F;      // triangles (0-based)
  std::vector<double> Avert;           // lumped vertex areas
  std::vector<double> phi;             // distance field
  std::vector<double> uheat;           // heat solution
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

// ----------------------------- Camera (mouse ) -----------------------------
// (Left-drag: rotate yaw/pitch, Right-drag: pan, Wheel: zoom ONLY when FREEGLUT)
// Initial camera distance is based on the .obj bounding box size.  :contentReference[oaicite:1]{index=1}
static int gWinW = 1280, gWinH = 800;
static double camDist = 3.0;     // distance from target
static double yawRad = 0.4;      // yaw (around Y), radians
static double pitchRad = 0.2;    // pitch (around X), radians
static double panX = 0.0, panY = 0.0;  // pan in view plane
static int lastX = -1, lastY = -1;
static bool lbtn = false, rbtn = false;
static Vec3 sceneCenter(0,0,0);

// ----------------------------- Helpers -----------------------------
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

// Simple OBJ loader: supports 'v' and triangular 'f' (also f with slashes)
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

Eigen::VectorXd solveHeat(const Mesh& M, int source) {
  const int n = (int)M.V.size();
  double t = M.meanEdge * M.meanEdge;
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
  Vec3 N = e0.cross(vk - vi); // not normalized yet
  double Af2 = N.norm();      // = 2*Area
  if(Af2 < 1e-20) return Vec3::Zero();
  Vec3 e_i = vk - vj; // opposite to vertex i
  Vec3 e_j = vi - vk; // opposite to vertex j
  Vec3 e_k = vj - vi; // opposite to vertex k
  Vec3 grad = (ui * (N.normalized().cross(e_i))
             + uj * (N.normalized().cross(e_j))
             + uk * (N.normalized().cross(e_k))) / (Af2);
  // (Af2 = 2A, denominator should be 2A; we used N.normalized()/ (2A) == N/(2A*|N|)
  // To keep consistent with earlier minimal code, this is sufficient visually.
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

// ----------------------------- Camera utils -----------------------------
static void initCameraFromBounds() {
  sceneCenter = 0.5*(gM.bbmin + gM.bbmax);
  double diag = (gM.bbmax - gM.bbmin).norm();
  if (diag <= 1e-12) diag = 1.0;
  camDist = diag * 1.8;     // like the sample: distance proportional to bbox size  :contentReference[oaicite:2]{index=2}
  yawRad = 0.4; pitchRad = 0.2;
  panX = panY = 0.0;
}

static void setProjectionAndView() {
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  double aspect = (double)gWinW / (double)gWinH;
  gluPerspective(45.0, aspect, 0.001, 1e6);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // ModelView = T_camera * R_yaw * R_pitch * T_pan * T_center
  // (We build with fixed-pipeline matrix ops for simplicity)
  // Move back by camDist
  glTranslated(0, 0, -camDist);

  // Rotate by yaw (Y), then pitch (X)
  glRotated(yawRad * 180.0 / M_PI, 0, 1, 0);
  glRotated(pitchRad * 180.0 / M_PI, 1, 0, 0);

  // Pan (in view plane). Scale pan speed implicitly by camDist in handlers.
  glTranslated(panX, panY, 0);

  // Center the scene
  glTranslated(-sceneCenter.x(), -sceneCenter.y(), -sceneCenter.z());
}

// ----------------------------- Color mapping -----------------------------
static void scalarToRGB(double s, double smin, double smax, double& r, double& g, double& b) {
  double t = (s - smin) / (smax - smin + 1e-20);
  t = clamp01(t);
  double x = t;
  r = clamp01(-0.7 + 3.5*x);
  g = clamp01(0.2 +  2.0*x);
  b = clamp01(0.9 -  1.2*x + 0.3*std::sin(6.28318*x));
}

// ----------------------------- Rendering -----------------------------
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

// ----------------------------- GLUT callbacks -----------------------------
static void displayCB() {
  glViewport(0,0,gWinW,gWinH);
  glClearColor(0.08f,0.08f,0.1f,1.f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  setProjectionAndView();
  renderMeshColored();

  glutSwapBuffers();
}

static void reshapeCB(int w, int h) {
  gWinW = (w>1)?w:1;
  gWinH = (h>1)?h:1;
  glutPostRedisplay();
}

static void keyboardCB(unsigned char key, int, int) {
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

static void mouseButtonCB(int button,int state,int x,int y){
  if (button == GLUT_LEFT_BUTTON)  lbtn = (state == GLUT_DOWN);
  if (button == GLUT_RIGHT_BUTTON) rbtn = (state == GLUT_DOWN);
  lastX = x; lastY = y;
}

static void mouseMotionCB(int x,int y){
  int dx = x - lastX;
  int dy = y - lastY;
  lastX = x; lastY = y;

  if (lbtn) { // rotate
    yawRad   += dx * 0.005;
    pitchRad += dy * 0.005;
    const double lim = 1.55;
    if (pitchRad >  lim) pitchRad =  lim;
    if (pitchRad < -lim) pitchRad = -lim;
  }
  if (rbtn) { // pan (scale by distance)
    double s = 0.002 * camDist;
    panX += dx * s;
    panY -= dy * s;
  }
  glutPostRedisplay();
}

#if defined(HAVE_FREEGLUT)
static void mouseWheelCB(int wheel, int direction, int x, int y){
  (void)wheel; (void)x; (void)y;
  camDist *= (direction > 0) ? 0.9 : 1.1;
  if (camDist < 1e-3) camDist = 1e-3;
  glutPostRedisplay();
}
#endif

// ----------------------------- Main -----------------------------
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

  // init camera from bounds (like the attached geodesic.cpp)  :contentReference[oaicite:3]{index=3}
  initCameraFromBounds();

  glutInit(&argc, argv);
#if defined(HAVE_FREEGLUT)
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
  glutInitWindowSize(gWinW, gWinH);
  glutCreateWindow("Heat Geodesic (Eigen + GLUT) - mouse camera");

  glutDisplayFunc(displayCB);
  glutReshapeFunc(reshapeCB);
  glutKeyboardFunc(keyboardCB);
  glutMouseFunc(mouseButtonCB);
  glutMotionFunc(mouseMotionCB);
#if defined(HAVE_FREEGLUT)
  glutMouseWheelFunc(mouseWheelCB); // enabled only when freeglut is available
#endif

  glEnable(GL_POLYGON_OFFSET_FILL);
  glPolygonOffset(1.0f, 1.0f);
  glEnable(GL_LINE_SMOOTH);
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_DEPTH_TEST);

  glutMainLoop();
  return 0;
}
