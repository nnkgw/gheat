// heat_geodesic_viewer.cpp
// Minimal single-file implementation of the Heat Method geodesic distance
// using C++17 + Eigen (sparse) + freeglut. Renders an OBJ triangle mesh
// with per-vertex color = geodesic distance from a random source vertex.
// Controls:
//   r : re-pick random source and recompute
//   arrow keys : rotate
//   +/- : zoom
//   q/ESC : quit
//
// Notes:
// - This is a teaching-oriented minimal implementation: no indices/materials, only 'v' and triangular 'f'.
// - Laplacian: cotan; Mass: lumped (1/3 sum of incident triangle areas).
// - Heat step: (A - t*Lc) u = delta; with t = h^2 (h = mean edge length).
// - Gradient per face, normalize to unit vector, then divergence to build RHS b, solve Lc * phi = b.
// - Shift phi so that min(phi)=0 for coloring.
// - Coloring: simple scalar->RGB (viridis-like-ish) without external libs.

#if defined(WIN32)
#pragma warning(disable:4996)
#include <GL/freeglut.h>
#elif defined(__APPLE__) || defined(MACOSX)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/Geometry>
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

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Sparse = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

struct Mesh {
  std::vector<Vec3> V;     // vertices
  std::vector<Eigen::Vector3i> F; // triangle indices (0-based)
  std::vector<double> Avert; // lumped vertex areas
  std::vector<double> phi;   // distance field
  std::vector<double> uheat; // heat solution
  Sparse Lc; // cotan stiffness
  Sparse A;  // lumped mass (diagonal)
  double meanEdge = 1.0;
};

static Mesh gM;
static int gWinW = 1000, gWinH = 800;
static double gRotX = 20.0, gRotY = -30.0, gZoom = 2.0;
static Vec3 gCenter(0,0,0);
static int gSource = -1;
static std::mt19937_64 gRng(12345);
static Eigen::SimplicialLDLT<Sparse> gSolverHeat;
static Eigen::SimplicialLDLT<Sparse> gSolverPoisson;
static bool gReady = false;
static std::string gObjPath;

static inline double clamp01(double x){ return x<0?0:(x>1?1:x); }

// Simple OBJ loader (vertex 'v' and triangular 'f' only, 1-based to 0-based)
bool loadOBJ(const std::string& path, Mesh& M) {
  std::ifstream in(path);
  if(!in) { std::fprintf(stderr,"[ERR] cannot open: %s\n", path.c_str()); return false; }
  std::string line;
  std::vector<Vec3> V;
  std::vector<Eigen::Vector3i> F;
  while(std::getline(in,line)) {
    if(line.empty() || line[0]=='#') continue;
    std::istringstream ss(line);
    std::string tag; ss >> tag;
    if(tag=="v") {
      double x,y,z; ss>>x>>y>>z;
      V.emplace_back(x,y,z);
    } else if(tag=="f") {
      // accept forms: f i j k  or f i/... j/... k/...
      auto parseIndex = [&](const std::string& tok)->int{
        // split by '/'
        size_t p = tok.find('/');
        if(p==std::string::npos) return std::stoi(tok)-1;
        else return std::stoi(tok.substr(0,p))-1;
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
  return true;
}

// Compute per-face area and accumulate lumped vertex area (1/3 per incident face)
static inline double triArea(const Vec3& a, const Vec3& b, const Vec3& c) {
  return 0.5 * ((b-a).cross(c-a)).norm();
}

// Cotangent of angle at 'a' of triangle (a,b,c): angle between (b-a) and (c-a)
static inline double cotan_at(const Vec3& a, const Vec3& b, const Vec3& c) {
  Vec3 u = b - a, v = c - a;
  double num = u.dot(v);
  double den = u.cross(v).norm();
  if(den < 1e-20) return 0.0;
  return num / den;
}

// Build cotan Laplacian Lc (symmetric) and lumped mass A (diagonal)
void buildLaplacianAndMass(Mesh& M) {
  const int n = (int)M.V.size();
  std::vector<double> Avert(n,0.0);
  std::vector<Triplet> Ltrip;
  std::vector<double> diag(n, 0.0);

  // accumulate mean edge length
  double elSum = 0.0; size_t elCount = 0;

  // For each face: contributions
  for(const auto& f : M.F) {
    int i=f[0], j=f[1], k=f[2];
    const Vec3 &vi=M.V[i], &vj=M.V[j], &vk=M.V[k];

    // areas
    double Af = triArea(vi,vj,vk);
    if(!(Af>0)) continue;
    Avert[i] += Af/3.0;
    Avert[j] += Af/3.0;
    Avert[k] += Af/3.0;

    // per-vertex cotangents (angles at i,j,k)
    double coti = cotan_at(vi, vj, vk);
    double cotj = cotan_at(vj, vk, vi);
    double cotk = cotan_at(vk, vi, vj);

    // edge-based weights: w_ij = (cot at k)/2, etc. (cot(alpha_ij)+cot(beta_ij)) will be accumulated over faces)
    double wij = 0.5 * cotk;
    double wjk = 0.5 * coti;
    double wki = 0.5 * cotj;

    // symmetric assembly: off-diagonals negative weights (stiffness), diagonals sum
    auto add_edge = [&](int a,int b,double w){
      Ltrip.emplace_back(a,b,-w);
      Ltrip.emplace_back(b,a,-w);
      diag[a] += w;
      diag[b] += w;
    };
    add_edge(i,j,wij);
    add_edge(j,k,wjk);
    add_edge(k,i,wki);

    // accumulate edges for mean edge length
    elSum += (vi - vj).norm(); elCount++;
    elSum += (vj - vk).norm(); elCount++;
    elSum += (vk - vi).norm(); elCount++;
  }

  // finalize diagonals
  for(int i=0;i<n;++i) Ltrip.emplace_back(i,i, diag[i]);

  // build Lc
  M.Lc.resize(n,n);
  M.Lc.setFromTriplets(Ltrip.begin(), Ltrip.end());
  M.Lc.makeCompressed();

  // build A mass (diagonal)
  std::vector<Triplet> Atrip; Atrip.reserve(n);
  for(int i=0;i<n;++i) Atrip.emplace_back(i,i, std::max(1e-16, Avert[i]));
  M.A.resize(n,n);
  M.A.setFromTriplets(Atrip.begin(), Atrip.end());
  M.A.makeCompressed();

  M.Avert.swap(Avert);
  M.meanEdge = (elCount>0) ? (elSum / (double)elCount) : 1.0;
}

// Solve (A - t*Lc) u = delta_source
Eigen::VectorXd solveHeat(const Mesh& M, int source) {
  const int n = (int)M.V.size();
  double t = M.meanEdge * M.meanEdge;

  Sparse H = M.A - t * M.Lc;

  // factorize (re-using global solver for speed if needed)
  gSolverHeat.compute(H);
  if(gSolverHeat.info()!=Eigen::Success) {
    std::fprintf(stderr,"[ERR] Heat factorization failed\n");
  }

  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
  // Use a Kronecker delta (integrated Dirac) as in the paper
  rhs[source] = 1.0;

  Eigen::VectorXd u = gSolverHeat.solve(rhs);
  if(gSolverHeat.info()!=Eigen::Success){
    std::fprintf(stderr,"[ERR] Heat solve failed\n");
  }
  return u;
}

// Compute per-face gradient of u (constant per face)
static inline Vec3 faceGradient(const Vec3& vi, const Vec3& vj, const Vec3& vk,
                                double ui, double uj, double uk) {
  Vec3 e0 = vj - vi;
  Vec3 e1 = vk - vj;
  Vec3 e2 = vi - vk; // cyclic
  Vec3 N = e0.cross(vk - vi).normalized();
  double Af = 0.5 * (e0.cross(vk - vi)).norm();
  if(Af < 1e-20) return Vec3::Zero();

  // Opposite edge vectors for each vertex (oriented CCW): e_i = v_k - v_j for vertex i opposite jk
  Vec3 e_i = vk - vj; // opposite to vertex i
  Vec3 e_j = vi - vk; // opposite to vertex j
  Vec3 e_k = vj - vi; // opposite to vertex k

  Vec3 grad =
    (ui * (N.cross(e_i)) + uj * (N.cross(e_j)) + uk * (N.cross(e_k))) / (2.0 * Af);
  return grad;
}

// Build divergence b using per-face unit vector field X = -grad(u)/|grad(u)|
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
    Vec3 X = -gu / gu_norm; // unit vector field along geodesics

    // Local per-vertex cotangents for this face
    double coti = cotan_at(vi, vj, vk);
    double cotj = cotan_at(vj, vk, vi);
    double cotk = cotan_at(vk, vi, vj);

    // For vertex i: e1 = vj - vi (opposite angle at k), e2 = vk - vi (opposite angle at j)
    Vec3 e1 = vj - vi; // opposite angle at k
    Vec3 e2 = vk - vi; // opposite angle at j
    double div_i = 0.5 * ( cotk * e1.dot(X) + cotj * e2.dot(X) );

    // For vertex j
    Vec3 e1j = vk - vj; // opposite angle at i
    Vec3 e2j = vi - vj; // opposite angle at k
    double div_j = 0.5 * ( coti * e1j.dot(X) + cotk * e2j.dot(X) );

    // For vertex k
    Vec3 e1k = vi - vk; // opposite angle at j
    Vec3 e2k = vj - vk; // opposite angle at i
    double div_k = 0.5 * ( cotj * e1k.dot(X) + coti * e2k.dot(X) );

    b[i] += div_i;
    b[j] += div_j;
    b[k] += div_k;
  }
  return b;
}

// Solve Lc * phi = b, then shift so min(phi)=0
Eigen::VectorXd solvePoisson(const Mesh& M, const Eigen::VectorXd& b) {
  gSolverPoisson.compute(M.Lc);
  if(gSolverPoisson.info()!=Eigen::Success) {
    std::fprintf(stderr,"[ERR] Poisson factorization failed\n");
  }
  Eigen::VectorXd phi = gSolverPoisson.solve(b);
  if(gSolverPoisson.info()!=Eigen::Success){
    std::fprintf(stderr,"[ERR] Poisson solve failed\n");
  }
  // shift min to 0
  double mn = phi.minCoeff();
  phi.array() -= mn;
  return phi;
}

void computeDistance(Mesh& M, int source) {
  if(source<0 || source >= (int)M.V.size()) return;
  Eigen::VectorXd u = solveHeat(M, source);
  Eigen::VectorXd b = buildDivergenceRHS(M, u);
  Eigen::VectorXd phi = solvePoisson(M, b);
  // store
  M.uheat.assign(u.data(), u.data()+u.size());
  M.phi.assign(phi.data(), phi.data()+phi.size());
}

// Compute center and scale for viewing
void computeCenterAndScale() {
  if(gM.V.empty()) return;
  Vec3 mn = gM.V[0], mx = gM.V[0];
  for(const auto& p : gM.V){
    mn = mn.cwiseMin(p);
    mx = mx.cwiseMax(p);
  }
  gCenter = 0.5*(mn+mx);
  double diag = (mx - mn).norm();
  if(diag>1e-12) gZoom = 2.2 / diag;
}

// Map scalar s in [smin, smax] to RGB (lightweight "viridis-ish")
void scalarToRGB(double s, double smin, double smax, double& r, double& g, double& b) {
  double t = (s - smin) / (smax - smin + 1e-20);
  t = clamp01(t);
  // piecewise polynomials approximating viridis-like palette
  // (quick approximation: blue->teal->green->yellow)
  double x = t;
  r = clamp01(-0.7 + 3.5*x);
  g = clamp01(0.2 + 2.0*x);
  b = clamp01(0.9 - 1.2*x + 0.3*std::sin(6.28318*x));
}

// Render triangles with per-vertex color from distance
void renderMesh() {
  if(gM.V.empty() || gM.F.empty() || gM.phi.empty()) return;

  // Choose color range: [0, p95] to enhance contrast
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
      Vec3 p = gM.V[vid];
      glVertex3d(p[0], p[1], p[2]);
    }
  }
  glEnd();

  // mark source vertex (small white point)
  if(gSource>=0){
    glPointSize(6.f);
    glBegin(GL_POINTS);
    glColor3d(1,1,1);
    Vec3 p = gM.V[gSource];
    glVertex3d(p[0], p[1], p[2]);
    glEnd();
  }
}

void displayCB() {
  glClearColor(0.08f,0.08f,0.1f,1.f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glEnable(GL_DEPTH_TEST);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  double aspect = (double)gWinW / (double)gWinH;
  gluPerspective(45.0, aspect, 0.01, 100.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslated(0,0,-3.0);
  glScaled(gZoom, gZoom, gZoom);
  glRotated(gRotX, 1,0,0);
  glRotated(gRotY, 0,1,0);
  glTranslated(-gCenter[0], -gCenter[1], -gCenter[2]);

  renderMesh();

  glutSwapBuffers();
}

void reshapeCB(int w, int h) {
  gWinW = (w>1)?w:1;
  gWinH = (h>1)?h:1;
  glViewport(0,0,gWinW,gWinH);
}

void keyboardCB(unsigned char key, int, int) {
  if(key==27 || key=='q') {
    std::exit(0);
  } else if(key=='+') {
    gZoom *= 1.1;
  } else if(key=='-') {
    gZoom /= 1.1;
  } else if(key=='r') {
    std::uniform_int_distribution<int> dist(0, (int)gM.V.size()-1);
    gSource = dist(gRng);
    computeDistance(gM, gSource);
  }
  glutPostRedisplay();
}

void specialCB(int key, int, int) {
  if(key==GLUT_KEY_LEFT) gRotY -= 5;
  if(key==GLUT_KEY_RIGHT) gRotY += 5;
  if(key==GLUT_KEY_UP) gRotX -= 5;
  if(key==GLUT_KEY_DOWN) gRotX += 5;
  glutPostRedisplay();
}

int main(int argc, char** argv) {
  if(argc < 2) {
    std::fprintf(stderr,"Usage: %s mesh.obj\n", argv[0]);
    return 1;
  }
  gObjPath = argv[1];
  if(!loadOBJ(gObjPath, gM)) return 1;

  buildLaplacianAndMass(gM);
  computeCenterAndScale();

  // pick initial random source and compute
  std::uniform_int_distribution<int> dist(0, (int)gM.V.size()-1);
  gSource = dist(gRng);
  computeDistance(gM, gSource);

  // init glut
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
  glutInitWindowSize(gWinW, gWinH);
  glutCreateWindow("Heat Geodesic (Eigen + freeglut) - minimal");
  glutDisplayFunc(displayCB);
  glutReshapeFunc(reshapeCB);
  glutKeyboardFunc(keyboardCB);
  glutSpecialFunc(specialCB);

  glEnable(GL_POLYGON_OFFSET_FILL);
  glPolygonOffset(1.0f, 1.0f);

  glutMainLoop();
  return 0;
}
