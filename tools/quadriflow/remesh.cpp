// QuadriFlow's pipeline (src/main.cpp, BSD-3-Clause, see NOTICE) as a function for a WebAssembly build. The mesh
// comes in as arrays (QuadriFlow's own Load = OBJ parse + NormalizeMesh; parsing text cost up to 0.3 s), the quads
// go out as an OBJ file in the module's memory file system. The external-minisat SAT pass (-sat) is not used.
//   qf_remesh(vertices[3 nv], nv, triangles[3 nt], nt, quad count, seed, output.obj) -> 0 on success
#include "config.hpp"
#include "field-math.hpp"
#include "optimizer.hpp"
#include "parametrizer.hpp"
#include <exception>
#include <iostream>

using namespace qflow;

extern "C" int qf_remesh(const double *vertices, int nv, const int *triangles, int nt, int faces, int seed, const char *output) {
    Parametrizer *field = new Parametrizer();
    int status = 0;
    try {
        field->hierarchy.rng_seed = seed;
        field->V.resize(3, nv);
        for (int i = 0; i < nv; ++i) for (int c = 0; c < 3; ++c) field->V(c, i) = vertices[i * 3 + c];
        field->F.resize(3, nt);
        for (int i = 0; i < nt; ++i) for (int c = 0; c < 3; ++c) field->F(c, i) = triangles[i * 3 + c];
        field->NormalizeMesh();
        field->Initialize(faces);
        Optimizer::optimize_orientations(field->hierarchy);
        field->ComputeOrientationSingularities();
        Optimizer::optimize_scale(field->hierarchy, field->rho, field->flag_adaptive_scale);
        field->flag_adaptive_scale = 1;
        Optimizer::optimize_positions(field->hierarchy, field->flag_adaptive_scale);
        field->ComputePositionSingularities();
        field->ComputeIndexMap();
        field->OutputMesh(output);
    } catch (const std::exception &e) {
        std::cerr << "qf_remesh: " << e.what() << std::endl;
        status = 1;
    }
    delete field;
    return status;
}
