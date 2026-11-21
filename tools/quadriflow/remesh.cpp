// QuadriFlow's pipeline (src/main.cpp, BSD-3-Clause, see NOTICE) as a function for a WebAssembly build:
// qf_remesh(input.obj, output.obj, faces, minimumCostFlow, seed). The external-minisat SAT pass (-sat) is not used.
#include "config.hpp"
#include "field-math.hpp"
#include "optimizer.hpp"
#include "parametrizer.hpp"
#include <exception>
#include <iostream>

using namespace qflow;

extern "C" int qf_remesh(const char *input, const char *output, int faces, int minimumCostFlow, int seed) {
    Parametrizer *field = new Parametrizer();
    int status = 0;
    try {
        field->flag_minimum_cost_flow = minimumCostFlow;
        field->hierarchy.rng_seed = seed;
        field->Load(input);
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
