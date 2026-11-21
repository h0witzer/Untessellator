// Instant Meshes' batch pipeline (src/batch.cpp, BSD-3-Clause, see NOTICE) for a mesh in memory-file form, run
// inline: the optimizer's level loop is called directly instead of on its worker thread, which a single-threaded
// WebAssembly build cannot start. Exported as im_remesh(input.obj, output.obj, faces, pureQuad, creaseDegrees).
#include "batch.h"
#include "meshio.h"
#include "dedge.h"
#include "subdivide.h"
#include "meshstats.h"
#include "hierarchy.h"
#include "field.h"
#include "normal.h"
#include "extract.h"
#include <set>

extern void freeze_ivars_orientations(MultiResolutionHierarchy &mRes, int level, bool extrinsic, int rosy);
extern void freeze_ivars_positions(MultiResolutionHierarchy &mRes, int level, bool extrinsic, int posy);

static void optimize(MultiResolutionHierarchy &mRes, bool orientations, bool extrinsic, int sym) {
    const int levelIterations = 6;
    auto noop = [](uint32_t) { };
    if (orientations) mRes.setFrozenQ(false); else mRes.setFrozenO(false);
    for (int level = mRes.levels() - 1; level >= 0; --level) {
        for (int it = 0; it < levelIterations; ++it) {
            if (orientations) optimize_orientations(mRes, level, extrinsic, sym, noop);
            else optimize_positions(mRes, level, extrinsic, sym, noop);
        }
        if (level == 0) break;
        const int i = level - 1;                      // hand the field down one level, as the worker does
        const MatrixXu &toUpper = mRes.toUpper(i);
        const MatrixXf &N = mRes.N(i);
        if (orientations) {
            const MatrixXf &src = mRes.Q(i + 1);
            MatrixXf &dst = mRes.Q(i);
            for (uint32_t j = 0; j < (uint32_t) src.cols(); ++j) for (int k = 0; k < 2; ++k) {
                uint32_t d = toUpper(k, j);
                if (d == INVALID) continue;
                Vector3f q = src.col(j), n = N.col(d);
                dst.col(d) = q - n * n.dot(q);
            }
        } else {
            const MatrixXf &src = mRes.O(i + 1), &V = mRes.V(i);
            MatrixXf &dst = mRes.O(i);
            for (uint32_t j = 0; j < (uint32_t) src.cols(); ++j) for (int k = 0; k < 2; ++k) {
                uint32_t d = toUpper(k, j);
                if (d == INVALID) continue;
                Vector3f o = src.col(j), n = N.col(d), v = V.col(d);
                o -= n * n.dot(o - v);
                dst.col(d) = o;
            }
        }
    }
    if (orientations) {
        if (!mRes.frozenQ()) freeze_ivars_orientations(mRes, 0, extrinsic, sym);
        mRes.propagateSolution(sym);                  // Optimizer::stop() after orientations
    } else if (!mRes.frozenO()) {
        freeze_ivars_positions(mRes, 0, extrinsic, sym);
    }
}

extern "C" int im_remesh(const char *input, const char *output, int faceCount, int pureQuad, float creaseAngle) {
    try {
        const int rosy = 4, posy = 4, smoothIter = 0;
        const bool extrinsic = true, deterministic = true;
        MatrixXu F;
        MatrixXf V, N;
        VectorXf A;
        std::set<uint32_t> creaseIn, creaseOut;
        load_mesh_or_pointcloud(input, F, V, N);
        if (F.size() == 0) return 2;                  // point clouds are not taken here
        MeshStats stats = compute_mesh_stats(F, V, deterministic);
        const Float faceArea = stats.mSurfaceArea / faceCount;
        const Float scale = std::sqrt(faceArea);
        VectorXu V2E, E2E;
        VectorXb boundary, nonManifold;
        if (stats.mMaximumEdgeLength * 2 > scale || stats.mMaximumEdgeLength > stats.mAverageEdgeLength * 2) {
            build_dedge(F, V, V2E, E2E, boundary, nonManifold);
            subdivide(F, V, V2E, E2E, boundary, nonManifold, std::min(scale / 2, (Float) stats.mAverageEdgeLength * 2), deterministic);
        }
        build_dedge(F, V, V2E, E2E, boundary, nonManifold);
        AdjacencyMatrix adj = generate_adjacency_matrix_uniform(F, V2E, E2E, nonManifold);
        if (creaseAngle >= 0) generate_crease_normals(F, V, V2E, E2E, boundary, nonManifold, creaseAngle, N, creaseIn);
        else generate_smooth_normals(F, V, V2E, E2E, nonManifold, N);
        compute_dual_vertex_areas(F, V, V2E, E2E, nonManifold, A);
        MultiResolutionHierarchy mRes;
        mRes.setE2E(std::move(E2E));
        mRes.setAdj(std::move(adj));
        mRes.setF(std::move(F));
        mRes.setV(std::move(V));
        mRes.setA(std::move(A));
        mRes.setN(std::move(N));
        mRes.setScale(scale);
        mRes.build(deterministic);
        mRes.resetSolution();
        optimize(mRes, true, extrinsic, rosy);
        optimize(mRes, false, extrinsic, posy);
        MatrixXf O, Nx, Nf;
        std::vector<std::vector<TaggedLink>> adjExtr;
        extract_graph(mRes, extrinsic, rosy, posy, adjExtr, O, Nx, creaseIn, creaseOut, deterministic);
        MatrixXu Fx;
        extract_faces(adjExtr, O, Nx, Nf, Fx, posy, mRes.scale(), creaseOut, true, pureQuad != 0, nullptr, smoothIter);
        write_mesh(output, Fx, O, MatrixXf(), Nf);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "im_remesh: " << e.what() << std::endl;
        return 1;
    }
}
