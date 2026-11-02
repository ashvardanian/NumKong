/**
 *  @file bench/shapes.cpp
 *  @author Ash Vardanian
 *  @date March 14, 2023
 *  @brief Many-to-many Chamfer, Hausdorff, Wasserstein-Procrustes, Gromov-Wasserstein benchmarks.
 */

#include "numkong/shapes.h"

#include "cross.hpp"

namespace ashvardanian::numkong::bench {

void bench_shapes(environment_t const &env) {
    host_backend_t const backend;

    section(env, "Shapes Serial", nk_cap_serial_k);
    run_shapes_packed<nk_f64_k>(env, "rmsds_packed_f64_serial", backend, nk_rmsds_packed_f64_serial);
    run_shapes_packed<nk_f32_k>(env, "rmsds_packed_f32_serial", backend, nk_rmsds_packed_f32_serial);
    run_shapes_packed<nk_f16_k>(env, "rmsds_packed_f16_serial", backend, nk_rmsds_packed_f16_serial);
    run_shapes_packed<nk_f64_k>(env, "hausdorffs_packed_f64_serial", backend, nk_hausdorffs_packed_f64_serial);
    run_shapes_packed<nk_f32_k>(env, "hausdorffs_packed_f32_serial", backend, nk_hausdorffs_packed_f32_serial);
    run_shapes_packed<nk_f16_k>(env, "hausdorffs_packed_f16_serial", backend, nk_hausdorffs_packed_f16_serial);
    run_shapes_packed<nk_f64_k>(env, "chamfers_packed_f64_serial", backend,
                                with_workspace(nk_chamfers_packed_f64_serial, nk_chamfers_workspace_size_f64_serial));
    run_shapes_packed<nk_f32_k>(env, "chamfers_packed_f32_serial", backend,
                                with_workspace(nk_chamfers_packed_f32_serial, nk_chamfers_workspace_size_f32_serial));
    run_shapes_packed<nk_f16_k>(env, "chamfers_packed_f16_serial", backend,
                                with_workspace(nk_chamfers_packed_f16_serial, nk_chamfers_workspace_size_f16_serial));
    run_shapes_packed<nk_f64_k>(
        env, "wassersteinprocrustes_packed_f64_serial", backend,
        with_workspace(nk_wassersteinprocrustes_packed_f64_serial, nk_wassersteinprocrustes_workspace_size_f64_serial));
    run_shapes_packed<nk_f32_k>(
        env, "wassersteinprocrustes_packed_f32_serial", backend,
        with_workspace(nk_wassersteinprocrustes_packed_f32_serial, nk_wassersteinprocrustes_workspace_size_f32_serial));
    run_shapes_packed<nk_f16_k>(
        env, "wassersteinprocrustes_packed_f16_serial", backend,
        with_workspace(nk_wassersteinprocrustes_packed_f16_serial, nk_wassersteinprocrustes_workspace_size_f16_serial));
    run_shapes_packed<nk_f64_k>(
        env, "gromovwassersteins_packed_f64_serial", backend,
        with_workspace(nk_gromovwassersteins_packed_f64_serial, nk_gromovwassersteins_workspace_size_f64_serial));
    run_shapes_packed<nk_f32_k>(
        env, "gromovwassersteins_packed_f32_serial", backend,
        with_workspace(nk_gromovwassersteins_packed_f32_serial, nk_gromovwassersteins_workspace_size_f32_serial));
    run_shapes_packed<nk_f16_k>(
        env, "gromovwassersteins_packed_f16_serial", backend,
        with_workspace(nk_gromovwassersteins_packed_f16_serial, nk_gromovwassersteins_workspace_size_f16_serial));

#if NUMKONG_TARGET_NEON
    if (section(env, "Shapes NEON", nk_cap_neon_k)) {
        run_shapes_packed<nk_f64_k>(env, "rmsds_packed_f64_neon", backend, nk_rmsds_packed_f64_neon);
        run_shapes_packed<nk_f32_k>(env, "rmsds_packed_f32_neon", backend, nk_rmsds_packed_f32_neon);
        run_shapes_packed<nk_f16_k>(env, "rmsds_packed_f16_neon", backend, nk_rmsds_packed_f16_neon);
        run_shapes_packed<nk_f32_k>(env, "hausdorffs_packed_f32_neon", backend, nk_hausdorffs_packed_f32_neon);
        run_shapes_packed<nk_f16_k>(env, "hausdorffs_packed_f16_neon", backend, nk_hausdorffs_packed_f16_neon);
        run_shapes_packed<nk_f32_k>(env, "chamfers_packed_f32_neon", backend,
                                    with_workspace(nk_chamfers_packed_f32_neon, nk_chamfers_workspace_size_f32_serial));
        run_shapes_packed<nk_f16_k>(env, "chamfers_packed_f16_neon", backend,
                                    with_workspace(nk_chamfers_packed_f16_neon, nk_chamfers_workspace_size_f16_serial));
        run_shapes_packed<nk_f32_k>(env, "wassersteinprocrustes_packed_f32_neon", backend,
                                    with_workspace(nk_wassersteinprocrustes_packed_f32_neon,
                                                   nk_wassersteinprocrustes_workspace_size_f32_serial));
        run_shapes_packed<nk_f16_k>(env, "wassersteinprocrustes_packed_f16_neon", backend,
                                    with_workspace(nk_wassersteinprocrustes_packed_f16_neon,
                                                   nk_wassersteinprocrustes_workspace_size_f16_serial));
        run_shapes_packed<nk_f32_k>(
            env, "gromovwassersteins_packed_f32_neon", backend,
            with_workspace(nk_gromovwassersteins_packed_f32_neon, nk_gromovwassersteins_workspace_size_f32_serial));
        run_shapes_packed<nk_f16_k>(
            env, "gromovwassersteins_packed_f16_neon", backend,
            with_workspace(nk_gromovwassersteins_packed_f16_neon, nk_gromovwassersteins_workspace_size_f16_serial));
    }
#endif

#if NUMKONG_TARGET_NEONFHM
    if (section(env, "Shapes NEON FHM", nk_cap_neonfhm_k)) {
        run_shapes_packed<nk_f16_k>(env, "rmsds_packed_f16_neonfhm", backend, nk_rmsds_packed_f16_neonfhm);
        run_shapes_packed<nk_f16_k>(env, "hausdorffs_packed_f16_neonfhm", backend, nk_hausdorffs_packed_f16_neonfhm);
        run_shapes_packed<nk_f16_k>(
            env, "chamfers_packed_f16_neonfhm", backend,
            with_workspace(nk_chamfers_packed_f16_neonfhm, nk_chamfers_workspace_size_f16_serial));
    }
#endif

#if NUMKONG_TARGET_HASWELL
    if (section(env, "Shapes Haswell", nk_cap_haswell_k)) {
        run_shapes_packed<nk_f64_k>(env, "rmsds_packed_f64_haswell", backend, nk_rmsds_packed_f64_haswell);
        run_shapes_packed<nk_f32_k>(env, "rmsds_packed_f32_haswell", backend, nk_rmsds_packed_f32_haswell);
        run_shapes_packed<nk_f16_k>(env, "rmsds_packed_f16_haswell", backend, nk_rmsds_packed_f16_haswell);
        run_shapes_packed<nk_f32_k>(env, "hausdorffs_packed_f32_haswell", backend, nk_hausdorffs_packed_f32_haswell);
        run_shapes_packed<nk_f16_k>(env, "hausdorffs_packed_f16_haswell", backend, nk_hausdorffs_packed_f16_haswell);
        run_shapes_packed<nk_f32_k>(
            env, "chamfers_packed_f32_haswell", backend,
            with_workspace(nk_chamfers_packed_f32_haswell, nk_chamfers_workspace_size_f32_serial));
        run_shapes_packed<nk_f16_k>(
            env, "chamfers_packed_f16_haswell", backend,
            with_workspace(nk_chamfers_packed_f16_haswell, nk_chamfers_workspace_size_f16_serial));
        run_shapes_packed<nk_f32_k>(env, "wassersteinprocrustes_packed_f32_haswell", backend,
                                    with_workspace(nk_wassersteinprocrustes_packed_f32_haswell,
                                                   nk_wassersteinprocrustes_workspace_size_f32_serial));
        run_shapes_packed<nk_f16_k>(env, "wassersteinprocrustes_packed_f16_haswell", backend,
                                    with_workspace(nk_wassersteinprocrustes_packed_f16_haswell,
                                                   nk_wassersteinprocrustes_workspace_size_f16_serial));
        run_shapes_packed<nk_f32_k>(
            env, "gromovwassersteins_packed_f32_haswell", backend,
            with_workspace(nk_gromovwassersteins_packed_f32_haswell, nk_gromovwassersteins_workspace_size_f32_serial));
        run_shapes_packed<nk_f16_k>(
            env, "gromovwassersteins_packed_f16_haswell", backend,
            with_workspace(nk_gromovwassersteins_packed_f16_haswell, nk_gromovwassersteins_workspace_size_f16_serial));
    }
#endif

#if NUMKONG_TARGET_SKYLAKE
    if (section(env, "Shapes Skylake", nk_cap_skylake_k)) {
        run_shapes_packed<nk_f64_k>(env, "rmsds_packed_f64_skylake", backend, nk_rmsds_packed_f64_skylake);
        run_shapes_packed<nk_f32_k>(env, "rmsds_packed_f32_skylake", backend, nk_rmsds_packed_f32_skylake);
        run_shapes_packed<nk_f16_k>(env, "rmsds_packed_f16_skylake", backend, nk_rmsds_packed_f16_skylake);
        run_shapes_packed<nk_f64_k>(env, "hausdorffs_packed_f64_skylake", backend, nk_hausdorffs_packed_f64_skylake);
        run_shapes_packed<nk_f32_k>(env, "hausdorffs_packed_f32_skylake", backend, nk_hausdorffs_packed_f32_skylake);
        run_shapes_packed<nk_f16_k>(env, "hausdorffs_packed_f16_skylake", backend, nk_hausdorffs_packed_f16_skylake);
        run_shapes_packed<nk_f64_k>(
            env, "chamfers_packed_f64_skylake", backend,
            with_workspace(nk_chamfers_packed_f64_skylake, nk_chamfers_workspace_size_f64_serial));
        run_shapes_packed<nk_f32_k>(
            env, "chamfers_packed_f32_skylake", backend,
            with_workspace(nk_chamfers_packed_f32_skylake, nk_chamfers_workspace_size_f32_serial));
        run_shapes_packed<nk_f16_k>(
            env, "chamfers_packed_f16_skylake", backend,
            with_workspace(nk_chamfers_packed_f16_skylake, nk_chamfers_workspace_size_f16_serial));
        run_shapes_packed<nk_f64_k>(env, "wassersteinprocrustes_packed_f64_skylake", backend,
                                    with_workspace(nk_wassersteinprocrustes_packed_f64_skylake,
                                                   nk_wassersteinprocrustes_workspace_size_f64_serial));
        run_shapes_packed<nk_f32_k>(env, "wassersteinprocrustes_packed_f32_skylake", backend,
                                    with_workspace(nk_wassersteinprocrustes_packed_f32_skylake,
                                                   nk_wassersteinprocrustes_workspace_size_f32_serial));
        run_shapes_packed<nk_f16_k>(env, "wassersteinprocrustes_packed_f16_skylake", backend,
                                    with_workspace(nk_wassersteinprocrustes_packed_f16_skylake,
                                                   nk_wassersteinprocrustes_workspace_size_f16_serial));
        run_shapes_packed<nk_f64_k>(
            env, "gromovwassersteins_packed_f64_skylake", backend,
            with_workspace(nk_gromovwassersteins_packed_f64_skylake, nk_gromovwassersteins_workspace_size_f64_serial));
        run_shapes_packed<nk_f32_k>(
            env, "gromovwassersteins_packed_f32_skylake", backend,
            with_workspace(nk_gromovwassersteins_packed_f32_skylake, nk_gromovwassersteins_workspace_size_f32_serial));
        run_shapes_packed<nk_f16_k>(
            env, "gromovwassersteins_packed_f16_skylake", backend,
            with_workspace(nk_gromovwassersteins_packed_f16_skylake, nk_gromovwassersteins_workspace_size_f16_serial));
    }
#endif

#if NUMKONG_TARGET_RVV
    if (section(env, "Shapes RVV", nk_cap_rvv_k)) {
        run_shapes_packed<nk_f64_k>(env, "rmsds_packed_f64_rvv", backend, nk_rmsds_packed_f64_rvv);
        run_shapes_packed<nk_f32_k>(env, "rmsds_packed_f32_rvv", backend, nk_rmsds_packed_f32_rvv);
        run_shapes_packed<nk_f16_k>(env, "rmsds_packed_f16_rvv", backend, nk_rmsds_packed_f16_rvv);
        run_shapes_packed<nk_f32_k>(env, "hausdorffs_packed_f32_rvv", backend, nk_hausdorffs_packed_f32_rvv);
        run_shapes_packed<nk_f16_k>(env, "hausdorffs_packed_f16_rvv", backend, nk_hausdorffs_packed_f16_rvv);
        run_shapes_packed<nk_f32_k>(env, "chamfers_packed_f32_rvv", backend,
                                    with_workspace(nk_chamfers_packed_f32_rvv, nk_chamfers_workspace_size_f32_serial));
        run_shapes_packed<nk_f16_k>(env, "chamfers_packed_f16_rvv", backend,
                                    with_workspace(nk_chamfers_packed_f16_rvv, nk_chamfers_workspace_size_f16_serial));
        run_shapes_packed<nk_f32_k>(env, "wassersteinprocrustes_packed_f32_rvv", backend,
                                    with_workspace(nk_wassersteinprocrustes_packed_f32_rvv,
                                                   nk_wassersteinprocrustes_workspace_size_f32_serial));
        run_shapes_packed<nk_f16_k>(env, "wassersteinprocrustes_packed_f16_rvv", backend,
                                    with_workspace(nk_wassersteinprocrustes_packed_f16_rvv,
                                                   nk_wassersteinprocrustes_workspace_size_f16_serial));
        run_shapes_packed<nk_f32_k>(
            env, "gromovwassersteins_packed_f32_rvv", backend,
            with_workspace(nk_gromovwassersteins_packed_f32_rvv, nk_gromovwassersteins_workspace_size_f32_serial));
        run_shapes_packed<nk_f16_k>(
            env, "gromovwassersteins_packed_f16_rvv", backend,
            with_workspace(nk_gromovwassersteins_packed_f16_rvv, nk_gromovwassersteins_workspace_size_f16_serial));
    }
#endif

#if NUMKONG_TARGET_V128RELAXED
    if (section(env, "Shapes V128 Relaxed", nk_cap_v128relaxed_k)) {
        run_shapes_packed<nk_f64_k>(env, "rmsds_packed_f64_v128relaxed", backend, nk_rmsds_packed_f64_v128relaxed);
        run_shapes_packed<nk_f32_k>(env, "rmsds_packed_f32_v128relaxed", backend, nk_rmsds_packed_f32_v128relaxed);
        run_shapes_packed<nk_f16_k>(env, "rmsds_packed_f16_v128relaxed", backend, nk_rmsds_packed_f16_v128relaxed);
        run_shapes_packed<nk_f32_k>(env, "hausdorffs_packed_f32_v128relaxed", backend,
                                    nk_hausdorffs_packed_f32_v128relaxed);
        run_shapes_packed<nk_f16_k>(env, "hausdorffs_packed_f16_v128relaxed", backend,
                                    nk_hausdorffs_packed_f16_v128relaxed);
        run_shapes_packed<nk_f32_k>(
            env, "chamfers_packed_f32_v128relaxed", backend,
            with_workspace(nk_chamfers_packed_f32_v128relaxed, nk_chamfers_workspace_size_f32_serial));
        run_shapes_packed<nk_f16_k>(
            env, "chamfers_packed_f16_v128relaxed", backend,
            with_workspace(nk_chamfers_packed_f16_v128relaxed, nk_chamfers_workspace_size_f16_serial));
        run_shapes_packed<nk_f32_k>(env, "wassersteinprocrustes_packed_f32_v128relaxed", backend,
                                    with_workspace(nk_wassersteinprocrustes_packed_f32_v128relaxed,
                                                   nk_wassersteinprocrustes_workspace_size_f32_serial));
        run_shapes_packed<nk_f16_k>(env, "wassersteinprocrustes_packed_f16_v128relaxed", backend,
                                    with_workspace(nk_wassersteinprocrustes_packed_f16_v128relaxed,
                                                   nk_wassersteinprocrustes_workspace_size_f16_serial));
        run_shapes_packed<nk_f32_k>(env, "gromovwassersteins_packed_f32_v128relaxed", backend,
                                    with_workspace(nk_gromovwassersteins_packed_f32_v128relaxed,
                                                   nk_gromovwassersteins_workspace_size_f32_serial));
        run_shapes_packed<nk_f16_k>(env, "gromovwassersteins_packed_f16_v128relaxed", backend,
                                    with_workspace(nk_gromovwassersteins_packed_f16_v128relaxed,
                                                   nk_gromovwassersteins_workspace_size_f16_serial));
    }
#endif
}

} // namespace ashvardanian::numkong::bench
