// LatticeRunner WebGPU — Hierarchy Pooling & Top-Down Compute Shaders
// Bottom-up: L4→L3→L2→L1
// Top-down:  L3→L4, L2→L3, L1→L2

struct PhysicsParams {
    energy_persist: f32,
    fatigue_rate: f32,
    fatigue_recovery: f32,
    temperature: f32,
    facilitation: f32,
    inhibition: f32,
    anti_facilitation_mult: f32,
    hebbian_rate: f32,
    alpha_l4: f32,
    alpha_l3: f32,
    alpha_l2: f32,
    alpha_l1: f32,
    beta_l4: f32,
    beta_l3: f32,
    beta_l2: f32,
    beta_l1: f32,
    l3_persist: f32,
    l2_persist: f32,
    l1_persist: f32,
    kwta_threshold: f32,
    hierarchy_depth: f32,
    temp_annealing: f32,
    hebbian_in_settle: f32,
    hybrid_topdown: f32,
    lattice_size: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
};

@group(0) @binding(0) var<storage, read> state: array<u32>;
@group(0) @binding(1) var<storage, read_write> l3_state: array<f32>;
@group(0) @binding(2) var<storage, read_write> l2_state: array<f32>;
@group(0) @binding(3) var<storage, read_write> l1_state: array<f32>;
@group(0) @binding(4) var<uniform> params: PhysicsParams;

// =============================================================================
// L4 → L3 pooling (16x16 block → 1 L3 cell, sqrt-density)
// L3 is 256x256 for a 4096x4096 L4
// =============================================================================
@compute @workgroup_size(256)
fn l4_to_l3_pool(@builtin(global_invocation_id) gid: vec3<u32>) {
    let l3_idx = gid.x;
    let size = u32(params.lattice_size);
    let l3_size = size / 16u;  // 256 for 4096
    let total = l3_size * l3_size;
    if (l3_idx >= total) { return; }

    let ly = l3_idx / l3_size;
    let lx = l3_idx % l3_size;
    let start_y = ly * 16u;
    let start_x = lx * 16u;

    var count_on: u32 = 0u;
    for (var dy: u32 = 0u; dy < 16u; dy++) {
        for (var dx: u32 = 0u; dx < 16u; dx++) {
            let gy = start_y + dy;
            let gx = start_x + dx;
            if (gy < size && gx < size) {
                let idx = gy * size + gx;
                let s4 = (state[idx] >> 16u) & 0x3u;
                if (s4 > 0u) { count_on++; }
            }
        }
    }

    let density = sqrt(f32(count_on) / 256.0);
    let old_val = l3_state[l3_idx];
    l3_state[l3_idx] = old_val * params.l3_persist + density * params.alpha_l3;
}

// =============================================================================
// L3 → L2 pooling (4x4 block → 1 L2 cell, mean)
// L2 is 64x64
// =============================================================================
@compute @workgroup_size(256)
fn l3_to_l2_pool(@builtin(global_invocation_id) gid: vec3<u32>) {
    let l2_idx = gid.x;
    let size = u32(params.lattice_size);
    let l3_size = size / 16u;  // 256
    let l2_size = size / 64u;  // 64
    let total = l2_size * l2_size;
    if (l2_idx >= total) { return; }

    let ly = l2_idx / l2_size;
    let lx = l2_idx % l2_size;

    var sum: f32 = 0.0;
    for (var dy: u32 = 0u; dy < 4u; dy++) {
        for (var dx: u32 = 0u; dx < 4u; dx++) {
            sum += l3_state[(ly * 4u + dy) * l3_size + (lx * 4u + dx)];
        }
    }

    let old_val = l2_state[l2_idx];
    l2_state[l2_idx] = old_val * params.l2_persist + (sum / 16.0) * params.alpha_l2;
}

// =============================================================================
// L2 → L1 pooling (4x4 block → 1 L1 cell, mean)
// L1 is 16x16
// =============================================================================
@compute @workgroup_size(256)
fn l2_to_l1_pool(@builtin(global_invocation_id) gid: vec3<u32>) {
    let l1_idx = gid.x;
    let size = u32(params.lattice_size);
    let l2_size = size / 64u;  // 64
    let l1_size = 16u;
    let total = l1_size * l1_size;
    if (l1_idx >= total) { return; }

    let ly = l1_idx / l1_size;
    let lx = l1_idx % l1_size;

    var sum: f32 = 0.0;
    for (var dy: u32 = 0u; dy < 4u; dy++) {
        for (var dx: u32 = 0u; dx < 4u; dx++) {
            sum += l2_state[(ly * 4u + dy) * l2_size + (lx * 4u + dx)];
        }
    }

    let old_val = l1_state[l1_idx];
    l1_state[l1_idx] = old_val * params.l1_persist + (sum / 16.0) * params.alpha_l1;
}

// =============================================================================
// L3 → L4 top-down (only boosts already-active neurons)
// =============================================================================
// NOTE: This shader needs read_write access to state, so it uses a separate
// bind group from the pooling shaders. The engine binds it differently.
// For now, top-down is deferred to the GPU pipeline wiring phase.

// =============================================================================
// L2 → L3 top-down
// =============================================================================
@compute @workgroup_size(256)
fn l2_to_l3_topdown(@builtin(global_invocation_id) gid: vec3<u32>) {
    let l3_idx = gid.x;
    let size = u32(params.lattice_size);
    let l3_size = size / 16u;
    let l2_size = size / 64u;
    let total = l3_size * l3_size;
    if (l3_idx >= total) { return; }

    let ly = l3_idx / l3_size;
    let lx = l3_idx % l3_size;
    let l2_y = ly / 4u;
    let l2_x = lx / 4u;
    let l2_val = l2_state[l2_y * l2_size + l2_x];

    l3_state[l3_idx] += l2_val * params.beta_l3;
}

// =============================================================================
// L1 → L2 top-down
// =============================================================================
@compute @workgroup_size(256)
fn l1_to_l2_topdown(@builtin(global_invocation_id) gid: vec3<u32>) {
    let l2_idx = gid.x;
    let size = u32(params.lattice_size);
    let l2_size = size / 64u;
    let l1_size = 16u;
    let total = l2_size * l2_size;
    if (l2_idx >= total) { return; }

    let ly = l2_idx / l2_size;
    let lx = l2_idx % l2_size;
    let l1_y = ly / 4u;
    let l1_x = lx / 4u;
    let l1_val = l1_state[l1_y * l1_size + l1_x];

    l2_state[l2_idx] += l1_val * params.beta_l2;
}
