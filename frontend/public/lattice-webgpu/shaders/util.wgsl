// LatticeRunner WebGPU — Utility Compute Shaders
// State swap, reset, signature generation

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

@group(0) @binding(0) var<storage, read_write> state: array<u32>;
@group(0) @binding(1) var<uniform> params: PhysicsParams;

// =============================================================================
// SWAP STATES — ping-pong s4_current <-> s4_next
// =============================================================================
@compute @workgroup_size(256)
fn swap_states(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let size = u32(params.lattice_size);
    let total = size * size;
    if (idx >= total) { return; }

    let packed = state[idx];
    let s4_cur = (packed >> 16u) & 0x3u;
    let s4_next = (packed >> 18u) & 0x3u;

    // Swap: current becomes next, next becomes current
    let swapped = (packed & 0xFFF0FFFFu) | (s4_next << 16u) | (s4_cur << 18u);
    state[idx] = swapped;
}

// =============================================================================
// RESET LATTICE — zero all neurons
// =============================================================================
@compute @workgroup_size(256)
fn reset_lattice(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let size = u32(params.lattice_size);
    let total = size * size;
    if (idx >= total) { return; }

    state[idx] = 0u;
}
