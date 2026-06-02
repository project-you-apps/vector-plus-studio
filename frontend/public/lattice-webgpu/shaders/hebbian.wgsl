// LatticeRunner WebGPU — Hebbian Learning Compute Shader
// Ports kernel_hebbian_learning from lattice_cuda_v7.cu
//
// Rule: If neuron i is active AND neighbor j is active,
//       strengthen the weight w[i→j] by hebb_step (capped at 127).
// Only runs on rows where ROW_SKIP_LEARNING (0x40) is NOT set.
//
// Weights packed as uint32: N(7:0) | E(15:8) | S(23:16) | W(31:24)

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
@group(0) @binding(1) var<storage, read_write> weights: array<u32>;
@group(0) @binding(2) var<uniform> params: PhysicsParams;
@group(0) @binding(3) var<storage, read> row_flags: array<u32>;

// Unpack current state bit
fn unpack_s4_cur(packed: u32) -> u32 { return (packed >> 16u) & 0x3u; }

// Get row physics flags
fn get_row_flags(region_row: u32) -> u32 {
    let word = region_row / 4u;
    let byte_pos = region_row % 4u;
    return (row_flags[word] >> (byte_pos * 8u)) & 0xFFu;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let size = u32(params.lattice_size);
    let total = size * size;

    if (idx >= total) { return; }

    let row = idx / size;
    let col = idx % size;
    let region_row = row / 64u;
    let flags = get_row_flags(region_row);

    // Skip if learning disabled for this row
    if ((flags & 0x40u) != 0u) { return; }

    // Only active neurons learn
    let packed = state[idx];
    let s4 = unpack_s4_cur(packed);
    if (s4 == 0u) { return; }

    let hebb_step = u32(params.hebbian_rate * 16.0);  // Match CUDA: hebb_step = rate * 16
    var w = weights[idx];

    // Unpack 4 directional weights
    var wN = w & 0xFFu;
    var wE = (w >> 8u) & 0xFFu;
    var wS = (w >> 16u) & 0xFFu;
    var wW = (w >> 24u) & 0xFFu;

    // North neighbor
    if (row > 0u) {
        let nIdx = (row - 1u) * size + col;
        if (unpack_s4_cur(state[nIdx]) > 0u && wN < 127u) {
            wN = min(wN + hebb_step, 127u);
        }
    }

    // East neighbor
    if (col < size - 1u) {
        let nIdx = row * size + col + 1u;
        if (unpack_s4_cur(state[nIdx]) > 0u && wE < 127u) {
            wE = min(wE + hebb_step, 127u);
        }
    }

    // South neighbor
    if (row < size - 1u) {
        let nIdx = (row + 1u) * size + col;
        if (unpack_s4_cur(state[nIdx]) > 0u && wS < 127u) {
            wS = min(wS + hebb_step, 127u);
        }
    }

    // West neighbor
    if (col > 0u) {
        let nIdx = row * size + col - 1u;
        if (unpack_s4_cur(state[nIdx]) > 0u && wW < 127u) {
            wW = min(wW + hebb_step, 127u);
        }
    }

    // Pack and write back
    weights[idx] = (wN & 0xFFu) | ((wE & 0xFFu) << 8u) | ((wS & 0xFFu) << 16u) | ((wW & 0xFFu) << 24u);
}
