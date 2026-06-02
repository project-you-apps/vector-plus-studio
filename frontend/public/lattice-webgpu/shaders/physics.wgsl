// LatticeRunner WebGPU — V7 Physics Frame Compute Shader
// Ports kernel_l4_physics_frame_v7 from lattice_cuda_v7.cu
//
// Neuron packing (32-bit):
//   bits  0-7:  e4 (energy, uint8)
//   bits  8-15: f4 (fatigue, uint8)
//   bits 16-17: s4_current (state, 2-bit)
//   bits 18-19: s4_next (ping-pong, 2-bit)
//   bits 20-31: reserved

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

struct FrameInfo {
    frame: u32,
    total_frames: u32,
    learn: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> state: array<u32>;
@group(0) @binding(1) var<storage, read_write> weights: array<u32>;
@group(0) @binding(2) var<uniform> params: PhysicsParams;
@group(0) @binding(3) var<uniform> frame_info: FrameInfo;
@group(0) @binding(4) var<storage, read> sigmoid_lut: array<u32>;
@group(0) @binding(5) var<storage, read> row_flags: array<u32>;

// Unpack helpers
fn unpack_e4(packed: u32) -> u32 { return packed & 0xFFu; }
fn unpack_f4(packed: u32) -> u32 { return (packed >> 8u) & 0xFFu; }
fn unpack_s4_cur(packed: u32) -> u32 { return (packed >> 16u) & 0x3u; }
fn unpack_s4_next(packed: u32) -> u32 { return (packed >> 18u) & 0x3u; }

// Pack helpers
fn pack_neuron(e4: u32, f4: u32, s4_cur: u32, s4_next: u32, rest: u32) -> u32 {
    return (e4 & 0xFFu) | ((f4 & 0xFFu) << 8u) | ((s4_cur & 0x3u) << 16u) | ((s4_next & 0x3u) << 18u) | (rest & 0xFFF00000u);
}

// Get row physics flags (packed as 4 uint8 per uint32)
fn get_row_flags(region_row: u32) -> u32 {
    let word = region_row / 4u;
    let byte_pos = region_row % 4u;
    return (row_flags[word] >> (byte_pos * 8u)) & 0xFFu;
}

// Deterministic PRNG (matches CUDA)
fn det_rng(idx: u32, frame: u32) -> u32 {
    return ((idx * frame * 1103515245u) + 12345u) & 0xFFu;
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

    // Fast path: fully protected row
    if (flags == 0xFFu) { return; }

    let packed = state[idx];
    var e4 = i32(unpack_e4(packed));
    var f4 = i32(unpack_f4(packed));
    let s4 = unpack_s4_cur(packed);

    // Q8 conversions
    let persist_q8 = i32(params.energy_persist * 255.0);
    let fatigue_q8 = i32(params.fatigue_rate * 255.0);
    let recovery_q8 = i32(params.fatigue_recovery * 255.0);
    let facil_q8 = i32(params.facilitation * 255.0);
    let inhib_q8 = i32(params.inhibition * 255.0);
    let antifacil_q8 = i32(params.facilitation * params.anti_facilitation_mult * 255.0);

    // Step 1: Energy decay
    if ((flags & 0x01u) == 0u) {
        e4 = (e4 * persist_q8) >> 8;
    }

    // Step 2: Fatigue penalty
    if ((flags & 0x02u) == 0u) {
        e4 = max(0, e4 - ((f4 * fatigue_q8) >> 8));
    }

    // Step 3: Mexican hat lateral inhibition (7x7 neighborhood)
    if ((flags & 0x04u) == 0u) {
        var lateral: i32 = 0;
        for (var dy: i32 = -3; dy <= 3; dy++) {
            for (var dx: i32 = -3; dx <= 3; dx++) {
                if (dx == 0 && dy == 0) { continue; }
                let ny = i32(row) + dy;
                let nx = i32(col) + dx;
                if (ny < 0 || ny >= i32(size) || nx < 0 || nx >= i32(size)) { continue; }
                let nIdx = u32(ny) * size + u32(nx);
                let ns = unpack_s4_cur(state[nIdx]);
                let dist = abs(dx) + abs(dy);

                if (ns > 0u) {
                    if (dist == 1) {
                        lateral += facil_q8;
                    } else if (dist >= 2 && dist <= 3) {
                        lateral -= inhib_q8;
                    }
                } else {
                    if (dist == 1) {
                        lateral -= antifacil_q8;
                    }
                }
            }
        }
        e4 = clamp(e4 + lateral, 0, 255);
    }

    // Step 4: Weight influence (4 immediate neighbors)
    if ((flags & 0x08u) == 0u) {
        let w = weights[idx];
        var influence: i32 = 0;

        // North
        if (row > 0u) {
            let nIdx = (row - 1u) * size + col;
            if (unpack_s4_cur(state[nIdx]) > 0u) {
                influence += i32(w & 0xFFu);
            }
        }
        // East
        if (col < size - 1u) {
            let nIdx = row * size + col + 1u;
            if (unpack_s4_cur(state[nIdx]) > 0u) {
                influence += i32((w >> 8u) & 0xFFu);
            }
        }
        // South
        if (row < size - 1u) {
            let nIdx = (row + 1u) * size + col;
            if (unpack_s4_cur(state[nIdx]) > 0u) {
                influence += i32((w >> 16u) & 0xFFu);
            }
        }
        // West
        if (col > 0u) {
            let nIdx = row * size + col - 1u;
            if (unpack_s4_cur(state[nIdx]) > 0u) {
                influence += i32((w >> 24u) & 0xFFu);
            }
        }
        e4 = min(255, e4 + influence);
    }

    // Step 5: Boltzmann stochastic activation
    var new_s4 = s4;
    if ((flags & 0x10u) == 0u) {
        var temp = params.temperature;
        if (params.temp_annealing > 0.5) {
            let progress = f32(frame_info.frame) / f32(max(frame_info.total_frames, 1u));
            temp *= (1.0 - progress * 0.5);
            temp = max(temp, 5.0);
        }
        let effective = f32(e4) - 128.0;
        let temp_scale = 128.0 / temp;
        var lut_idx = i32(effective * temp_scale) + 128;
        lut_idx = clamp(lut_idx, 0, 255);
        let prob = sigmoid_lut[u32(lut_idx)];
        let rand_val = det_rng(idx, frame_info.frame + 1u);
        if (rand_val < prob) {
            new_s4 = 1u;
        } else {
            new_s4 = 0u;
        }
    }

    // Step 6: Fatigue update
    if ((flags & 0x02u) == 0u) {
        if (new_s4 > 0u) {
            f4 = min(f4 + 30, 200);
        } else {
            f4 = (f4 * recovery_q8) >> 8;
        }
    }

    // Write packed result (preserve upper bits)
    state[idx] = pack_neuron(u32(clamp(e4, 0, 255)), u32(clamp(f4, 0, 255)), s4, new_s4, packed);
}
