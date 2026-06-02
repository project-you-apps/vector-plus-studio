// LatticeRunner WebGPU — kWTA Regional Sparsity Compute Shaders
// Three-pass approach: clear → histogram → apply threshold
//
// Each region is 64x64 = 4096 neurons.
// kWTA keeps only the top K active neurons per region by energy.

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
@group(0) @binding(2) var<storage, read_write> kwta_data: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> row_flags: array<u32>;

fn get_row_flags(region_row: u32) -> u32 {
    let word = region_row / 4u;
    let byte_pos = region_row % 4u;
    return (row_flags[word] >> (byte_pos * 8u)) & 0xFFu;
}

// Per-region layout in kwta_data: 258 u32 values
// [0..255] = histogram bins (count of neurons at each energy level)
// [256]    = total_on count
// [257]    = energy threshold (set by find_threshold pass)
const KWTA_STRIDE: u32 = 258u;

// =============================================================================
// PASS 1: Clear histogram
// =============================================================================
@compute @workgroup_size(256)
fn kwta_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let size = u32(params.lattice_size);
    let num_regions = size / 64u;
    let total_entries = num_regions * num_regions * KWTA_STRIDE;
    if (idx >= total_entries) { return; }
    atomicStore(&kwta_data[idx], 0u);
}

// =============================================================================
// PASS 2: Build histogram of active neuron energies per region
// =============================================================================
@compute @workgroup_size(256)
fn kwta_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let size = u32(params.lattice_size);
    let total = size * size;
    if (idx >= total) { return; }

    let row = idx / size;
    let col = idx % size;
    let region_row = row / 64u;
    let region_col = col / 64u;

    // Skip if this row has kWTA disabled
    if ((get_row_flags(region_row) & 0x20u) != 0u) { return; }

    let packed = state[idx];
    let s4_next = (packed >> 18u) & 0x3u;

    if (s4_next > 0u) {
        let e4 = packed & 0xFFu;
        let num_regions = size / 64u;
        let region_idx = region_row * num_regions + region_col;
        let base = region_idx * KWTA_STRIDE;

        atomicAdd(&kwta_data[base + e4], 1u);
        atomicAdd(&kwta_data[base + 256u], 1u); // total_on
    }
}

// =============================================================================
// PASS 3: Find threshold energy per region
// One workgroup per region.
// =============================================================================
@compute @workgroup_size(1)
fn kwta_find_threshold(@builtin(global_invocation_id) gid: vec3<u32>) {
    let region_idx = gid.x;
    let size = u32(params.lattice_size);
    let num_regions = size / 64u;
    let total_regions = num_regions * num_regions;
    if (region_idx >= total_regions) { return; }

    let base = region_idx * KWTA_STRIDE;
    let total_on = atomicLoad(&kwta_data[base + 256u]);
    let max_on = u32(4096.0 * params.kwta_threshold);

    if (total_on <= max_on) {
        // No thresholding needed — store 0
        atomicStore(&kwta_data[base + 257u], 0u);
        return;
    }

    // Scan histogram from high energy down to find threshold
    var cumsum: u32 = 0u;
    var threshold: u32 = 0u;
    for (var e: i32 = 255; e >= 0; e--) {
        let count = atomicLoad(&kwta_data[base + u32(e)]);
        if (cumsum + count >= max_on) {
            threshold = u32(e);
            break;
        }
        cumsum += count;
    }
    atomicStore(&kwta_data[base + 257u], threshold);
}

// =============================================================================
// PASS 4: Apply threshold — turn off neurons below threshold
// =============================================================================
@compute @workgroup_size(256)
fn kwta_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let size = u32(params.lattice_size);
    let total = size * size;
    if (idx >= total) { return; }

    let row = idx / size;
    let col = idx % size;
    let region_row = row / 64u;
    let region_col = col / 64u;

    if ((get_row_flags(region_row) & 0x20u) != 0u) { return; }

    let packed = state[idx];
    let s4_next = (packed >> 18u) & 0x3u;

    if (s4_next > 0u) {
        let e4 = packed & 0xFFu;
        let num_regions = size / 64u;
        let region_idx = region_row * num_regions + region_col;
        let base = region_idx * KWTA_STRIDE;
        let threshold = atomicLoad(&kwta_data[base + 257u]);

        if (e4 < threshold) {
            // Turn off: clear s4_next
            state[idx] = packed & 0xFFF3FFFFu;
        }
        // Tie-breaking via spatial hash (simplified from CUDA)
        else if (e4 == threshold) {
            let local_row = row % 64u;
            let local_col = col % 64u;
            var hash = (local_row * 7919u) ^ (local_col * 6971u);
            hash = (hash * 2654435761u) >> 16u;
            let total_on = atomicLoad(&kwta_data[base + 256u]);
            let max_on = u32(4096.0 * params.kwta_threshold);
            let at_threshold = atomicLoad(&kwta_data[base + e4]);

            if (total_on > max_on && at_threshold > 0u) {
                let to_remove = total_on - max_on;
                let remove_prob = f32(to_remove) / f32(at_threshold);
                let hash_norm = f32(hash & 0xFFFFu) / 65536.0;
                if (hash_norm < remove_prob) {
                    state[idx] = packed & 0xFFF3FFFFu;
                }
            }
        }
    }
}
