/**
 * LatticeRunner WebGPU — V7-compatible lattice physics engine
 * Full port of lattice_cuda_v7.cu to WebGPU compute shaders.
 *
 * Hierarchy: L1(16x16) | L2(64x64) | L3(256x256) | L4(4096x4096)
 * Neuron packing: 32-bit word (e4:8, f4:8, s4_cur:2, s4_next:2, weights:12)
 * Separate weight buffer: 4 x uint8 per neuron (N,E,S,W)
 */

// Physics profile presets (matching CUDA BALANCED defaults)
const PROFILES = {
    fast: {
        energy_persist: 0.96, fatigue_rate: 0.2, fatigue_recovery: 0.94,
        temperature: 14.0, facilitation: 0.27, inhibition: 0.055,
        anti_facilitation_mult: 0.0, hebbian_rate: 0.25,
        alpha_l4: 0.86, alpha_l3: 0.5, alpha_l2: 0.5, alpha_l1: 0.5,
        beta_l4: 1.0, beta_l3: 0.7, beta_l2: 0.6, beta_l1: 0.5,
        l3_persist: 0.1, l2_persist: 0.2, l1_persist: 0.2,
        kwta_threshold: 0.46, hierarchy_depth: 2,
        temp_annealing: 0, hebbian_in_settle: 0, hybrid_topdown: 0
    },
    balanced: {
        energy_persist: 0.96, fatigue_rate: 0.2, fatigue_recovery: 0.94,
        temperature: 14.0, facilitation: 0.27, inhibition: 0.055,
        anti_facilitation_mult: 0.5, hebbian_rate: 0.25,
        alpha_l4: 0.86, alpha_l3: 0.5, alpha_l2: 0.5, alpha_l1: 0.5,
        beta_l4: 1.0, beta_l3: 0.7, beta_l2: 0.6, beta_l1: 0.5,
        l3_persist: 0.1, l2_persist: 0.2, l1_persist: 0.2,
        kwta_threshold: 0.42, hierarchy_depth: 2,
        temp_annealing: 1, hebbian_in_settle: 1, hybrid_topdown: 1
    },
    quality: {
        energy_persist: 0.96, fatigue_rate: 0.2, fatigue_recovery: 0.94,
        temperature: 14.0, facilitation: 0.27, inhibition: 0.055,
        anti_facilitation_mult: 1.0, hebbian_rate: 0.25,
        alpha_l4: 0.86, alpha_l3: 0.5, alpha_l2: 0.5, alpha_l1: 0.5,
        beta_l4: 1.0, beta_l3: 0.7, beta_l2: 0.6, beta_l1: 0.5,
        l3_persist: 0.1, l2_persist: 0.2, l1_persist: 0.2,
        kwta_threshold: 0.35, hierarchy_depth: 4,
        temp_annealing: 1, hebbian_in_settle: 1, hybrid_topdown: 1
    }
};

// Per-row physics skip flags
const ROW_SKIP_DECAY      = 0x01;
const ROW_SKIP_FATIGUE    = 0x02;
const ROW_SKIP_INHIBITION = 0x04;
const ROW_SKIP_WEIGHTS    = 0x08;
const ROW_SKIP_BOLTZMANN  = 0x10;
const ROW_SKIP_KWTA       = 0x20;
const ROW_SKIP_LEARNING   = 0x40;
const ROW_SKIP_TOPDOWN    = 0x80;
const ROW_FULLY_PROTECTED = 0xFF;
const ROW_LEARN_ONLY      = 0xBF;

// Hippocampus epoch (2025-01-01 00:00 UTC)
const HIPPO_EPOCH = 1735689600;

export class LatticeEngine {
    constructor(latticeSize = 4096) {
        this.size = latticeSize;
        this.numNeurons = latticeSize * latticeSize;
        this.numRegions = latticeSize / 64;
        this.device = null;
        this.ready = false;
        this.profile = 'balanced';
        this.physics = { ...PROFILES.balanced };

        // GPU buffers (created in init())
        this.buffers = {};
        this.pipelines = {};

        // Per-row physics flags (64 rows, default 0 = full physics)
        this.rowPhysics = new Uint8Array(64);

        // Protected rows mask (64-bit as two 32-bit values)
        this.protectedRowsLo = 0;
        this.protectedRowsHi = 0;

        // Sigmoid LUT (precomputed)
        this.sigmoidLUT = new Uint8Array(256);
        this._buildSigmoidLUT(this.physics.temperature);
    }

    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    async init() {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported in this browser');
        }

        const adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance'
        });
        if (!adapter) {
            throw new Error('No WebGPU adapter found');
        }

        // Request device with maximum buffer size
        const limits = adapter.limits;
        this.device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
                maxBufferSize: limits.maxBufferSize,
                maxComputeWorkgroupSizeX: 256,
                maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension
            }
        });

        this.device.lost.then((info) => {
            console.error('WebGPU device lost:', info.message);
            this.ready = false;
        });

        await this._createBuffers();
        await this._createPipelines();

        this.ready = true;
        console.log(`[LatticeEngine] Initialized: ${this.size}x${this.size} (${this.numNeurons} neurons, ${this.numRegions}x${this.numRegions} regions)`);
        return this;
    }

    async _createBuffers() {
        const N = this.numNeurons;
        const dev = this.device;

        // L4 state: packed 32-bit neurons (e4:8, f4:8, s4_cur:2, s4_next:2, w_packed:12)
        this.buffers.state = dev.createBuffer({
            size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: 'L4 state'
        });

        // L4 weights: 4 x uint8 packed into uint32 per neuron
        this.buffers.weights = dev.createBuffer({
            size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: 'L4 weights'
        });

        // L3 state (256x256 floats)
        const l3Size = 256 * 256;
        this.buffers.l3State = dev.createBuffer({
            size: l3Size * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: 'L3 state'
        });

        // L2 state (64x64 floats)
        const l2Size = 64 * 64;
        this.buffers.l2State = dev.createBuffer({
            size: l2Size * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: 'L2 state'
        });

        // L1 state (16x16 floats)
        const l1Size = 16 * 16;
        this.buffers.l1State = dev.createBuffer({
            size: l1Size * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: 'L1 state'
        });

        // Physics params uniform buffer
        this.buffers.params = dev.createBuffer({
            size: 128, // Padded to 128 bytes for alignment
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'physics params'
        });

        // Per-row physics flags (64 uint8, padded to 64 bytes = 16 uint32)
        this.buffers.rowFlags = dev.createBuffer({
            size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'row physics flags'
        });

        // Sigmoid LUT (256 uint8, padded as uint32 array for shader access)
        this.buffers.sigmoidLUT = dev.createBuffer({
            size: 256 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'sigmoid LUT'
        });

        // Frame info uniform (frame number, total frames, learn flag)
        this.buffers.frameInfo = dev.createBuffer({
            size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'frame info'
        });

        // kWTA histogram buffer (per-region: 256 bins + total_on + threshold)
        const kwtaSize = this.numRegions * this.numRegions * 258 * 4;
        this.buffers.kwtaHist = dev.createBuffer({
            size: kwtaSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'kWTA histogram'
        });

        // Signature output (numRegions^2 floats)
        const sigSize = this.numRegions * this.numRegions;
        this.buffers.signature = dev.createBuffer({
            size: sigSize * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            label: 'signature'
        });

        // Readback staging buffers
        this.buffers.readbackState = dev.createBuffer({
            size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            label: 'readback state'
        });
        this.buffers.readbackSig = dev.createBuffer({
            size: sigSize * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            label: 'readback signature'
        });
        this.buffers.readbackL3 = dev.createBuffer({
            size: l3Size * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            label: 'readback L3'
        });
        this.buffers.readbackL2 = dev.createBuffer({
            size: l2Size * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            label: 'readback L2'
        });
        this.buffers.readbackL1 = dev.createBuffer({
            size: l1Size * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            label: 'readback L1'
        });

        // Embedding input buffer (max 768 dims for Nomic)
        this.buffers.embedding = dev.createBuffer({
            size: 768 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'embedding input'
        });

        // Upload initial sigmoid LUT
        this._uploadSigmoidLUT();
        // Upload initial physics params
        this._uploadPhysicsParams();
        // Upload initial row flags
        this._uploadRowFlags();
    }

    async _createPipelines() {
        // Import shader modules
        const physicsCode = await this._loadShader('shaders/physics.wgsl');
        const kwtaCode = await this._loadShader('shaders/kwta.wgsl');
        const hierarchyCode = await this._loadShader('shaders/hierarchy.wgsl');
        const imprintCode = await this._loadShader('shaders/imprint.wgsl');
        const utilCode = await this._loadShader('shaders/util.wgsl');

        // Physics frame pipeline
        this.pipelines.physics = this._createComputePipeline('physics', physicsCode);

        // State swap pipeline
        this.pipelines.swap = this._createComputePipeline('swap', utilCode, 'swap_states');

        // kWTA pipelines (3 passes)
        this.pipelines.kwtaClear = this._createComputePipeline('kwta_clear', kwtaCode, 'kwta_clear');
        this.pipelines.kwtaHist = this._createComputePipeline('kwta_hist', kwtaCode, 'kwta_histogram');
        this.pipelines.kwtaThreshold = this._createComputePipeline('kwta_threshold', kwtaCode, 'kwta_find_threshold');
        this.pipelines.kwtaApply = this._createComputePipeline('kwta_apply', kwtaCode, 'kwta_apply');

        // Hierarchy pipelines
        this.pipelines.l4ToL3 = this._createComputePipeline('l4_to_l3', hierarchyCode, 'l4_to_l3_pool');
        this.pipelines.l3ToL2 = this._createComputePipeline('l3_to_l2', hierarchyCode, 'l3_to_l2_pool');
        this.pipelines.l2ToL1 = this._createComputePipeline('l2_to_l1', hierarchyCode, 'l2_to_l1_pool');
        // l3_to_l4_topdown deferred — needs read_write state access, separate bind group
        // this.pipelines.l3ToL4TD = this._createComputePipeline('l3_to_l4_td', hierarchyCode, 'l3_to_l4_topdown');
        this.pipelines.l2ToL3TD = this._createComputePipeline('l2_to_l3_td', hierarchyCode, 'l2_to_l3_topdown');
        this.pipelines.l1ToL2TD = this._createComputePipeline('l1_to_l2_td', hierarchyCode, 'l1_to_l2_topdown');

        // Imprint pipelines (imprint.wgsl not yet written — CPU fallback for now)
        if (imprintCode) {
            this.pipelines.imprintPattern = this._createComputePipeline('imprint_pattern', imprintCode, 'imprint_pattern');
            this.pipelines.imprintNomic = this._createComputePipeline('imprint_nomic', imprintCode, 'imprint_nomic');
        }

        // Hebbian learning (separate shader)
        const hebbianCode = await this._loadShader('shaders/hebbian.wgsl');
        if (hebbianCode) {
            this.pipelines.hebbian = this._createComputePipeline('hebbian', hebbianCode);
        }

        // Signature/reset — only create if entry points exist in util.wgsl
        this.pipelines.reset = this._createComputePipeline('reset', utilCode, 'reset_lattice');
        // signature and hippocampus_encode not yet in util.wgsl
        // this.pipelines.signature = this._createComputePipeline('signature', utilCode, 'generate_signature');
        // this.pipelines.hippoEncode = this._createComputePipeline('hippo_encode', utilCode, 'hippocampus_encode');
    }

    _createComputePipeline(name, code, entryPoint = 'main') {
        if (!code) {
            console.warn(`[LatticeEngine] No shader code for ${name}, GPU dispatch unavailable`);
            return null;
        }
        const module = this.device.createShaderModule({
            label: `${name} shader`,
            code: code
        });

        // Create pipeline with auto bind group layout
        const pipeline = this.device.createComputePipeline({
            label: `${name} pipeline`,
            layout: 'auto',
            compute: { module, entryPoint }
        });

        return { pipeline, entryPoint, name };
    }

    async _loadShader(path) {
        // In standalone mode, load from relative path
        // In production, these would be bundled
        try {
            const response = await fetch(path);
            if (!response.ok) throw new Error(`Failed to load ${path}: ${response.status}`);
            const text = await response.text();
            // Dev servers (Vite, webpack-dev-server) often serve their SPA
            // index.html as a 200 fallback for missing assets. Detect that
            // and treat as a missing shader so we don't try to compile HTML.
            const trimmed = text.trimStart();
            if (trimmed.startsWith('<!') || trimmed.startsWith('<html') || trimmed.startsWith('<!DOCTYPE')) {
                console.warn(`[LatticeEngine] ${path} returned HTML (SPA fallback); treating as missing`);
                return null;
            }
            return text;
        } catch (e) {
            console.warn(`[LatticeEngine] Could not load ${path}, using embedded shader`);
            return null;
        }
    }

    // =========================================================================
    // PHYSICS CONFIGURATION
    // =========================================================================

    setProfile(profile) {
        if (!PROFILES[profile]) throw new Error(`Unknown profile: ${profile}`);
        this.profile = profile;
        this.physics = { ...PROFILES[profile] };
        this._buildSigmoidLUT(this.physics.temperature);
        if (this.ready) {
            this._uploadPhysicsParams();
            this._uploadSigmoidLUT();
        }
    }

    setPhysics(params) {
        Object.assign(this.physics, params);
        if (params.temperature !== undefined) {
            this._buildSigmoidLUT(params.temperature);
        }
        if (this.ready) {
            this._uploadPhysicsParams();
            if (params.temperature !== undefined) this._uploadSigmoidLUT();
        }
    }

    // =========================================================================
    // CORE OPERATIONS
    // =========================================================================

    async reset() {
        // Zero only neuron state — weights survive so a loaded brain remains
        // active across multiple imprint/settle cycles (matches Python wrapper
        // semantics used by the two-sided Associate v83 loop).
        const zeros = new Uint32Array(this.numNeurons);
        this.device.queue.writeBuffer(this.buffers.state, 0, zeros);
    }

    async resetAll() {
        // Zero state AND weights — for full re-init without throwing away the device.
        const zeros = new Uint32Array(this.numNeurons);
        this.device.queue.writeBuffer(this.buffers.state, 0, zeros);
        this.device.queue.writeBuffer(this.buffers.weights, 0, zeros);
    }

    async imprintPattern(pattern) {
        // pattern: Float32Array of numNeurons values (>0.5 = ON)
        if (pattern.length !== this.numNeurons) {
            throw new Error(`Pattern size mismatch: got ${pattern.length}, expected ${this.numNeurons}`);
        }

        // Upload pattern and dispatch imprint kernel
        // For now, CPU-side imprint (GPU pipeline will be added)
        const state = new Uint32Array(this.numNeurons);
        for (let i = 0; i < this.numNeurons; i++) {
            const on = pattern[i] > 0.5 ? 1 : 0;
            const e4 = on ? 200 : 50;
            const f4 = 0;
            state[i] = e4 | (f4 << 8) | (on << 16) | (on << 18);
        }
        this.device.queue.writeBuffer(this.buffers.state, 0, state);
    }

    async imprintVector(embedding, normalize = 'auto') {
        // embedding: Float32Array of 768 values (Nomic)
        let emb = new Float32Array(embedding);

        // Auto-normalize to [0,1]
        if (normalize === 'always' || (normalize === 'auto' && this._needsNormalize(emb))) {
            const min = Math.min(...emb);
            const max = Math.max(...emb);
            const range = max - min || 1;
            emb = emb.map(v => (v - min) / range);
        }

        // GPU thermometer encoding
        // Upload embedding, dispatch imprint_nomic kernel
        this.device.queue.writeBuffer(this.buffers.embedding, 0, emb);

        // For initial version: CPU-side thermometer encoding
        await this._cpuThermometerImprint(emb);
    }

    async settle(frames = 30, learn = true) {
        if (!this.ready) throw new Error('Engine not initialized');

        const useGPU = !!this.pipelines.physics;

        for (let frame = 0; frame < frames; frame++) {
            // Upload frame info
            const frameData = new Uint32Array([frame, frames, learn ? 1 : 0, 0]);
            this.device.queue.writeBuffer(this.buffers.frameInfo, 0, frameData);

            if (useGPU) {
                // GPU path: batch all passes into a single command encoder per frame
                await this._gpuFrame(frame, frames, learn);
            } else {
                // CPU fallback path
                await this._dispatchHierarchyBottomUp();
                await this._dispatchHierarchyTopDown();
                await this._dispatchPhysics();
                await this._dispatchKWTA();
                await this._dispatchSwap();
                if (learn && this.physics.hebbian_in_settle) {
                    await this._dispatchHebbian();
                }
            }
        }

        // Wait for all GPU work to complete
        await this.device.queue.onSubmittedWorkDone();
    }

    async _gpuFrame(frame, totalFrames, learn) {
        const N = this.numNeurons;
        const size = this.size;
        const numRegions = this.numRegions;
        const totalRegions = numRegions * numRegions;
        const l3Count = (size / 16) * (size / 16);
        const l2Count = (size / 64) * (size / 64);
        const l1Count = 16 * 16;
        const kwtaEntries = totalRegions * 258;

        const encoder = this.device.createCommandEncoder();

        // --- Hierarchy bottom-up ---
        // Auto layout strips unused bindings per entry point
        if (this.pipelines.l4ToL3) {
            let pass;

            // l4_to_l3_pool uses: state(0), l3_state(1), params(4)
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.l4ToL3.pipeline);
            pass.setBindGroup(0, this.device.createBindGroup({
                layout: this.pipelines.l4ToL3.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.buffers.state } },
                    { binding: 1, resource: { buffer: this.buffers.l3State } },
                    { binding: 4, resource: { buffer: this.buffers.params } },
                ]
            }));
            pass.dispatchWorkgroups(this._workgroups(l3Count));
            pass.end();

            // l3_to_l2_pool uses: l3_state(1), l2_state(2), params(4)
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.l3ToL2.pipeline);
            pass.setBindGroup(0, this.device.createBindGroup({
                layout: this.pipelines.l3ToL2.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 1, resource: { buffer: this.buffers.l3State } },
                    { binding: 2, resource: { buffer: this.buffers.l2State } },
                    { binding: 4, resource: { buffer: this.buffers.params } },
                ]
            }));
            pass.dispatchWorkgroups(this._workgroups(l2Count));
            pass.end();

            // l2_to_l1_pool uses: l2_state(2), l1_state(3), params(4)
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.l2ToL1.pipeline);
            pass.setBindGroup(0, this.device.createBindGroup({
                layout: this.pipelines.l2ToL1.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 2, resource: { buffer: this.buffers.l2State } },
                    { binding: 3, resource: { buffer: this.buffers.l1State } },
                    { binding: 4, resource: { buffer: this.buffers.params } },
                ]
            }));
            pass.dispatchWorkgroups(this._workgroups(l1Count));
            pass.end();
        }

        // --- Hierarchy top-down ---
        if (this.pipelines.l2ToL3TD) {
            let pass;

            // l1_to_l2_topdown uses: l2_state(2), l1_state(3), params(4)
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.l1ToL2TD.pipeline);
            pass.setBindGroup(0, this.device.createBindGroup({
                layout: this.pipelines.l1ToL2TD.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 2, resource: { buffer: this.buffers.l2State } },
                    { binding: 3, resource: { buffer: this.buffers.l1State } },
                    { binding: 4, resource: { buffer: this.buffers.params } },
                ]
            }));
            pass.dispatchWorkgroups(this._workgroups(l2Count));
            pass.end();

            // l2_to_l3_topdown uses: l3_state(1), l2_state(2), params(4)
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.l2ToL3TD.pipeline);
            pass.setBindGroup(0, this.device.createBindGroup({
                layout: this.pipelines.l2ToL3TD.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 1, resource: { buffer: this.buffers.l3State } },
                    { binding: 2, resource: { buffer: this.buffers.l2State } },
                    { binding: 4, resource: { buffer: this.buffers.params } },
                ]
            }));
            pass.dispatchWorkgroups(this._workgroups(l3Count));
            pass.end();
        }

        // --- Physics ---
        {
            const bg = this.device.createBindGroup({
                layout: this.pipelines.physics.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.buffers.state } },
                    { binding: 1, resource: { buffer: this.buffers.weights } },
                    { binding: 2, resource: { buffer: this.buffers.params } },
                    { binding: 3, resource: { buffer: this.buffers.frameInfo } },
                    { binding: 4, resource: { buffer: this.buffers.sigmoidLUT } },
                    { binding: 5, resource: { buffer: this.buffers.rowFlags } },
                ]
            });
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.physics.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(this._workgroups(N));
            pass.end();
        }

        // --- kWTA (4 passes, auto layout strips unused bindings) ---
        if (this.pipelines.kwtaClear) {
            let pass;

            // kwta_clear uses: params(1), kwta_data(2)
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.kwtaClear.pipeline);
            pass.setBindGroup(0, this.device.createBindGroup({
                layout: this.pipelines.kwtaClear.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 1, resource: { buffer: this.buffers.params } },
                    { binding: 2, resource: { buffer: this.buffers.kwtaHist } },
                ]
            }));
            pass.dispatchWorkgroups(this._workgroups(kwtaEntries));
            pass.end();

            // kwta_histogram uses: state(0), params(1), kwta_data(2), row_flags(3)
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.kwtaHist.pipeline);
            pass.setBindGroup(0, this.device.createBindGroup({
                layout: this.pipelines.kwtaHist.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.buffers.state } },
                    { binding: 1, resource: { buffer: this.buffers.params } },
                    { binding: 2, resource: { buffer: this.buffers.kwtaHist } },
                    { binding: 3, resource: { buffer: this.buffers.rowFlags } },
                ]
            }));
            pass.dispatchWorkgroups(this._workgroups(N));
            pass.end();

            // kwta_find_threshold uses: params(1), kwta_data(2)
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.kwtaThreshold.pipeline);
            pass.setBindGroup(0, this.device.createBindGroup({
                layout: this.pipelines.kwtaThreshold.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 1, resource: { buffer: this.buffers.params } },
                    { binding: 2, resource: { buffer: this.buffers.kwtaHist } },
                ]
            }));
            pass.dispatchWorkgroups(this._workgroups(totalRegions));
            pass.end();

            // kwta_apply uses: state(0), params(1), kwta_data(2), row_flags(3)
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.kwtaApply.pipeline);
            pass.setBindGroup(0, this.device.createBindGroup({
                layout: this.pipelines.kwtaApply.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.buffers.state } },
                    { binding: 1, resource: { buffer: this.buffers.params } },
                    { binding: 2, resource: { buffer: this.buffers.kwtaHist } },
                    { binding: 3, resource: { buffer: this.buffers.rowFlags } },
                ]
            }));
            pass.dispatchWorkgroups(this._workgroups(N));
            pass.end();
        }

        // --- State swap ---
        if (this.pipelines.swap) {
            const bg = this.device.createBindGroup({
                layout: this.pipelines.swap.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.buffers.state } },
                    { binding: 1, resource: { buffer: this.buffers.params } },
                ]
            });
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.swap.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(this._workgroups(N));
            pass.end();
        }

        // --- Hebbian learning (if learn=true and enabled) ---
        if (learn && this.physics.hebbian_in_settle && this.pipelines.hebbian) {
            const bg = this.device.createBindGroup({
                layout: this.pipelines.hebbian.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.buffers.state } },
                    { binding: 1, resource: { buffer: this.buffers.weights } },
                    { binding: 2, resource: { buffer: this.buffers.params } },
                    { binding: 3, resource: { buffer: this.buffers.rowFlags } },
                ]
            });
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.hebbian.pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(this._workgroups(N));
            pass.end();
        }

        // Submit entire frame as one command buffer
        this.device.queue.submit([encoder.finish()]);
    }

    async recall() {
        // Copy state buffer to readback, map, extract s4_current
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.buffers.state, 0, this.buffers.readbackState, 0, this.numNeurons * 4);
        this.device.queue.submit([encoder.finish()]);

        await this.buffers.readbackState.mapAsync(GPUMapMode.READ);
        const data = new Uint32Array(this.buffers.readbackState.getMappedRange());
        const result = new Float32Array(this.numNeurons);
        for (let i = 0; i < this.numNeurons; i++) {
            result[i] = ((data[i] >> 16) & 0x3) > 0 ? 1.0 : 0.0;
        }
        this.buffers.readbackState.unmap();
        return result;
    }

    async recallL3() {
        const size = 256 * 256;
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.buffers.l3State, 0, this.buffers.readbackL3, 0, size * 4);
        this.device.queue.submit([encoder.finish()]);

        await this.buffers.readbackL3.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(this.buffers.readbackL3.getMappedRange().slice(0));
        this.buffers.readbackL3.unmap();
        return result;
    }

    async recallL2() {
        const size = 64 * 64;
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.buffers.l2State, 0, this.buffers.readbackL2, 0, size * 4);
        this.device.queue.submit([encoder.finish()]);

        await this.buffers.readbackL2.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(this.buffers.readbackL2.getMappedRange().slice(0));
        this.buffers.readbackL2.unmap();
        return result;
    }

    async recallL1() {
        const size = 16 * 16;
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.buffers.l1State, 0, this.buffers.readbackL1, 0, size * 4);
        this.device.queue.submit([encoder.finish()]);

        await this.buffers.readbackL1.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(this.buffers.readbackL1.getMappedRange().slice(0));
        this.buffers.readbackL1.unmap();
        return result;
    }

    async generateSignature() {
        // Dispatch signature generation kernel
        await this._dispatchSignature();

        const sigSize = this.numRegions * this.numRegions;
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.buffers.signature, 0, this.buffers.readbackSig, 0, sigSize * 4);
        this.device.queue.submit([encoder.finish()]);

        await this.buffers.readbackSig.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(this.buffers.readbackSig.getMappedRange().slice(0));
        this.buffers.readbackSig.unmap();
        return result;
    }

    // =========================================================================
    // PROTECTED ROWS / PER-ROW PHYSICS
    // =========================================================================

    setProtectedRows(rows) {
        this.protectedRowsLo = 0;
        this.protectedRowsHi = 0;
        for (const row of rows) {
            if (row < 0 || row > 63) throw new Error(`Row ${row} out of range [0,63]`);
            if (row < 32) this.protectedRowsLo |= (1 << row);
            else this.protectedRowsHi |= (1 << (row - 32));
            this.rowPhysics[row] = ROW_FULLY_PROTECTED;
        }
        if (this.ready) this._uploadRowFlags();
    }

    setRowPhysics(row, flags) {
        if (row < 0 || row > 63) throw new Error(`Row ${row} out of range [0,63]`);
        this.rowPhysics[row] = flags & 0xFF;
        if (this.ready) this._uploadRowFlags();
    }

    setAllRowPhysics(flagsList) {
        if (flagsList.length !== 64) throw new Error('Must provide exactly 64 flags');
        for (let i = 0; i < 64; i++) this.rowPhysics[i] = flagsList[i] & 0xFF;
        if (this.ready) this._uploadRowFlags();
    }

    getRowPhysics(row) {
        return this.rowPhysics[row];
    }

    describeRowPhysics(row) {
        const f = this.rowPhysics[row];
        return {
            energy_decay: !(f & ROW_SKIP_DECAY),
            fatigue: !(f & ROW_SKIP_FATIGUE),
            inhibition: !(f & ROW_SKIP_INHIBITION),
            weight_readout: !(f & ROW_SKIP_WEIGHTS),
            boltzmann: !(f & ROW_SKIP_BOLTZMANN),
            kwta: !(f & ROW_SKIP_KWTA),
            learning: !(f & ROW_SKIP_LEARNING),
            topdown: !(f & ROW_SKIP_TOPDOWN)
        };
    }

    // =========================================================================
    // HIPPOCAMPUS
    // =========================================================================

    async encodeHippocampus(patternId, isDeleted = false, timestamp = null, flags = 0) {
        const ts = timestamp || Math.floor(Date.now() / 1000);
        const tsMinutes = Math.min(Math.floor((ts - HIPPO_EPOCH) / 60), (1 << 28) - 1);
        flags = flags & 0x07;

        // CPU-side hippocampus encoding into state buffer
        // Row 63: regions 0-31 = pattern_id, 32-59 = timestamp, 60 = tombstone, 61-63 = flags
        await this._cpuHippocampusEncode(patternId, tsMinutes, isDeleted, flags);
    }

    async decodeHippocampus(latticeVector = null) {
        const state = latticeVector || await this.recall();
        const size = this.size;

        // Read bits from row 63 regions
        let patternId = 0;
        let tsMinutes = 0;
        let isDeleted = false;
        let flags = 0;

        for (let region = 0; region < 64; region++) {
            // Calculate mean activation of this region in row 63
            const regionRow = 63;
            const startRow = regionRow * 64;
            const startCol = region * 64;
            let sum = 0;
            for (let r = 0; r < 64; r++) {
                for (let c = 0; c < 64; c++) {
                    const idx = (startRow + r) * size + (startCol + c);
                    sum += state[idx];
                }
            }
            const bit = (sum / 4096) > 0.5 ? 1 : 0;

            if (region < 32) {
                patternId |= (bit << region);
            } else if (region < 60) {
                tsMinutes |= (bit << (region - 32));
            } else if (region === 60) {
                isDeleted = bit === 1;
            } else {
                flags |= (bit << (region - 61));
            }
        }

        const timestamp = tsMinutes > 0 ? HIPPO_EPOCH + tsMinutes * 60 : 0;
        return { pattern_id: patternId, timestamp, is_deleted: isDeleted, flags };
    }

    // =========================================================================
    // PERSISTENCE
    // =========================================================================

    async saveBrain() {
        // Returns { state: Uint32Array, weights: Uint32Array }
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.buffers.state, 0, this.buffers.readbackState, 0, this.numNeurons * 4);
        this.device.queue.submit([encoder.finish()]);

        await this.buffers.readbackState.mapAsync(GPUMapMode.READ);
        const state = new Uint32Array(this.buffers.readbackState.getMappedRange().slice(0));
        this.buffers.readbackState.unmap();

        // Read weights too
        const wReadback = this.device.createBuffer({
            size: this.numNeurons * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        const enc2 = this.device.createCommandEncoder();
        enc2.copyBufferToBuffer(this.buffers.weights, 0, wReadback, 0, this.numNeurons * 4);
        this.device.queue.submit([enc2.finish()]);

        await wReadback.mapAsync(GPUMapMode.READ);
        const weights = new Uint32Array(wReadback.getMappedRange().slice(0));
        wReadback.unmap();
        wReadback.destroy();

        return { state, weights };
    }

    async loadBrain(state, weights) {
        this.device.queue.writeBuffer(this.buffers.state, 0, state);
        this.device.queue.writeBuffer(this.buffers.weights, 0, weights);
    }

    // =========================================================================
    // INTERNAL: Upload helpers
    // =========================================================================

    _buildSigmoidLUT(temperature) {
        for (let i = 0; i < 256; i++) {
            const energy = i - 128;
            const prob = 1.0 / (1.0 + Math.exp(-energy / (temperature + 1e-5)));
            this.sigmoidLUT[i] = Math.round(prob * 255);
        }
    }

    _uploadSigmoidLUT() {
        // Expand uint8 to uint32 for shader access
        const expanded = new Uint32Array(256);
        for (let i = 0; i < 256; i++) expanded[i] = this.sigmoidLUT[i];
        this.device.queue.writeBuffer(this.buffers.sigmoidLUT, 0, expanded);
    }

    _uploadPhysicsParams() {
        const p = this.physics;
        // Pack into Q8 format matching CUDA InternalPhysics
        const data = new Float32Array([
            p.energy_persist, p.fatigue_rate, p.fatigue_recovery, p.temperature,
            p.facilitation, p.inhibition, p.anti_facilitation_mult, p.hebbian_rate,
            p.alpha_l4, p.alpha_l3, p.alpha_l2, p.alpha_l1,
            p.beta_l4, p.beta_l3, p.beta_l2, p.beta_l1,
            p.l3_persist, p.l2_persist, p.l1_persist, p.kwta_threshold,
            p.hierarchy_depth, p.temp_annealing, p.hebbian_in_settle, p.hybrid_topdown,
            this.size, 0, 0, 0  // size + padding to 112 bytes (28 floats)
        ]);
        this.device.queue.writeBuffer(this.buffers.params, 0, data);
    }

    _uploadRowFlags() {
        // Pad to 64 bytes (16 uint32 values, each holding 4 uint8 flags)
        const packed = new Uint32Array(16);
        for (let i = 0; i < 64; i++) {
            const word = Math.floor(i / 4);
            const byte = i % 4;
            packed[word] |= (this.rowPhysics[i] << (byte * 8));
        }
        this.device.queue.writeBuffer(this.buffers.rowFlags, 0, packed);
    }

    _needsNormalize(emb) {
        for (let i = 0; i < emb.length; i++) {
            if (emb[i] < -0.1 || emb[i] > 1.1) return true;
        }
        return false;
    }

    // =========================================================================
    // INTERNAL: CPU-side fallbacks (until GPU shaders are fully wired)
    // =========================================================================

    async _cpuThermometerImprint(embedding) {
        const size = this.size;
        const nDims = embedding.length; // typically 768
        const state = new Uint32Array(this.numNeurons);
        const PHI = 2654435769;

        for (let dim = 0; dim < nDims; dim++) {
            const val = Math.max(0, Math.min(1, embedding[dim]));
            const dimHash = (dim * PHI) >>> 0;
            const baseRegion = dimHash % 64;
            const valueRow = Math.min(Math.floor(val * 64), 63);

            const regionRow = valueRow;
            const regionCol = baseRegion;
            const startRow = regionRow * 64;
            const startCol = regionCol * 64;

            const activePixels = Math.min(Math.floor(val * 4096), 4096);

            for (let r = 0; r < 64; r++) {
                const pixelsBefore = r * 64;
                const remaining = activePixels - pixelsBefore;
                const fillInRow = Math.max(0, Math.min(remaining, 64));

                for (let c = 0; c < 64; c++) {
                    const globalRow = startRow + r;
                    const globalCol = startCol + c;
                    const idx = globalRow * size + globalCol;

                    if (c < fillInRow) {
                        // ON: e4=200, f4=0, s4_cur=1, s4_next=1
                        state[idx] = 200 | (0 << 8) | (1 << 16) | (1 << 18);
                    } else if (state[idx] === 0) {
                        // Background: e4=50
                        state[idx] = 50;
                    }
                }
            }
        }

        this.device.queue.writeBuffer(this.buffers.state, 0, state);
    }

    async _cpuHippocampusEncode(patternId, tsMinutes, isDeleted, flags) {
        const size = this.size;

        // Read current state
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.buffers.state, 0, this.buffers.readbackState, 0, this.numNeurons * 4);
        this.device.queue.submit([encoder.finish()]);

        await this.buffers.readbackState.mapAsync(GPUMapMode.READ);
        const currentState = new Uint32Array(this.buffers.readbackState.getMappedRange().slice(0));
        this.buffers.readbackState.unmap();

        // Write hippocampus bits into row 63
        for (let region = 0; region < 64; region++) {
            let active = false;

            if (region < 32) {
                active = ((patternId >>> region) & 1) === 1;
            } else if (region < 60) {
                active = ((tsMinutes >>> (region - 32)) & 1) === 1;
            } else if (region === 60) {
                active = isDeleted;
            } else {
                active = ((flags >>> (region - 61)) & 1) === 1;
            }

            // Fill all 4096 pixels of this region in row 63
            const startRow = 63 * 64;
            const startCol = region * 64;
            for (let r = 0; r < 64; r++) {
                for (let c = 0; c < 64; c++) {
                    const idx = (startRow + r) * size + (startCol + c);
                    if (active) {
                        currentState[idx] = 255 | (0 << 8) | (1 << 16) | (1 << 18);
                    } else {
                        currentState[idx] = 0;
                    }
                }
            }
        }

        this.device.queue.writeBuffer(this.buffers.state, 0, currentState);
    }

    // =========================================================================
    // INTERNAL: GPU dispatch stubs
    // =========================================================================

    _workgroups(count) {
        // WebGPU max workgroups per dimension is 65535
        // At 4096x4096 / 256 = 65536, we clamp to 65535.
        // This skips the last 256 neurons (tail of row 63 / h-row)
        // which is protected and doesn't run physics anyway.
        return Math.min(Math.ceil(count / 256), 65535);
    }

    async _dispatchPhysics() {
        if (!this.pipelines.physics) { await this._cpuPhysicsFrame(); return; }
        const p = this.pipelines.physics;
        const bg = this.device.createBindGroup({
            layout: p.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.state } },
                { binding: 1, resource: { buffer: this.buffers.weights } },
                { binding: 2, resource: { buffer: this.buffers.params } },
                { binding: 3, resource: { buffer: this.buffers.frameInfo } },
                { binding: 4, resource: { buffer: this.buffers.sigmoidLUT } },
                { binding: 5, resource: { buffer: this.buffers.rowFlags } },
            ]
        });
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(p.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(this._workgroups(this.numNeurons));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    async _dispatchKWTA() {
        if (!this.pipelines.kwtaClear) { await this._cpuKWTA(); return; }
        // Standalone kWTA dispatch — uses batched _gpuFrame path instead
        // This stub exists for the CPU fallback settle loop
        await this._cpuKWTA();
    }

    async _dispatchSwap() {
        if (!this.pipelines.swap) { await this._cpuSwap(); return; }
        const p = this.pipelines.swap;
        const bg = this.device.createBindGroup({
            layout: p.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.state } },
                { binding: 1, resource: { buffer: this.buffers.params } },
            ]
        });
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(p.pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(this._workgroups(this.numNeurons));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    async _dispatchHebbian() {
        // Hebbian learning now handled inside _gpuFrame (after swap).
        // This standalone dispatch is only used by the CPU physics path.
        await this._cpuHebbian();
    }

    async _dispatchHierarchyBottomUp() {
        // Standalone dispatch — GPU path uses batched _gpuFrame instead
        await this._cpuHierarchyBottomUp();
    }

    async _dispatchHierarchyTopDown() {
        // Standalone dispatch — GPU path uses batched _gpuFrame instead
        await this._cpuHierarchyTopDown();
    }

    async _dispatchSignature() {
        // Signature shader not yet in util.wgsl — CPU fallback
        await this._cpuSignature();
    }

    // =========================================================================
    // INTERNAL: CPU physics fallbacks (temporary until GPU shaders wired)
    // =========================================================================

    async _cpuPhysicsFrame() {
        const N = this.numNeurons;
        const size = this.size;
        const p = this.physics;

        // Read state from GPU
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.buffers.state, 0, this.buffers.readbackState, 0, N * 4);
        this.device.queue.submit([enc.finish()]);
        await this.buffers.readbackState.mapAsync(GPUMapMode.READ);
        const state = new Uint32Array(this.buffers.readbackState.getMappedRange().slice(0));
        this.buffers.readbackState.unmap();

        // Read weights
        const wReadback = this.device.createBuffer({
            size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        const enc2 = this.device.createCommandEncoder();
        enc2.copyBufferToBuffer(this.buffers.weights, 0, wReadback, 0, N * 4);
        this.device.queue.submit([enc2.finish()]);
        await wReadback.mapAsync(GPUMapMode.READ);
        const weights = new Uint32Array(wReadback.getMappedRange().slice(0));
        wReadback.unmap();
        wReadback.destroy();

        const newState = new Uint32Array(state);

        const persist_q8 = Math.round(p.energy_persist * 255);
        const fatigue_q8 = Math.round(p.fatigue_rate * 255);
        const recovery_q8 = Math.round(p.fatigue_recovery * 255);
        const facil_q8 = Math.round(p.facilitation * 255);
        const inhib_q8 = Math.round(p.inhibition * 255);
        const antifacil_q8 = Math.round(p.facilitation * p.anti_facilitation_mult * 255);

        for (let idx = 0; idx < N; idx++) {
            const row = Math.floor(idx / size);
            const col = idx % size;
            const regionRow = Math.floor(row / 64);
            const flags = this.rowPhysics[regionRow];

            if (flags === ROW_FULLY_PROTECTED) continue;

            let packed = state[idx];
            let e4 = packed & 0xFF;
            let f4 = (packed >> 8) & 0xFF;
            const s4 = (packed >> 16) & 0x3;

            // Decay
            if (!(flags & ROW_SKIP_DECAY)) {
                e4 = (e4 * persist_q8) >> 8;
            }

            // Fatigue
            if (!(flags & ROW_SKIP_FATIGUE)) {
                e4 = Math.max(0, e4 - ((f4 * fatigue_q8) >> 8));
            }

            // Mexican hat (simplified: check 4 immediate neighbors)
            if (!(flags & ROW_SKIP_INHIBITION)) {
                let lateral = 0;
                // Check neighbors at Manhattan distances 1-3
                for (let dy = -3; dy <= 3; dy++) {
                    for (let dx = -3; dx <= 3; dx++) {
                        if (dx === 0 && dy === 0) continue;
                        const ny = row + dy;
                        const nx = col + dx;
                        if (ny < 0 || ny >= size || nx < 0 || nx >= size) continue;
                        const nIdx = ny * size + nx;
                        const ns = (state[nIdx] >> 16) & 0x3;
                        const dist = Math.abs(dx) + Math.abs(dy);

                        if (ns > 0) {
                            if (dist === 1) lateral += facil_q8;
                            else if (dist >= 2 && dist <= 3) lateral -= inhib_q8;
                        } else {
                            if (dist === 1) lateral -= antifacil_q8;
                        }
                    }
                }
                e4 = Math.max(0, Math.min(255, e4 + lateral));
            }

            // Weight influence
            if (!(flags & ROW_SKIP_WEIGHTS)) {
                const w = weights[idx];
                const neighbors = [
                    row > 0 ? (row - 1) * size + col : -1,           // N
                    col < size - 1 ? row * size + col + 1 : -1,      // E
                    row < size - 1 ? (row + 1) * size + col : -1,    // S
                    col > 0 ? row * size + col - 1 : -1              // W
                ];
                let influence = 0;
                for (let d = 0; d < 4; d++) {
                    if (neighbors[d] >= 0 && ((state[neighbors[d]] >> 16) & 0x3) > 0) {
                        influence += (w >> (d * 8)) & 0xFF;
                    }
                }
                e4 = Math.min(255, e4 + influence);
            }

            // Boltzmann
            let newS4 = s4;
            if (!(flags & ROW_SKIP_BOLTZMANN)) {
                const effective = e4 - 128;
                const tempScale = 128.0 / p.temperature;
                let lutIdx = Math.round(effective * tempScale) + 128;
                lutIdx = Math.max(0, Math.min(255, lutIdx));
                const prob = this.sigmoidLUT[lutIdx];
                const rand = ((idx * 7 * 1103515245 + 12345) >>> 0) & 0xFF;
                newS4 = rand < prob ? 1 : 0;
            }

            // Fatigue update
            if (!(flags & ROW_SKIP_FATIGUE)) {
                if (newS4 > 0) f4 = Math.min(f4 + 30, 200);
                else f4 = (f4 * recovery_q8) >> 8;
            }

            // Pack result
            newState[idx] = (e4 & 0xFF) | ((f4 & 0xFF) << 8) | ((newS4 & 0x3) << 16) | (((packed >> 18) & 0x3) << 18) | (packed & 0xFFF00000);
        }

        this.device.queue.writeBuffer(this.buffers.state, 0, newState);
    }

    async _cpuSwap() {
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.buffers.state, 0, this.buffers.readbackState, 0, this.numNeurons * 4);
        this.device.queue.submit([enc.finish()]);
        await this.buffers.readbackState.mapAsync(GPUMapMode.READ);
        const state = new Uint32Array(this.buffers.readbackState.getMappedRange().slice(0));
        this.buffers.readbackState.unmap();

        const swapped = new Uint32Array(state.length);
        for (let i = 0; i < state.length; i++) {
            const packed = state[i];
            const cur = (packed >> 16) & 0x3;
            const next = (packed >> 18) & 0x3;
            swapped[i] = (packed & 0xFFFF) | (next << 16) | (cur << 18) | (packed & 0xFFF00000);
        }
        this.device.queue.writeBuffer(this.buffers.state, 0, swapped);
    }

    async _cpuKWTA() {
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.buffers.state, 0, this.buffers.readbackState, 0, this.numNeurons * 4);
        this.device.queue.submit([enc.finish()]);
        await this.buffers.readbackState.mapAsync(GPUMapMode.READ);
        const state = new Uint32Array(this.buffers.readbackState.getMappedRange().slice(0));
        this.buffers.readbackState.unmap();

        const newState = new Uint32Array(state);
        const maxOn = Math.floor(4096 * this.physics.kwta_threshold);

        for (let ry = 0; ry < this.numRegions; ry++) {
            for (let rx = 0; rx < this.numRegions; rx++) {
                if (this.rowPhysics[ry] & ROW_SKIP_KWTA) continue;

                // Collect active neurons and their energies
                const active = [];
                for (let r = 0; r < 64; r++) {
                    for (let c = 0; c < 64; c++) {
                        const idx = (ry * 64 + r) * this.size + (rx * 64 + c);
                        const s4next = (state[idx] >> 18) & 0x3;
                        if (s4next > 0) {
                            active.push({ idx, e4: state[idx] & 0xFF });
                        }
                    }
                }

                if (active.length <= maxOn) continue;

                // Sort by energy descending, keep top maxOn
                active.sort((a, b) => b.e4 - a.e4);
                for (let i = maxOn; i < active.length; i++) {
                    const idx = active[i].idx;
                    newState[idx] = newState[idx] & ~(0x3 << 18); // Clear s4_next
                }
            }
        }

        this.device.queue.writeBuffer(this.buffers.state, 0, newState);
    }

    async _cpuHebbian() {
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.buffers.state, 0, this.buffers.readbackState, 0, this.numNeurons * 4);
        this.device.queue.submit([enc.finish()]);
        await this.buffers.readbackState.mapAsync(GPUMapMode.READ);
        const state = new Uint32Array(this.buffers.readbackState.getMappedRange().slice(0));
        this.buffers.readbackState.unmap();

        const wReadback = this.device.createBuffer({
            size: this.numNeurons * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        const enc2 = this.device.createCommandEncoder();
        enc2.copyBufferToBuffer(this.buffers.weights, 0, wReadback, 0, this.numNeurons * 4);
        this.device.queue.submit([enc2.finish()]);
        await wReadback.mapAsync(GPUMapMode.READ);
        const weights = new Uint32Array(wReadback.getMappedRange().slice(0));
        wReadback.unmap();
        wReadback.destroy();

        const newWeights = new Uint32Array(weights);
        const hebbStep = Math.round(this.physics.hebbian_rate * 16); // Scale to uint8

        for (let idx = 0; idx < this.numNeurons; idx++) {
            const row = Math.floor(idx / this.size);
            const col = idx % this.size;
            const regionRow = Math.floor(row / 64);
            if (this.rowPhysics[regionRow] & ROW_SKIP_LEARNING) continue;

            const s4 = (state[idx] >> 16) & 0x3;
            if (s4 === 0) continue;

            const neighbors = [
                row > 0 ? (row - 1) * this.size + col : -1,
                col < this.size - 1 ? row * this.size + col + 1 : -1,
                row < this.size - 1 ? (row + 1) * this.size + col : -1,
                col > 0 ? row * this.size + col - 1 : -1
            ];

            let w = weights[idx];
            for (let d = 0; d < 4; d++) {
                if (neighbors[d] >= 0 && ((state[neighbors[d]] >> 16) & 0x3) > 0) {
                    let wd = (w >> (d * 8)) & 0xFF;
                    wd = Math.min(127, wd + hebbStep);
                    w = (w & ~(0xFF << (d * 8))) | (wd << (d * 8));
                }
            }
            newWeights[idx] = w;
        }

        this.device.queue.writeBuffer(this.buffers.weights, 0, newWeights);
    }

    async _cpuSignature() {
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.buffers.state, 0, this.buffers.readbackState, 0, this.numNeurons * 4);
        this.device.queue.submit([enc.finish()]);
        await this.buffers.readbackState.mapAsync(GPUMapMode.READ);
        const state = new Uint32Array(this.buffers.readbackState.getMappedRange().slice(0));
        this.buffers.readbackState.unmap();

        const sigSize = this.numRegions * this.numRegions;
        const sig = new Float32Array(sigSize);

        for (let ry = 0; ry < this.numRegions; ry++) {
            for (let rx = 0; rx < this.numRegions; rx++) {
                let sum = 0;
                for (let r = 0; r < 64; r++) {
                    for (let c = 0; c < 64; c++) {
                        const idx = (ry * 64 + r) * this.size + (rx * 64 + c);
                        sum += ((state[idx] >> 16) & 0x3) > 0 ? 1 : 0;
                    }
                }
                sig[ry * this.numRegions + rx] = sum / 4096;
            }
        }

        this.device.queue.writeBuffer(this.buffers.signature, 0, sig);
    }

    // =========================================================================
    // INTERNAL: CPU hierarchy
    // =========================================================================

    async _cpuHierarchyBottomUp() {
        // Read L4 state
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.buffers.state, 0, this.buffers.readbackState, 0, this.numNeurons * 4);
        this.device.queue.submit([enc.finish()]);
        await this.buffers.readbackState.mapAsync(GPUMapMode.READ);
        const state = new Uint32Array(this.buffers.readbackState.getMappedRange().slice(0));
        this.buffers.readbackState.unmap();

        const p = this.physics;
        const size = this.size;
        const nr = this.numRegions; // 64 for 4096

        // L4 → L3 (16x16 pooling, sqrt-density)
        const l3Size = nr * 4; // 256
        const l3 = new Float32Array(l3Size * l3Size);

        // Read existing L3 for persistence
        const encL3 = this.device.createCommandEncoder();
        encL3.copyBufferToBuffer(this.buffers.l3State, 0, this.buffers.readbackL3, 0, l3Size * l3Size * 4);
        this.device.queue.submit([encL3.finish()]);
        await this.buffers.readbackL3.mapAsync(GPUMapMode.READ);
        const oldL3 = new Float32Array(this.buffers.readbackL3.getMappedRange().slice(0));
        this.buffers.readbackL3.unmap();

        for (let ly = 0; ly < l3Size; ly++) {
            for (let lx = 0; lx < l3Size; lx++) {
                // Each L3 cell pools from a 16x16 block of L4
                const startY = ly * 16;
                const startX = lx * 16;
                let countOn = 0;
                for (let dy = 0; dy < 16 && (startY + dy) < size; dy++) {
                    for (let dx = 0; dx < 16 && (startX + dx) < size; dx++) {
                        const idx = (startY + dy) * size + (startX + dx);
                        if (idx < state.length && ((state[idx] >> 16) & 0x3) > 0) countOn++;
                    }
                }
                const density = Math.sqrt(countOn / 256);
                const l3Idx = ly * l3Size + lx;
                l3[l3Idx] = oldL3[l3Idx] * p.l3_persist + density * p.alpha_l3;
            }
        }
        this.device.queue.writeBuffer(this.buffers.l3State, 0, l3);

        // L3 → L2 (4x4 pooling, mean)
        const l2Size = nr; // 64
        const l2 = new Float32Array(l2Size * l2Size);

        const encL2 = this.device.createCommandEncoder();
        encL2.copyBufferToBuffer(this.buffers.l2State, 0, this.buffers.readbackL2, 0, l2Size * l2Size * 4);
        this.device.queue.submit([encL2.finish()]);
        await this.buffers.readbackL2.mapAsync(GPUMapMode.READ);
        const oldL2 = new Float32Array(this.buffers.readbackL2.getMappedRange().slice(0));
        this.buffers.readbackL2.unmap();

        for (let ly = 0; ly < l2Size; ly++) {
            for (let lx = 0; lx < l2Size; lx++) {
                let sum = 0;
                for (let dy = 0; dy < 4; dy++) {
                    for (let dx = 0; dx < 4; dx++) {
                        sum += l3[(ly * 4 + dy) * l3Size + (lx * 4 + dx)];
                    }
                }
                const l2Idx = ly * l2Size + lx;
                l2[l2Idx] = oldL2[l2Idx] * p.l2_persist + (sum / 16) * p.alpha_l2;
            }
        }
        this.device.queue.writeBuffer(this.buffers.l2State, 0, l2);

        // L2 → L1 (4x4 pooling, mean) — only if hierarchy_depth >= 4
        if (p.hierarchy_depth >= 4) {
            const l1Size = 16;
            const l1 = new Float32Array(l1Size * l1Size);

            const encL1 = this.device.createCommandEncoder();
            encL1.copyBufferToBuffer(this.buffers.l1State, 0, this.buffers.readbackL1, 0, l1Size * l1Size * 4);
            this.device.queue.submit([encL1.finish()]);
            await this.buffers.readbackL1.mapAsync(GPUMapMode.READ);
            const oldL1 = new Float32Array(this.buffers.readbackL1.getMappedRange().slice(0));
            this.buffers.readbackL1.unmap();

            for (let ly = 0; ly < l1Size; ly++) {
                for (let lx = 0; lx < l1Size; lx++) {
                    let sum = 0;
                    for (let dy = 0; dy < 4; dy++) {
                        for (let dx = 0; dx < 4; dx++) {
                            sum += l2[(ly * 4 + dy) * l2Size + (lx * 4 + dx)];
                        }
                    }
                    const l1Idx = ly * l1Size + lx;
                    l1[l1Idx] = oldL1[l1Idx] * p.l1_persist + (sum / 16) * p.alpha_l1;
                }
            }
            this.device.queue.writeBuffer(this.buffers.l1State, 0, l1);
        }
    }

    async _cpuHierarchyTopDown() {
        const p = this.physics;
        // Top-down is only needed if hierarchy is active
        // For CPU fallback, skip top-down (it modifies L4 energy which is expensive)
        // The bottom-up pooling provides the hierarchy views
        // Full top-down will come with GPU shaders
    }

    // =========================================================================
    // CLEANUP
    // =========================================================================

    destroy() {
        if (this.device) {
            for (const buf of Object.values(this.buffers)) {
                buf.destroy();
            }
            this.device.destroy();
            this.device = null;
            this.ready = false;
        }
    }
}

// Export constants for external use
export {
    PROFILES, ROW_SKIP_DECAY, ROW_SKIP_FATIGUE, ROW_SKIP_INHIBITION,
    ROW_SKIP_WEIGHTS, ROW_SKIP_BOLTZMANN, ROW_SKIP_KWTA, ROW_SKIP_LEARNING,
    ROW_SKIP_TOPDOWN, ROW_FULLY_PROTECTED, ROW_LEARN_ONLY, HIPPO_EPOCH
};
