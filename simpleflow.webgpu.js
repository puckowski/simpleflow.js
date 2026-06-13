function relu(x) { return Math.max(0, x); }
function drelu(x) { return x > 0 ? 1 : 0; }
function randn() { return Math.random() * 2 - 1; }
function clippedRelu(x, cap = 1.0) {
    return Math.max(0, Math.min(cap, x));
}
function dClippedRelu(x, cap = 1.0) {
    return x > 0 && x < cap ? 1 : 0;
}
function leakyRelu(x) { return x > 0 ? x : 0.01 * x; }
function dLeakyRelu(x) { return x > 0 ? 1 : 0.01; }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function dsigmoid(x) {
    const s = sigmoid(x);
    return s * (1 - s);
}
function tanh(x) { return Math.tanh(x); }
function dtanh(x) { return 1 - Math.pow(Math.tanh(x), 2); }
function clip(x, min, max) { return Math.max(min, Math.min(max, x)); }
function clipArray(arr, min, max) { return arr.map(x => clip(x, min, max)); }

const ActivationLookup = {
    relu: [relu, drelu],
    leakyRelu: [leakyRelu, dLeakyRelu],
    clippedRelu: [
        (x, cap = 1.0) => clippedRelu(x, cap),
        (x, cap = 1.0) => dClippedRelu(x, cap)
    ],
    sigmoid: [sigmoid, dsigmoid],
    tanh: [tanh, dtanh],
};

const ShaderActivationLookup = {
    relu: 'return max(0.0, x);',
    leakyRelu: 'return select(0.01 * x, x, x > 0.0);',
    clippedRelu: 'return max(0.0, min(params.cap, x));',
    sigmoid: 'return 1.0 / (1.0 + exp(-x));',
    tanh: 'return tanh(x);',
    linear: 'return x;'
};

function flattenMatrix(matrix) {
    return Float32Array.from(matrix.flat());
}

function toFloat32(values) {
    return Float32Array.from(values);
}

class FlexibleNN {
    constructor(config) {
        this.layerSizes = config.layerSizes;
        this.learningRate = config.learningRate ?? 0.01;
        this.clipValue = config.clipValue ?? Number.MAX_SAFE_INTEGER / 2;
        this.activationCap = config.activationCap ?? 1.0;
        this.activationName = config.activationName ?? 'relu';
        this.outputActivationName = config.outputActivationName ?? null;
        this.outputActivationCap = config.outputActivationCap ?? 1.0;
        this.preferGPU = config.preferGPU ?? true;
        this.autoInitGPU = config.autoInitGPU ?? false;
        this.trainingExecution = config.trainingExecution ?? 'auto';
        this._gpuContext = null;
        this._gpuUnsupportedReason = null;
        this._gpuInitializationPromise = null;

        this._customHiddenActivation = typeof config.activation === 'function' && typeof config.dActivation === 'function';
        if (this._customHiddenActivation) {
            this.activation = config.activation;
            this.dActivation = config.dActivation;
        } else if (this.activationName === 'clippedRelu') {
            this.activation = (x) => ActivationLookup.clippedRelu[0](x, this.activationCap);
            this.dActivation = (x) => ActivationLookup.clippedRelu[1](x, this.activationCap);
        } else {
            this.activation = ActivationLookup[this.activationName]?.[0] ?? ActivationLookup.relu[0];
            this.dActivation = ActivationLookup[this.activationName]?.[1] ?? ActivationLookup.relu[1];
        }

        this._customOutputActivation = typeof config.outputActivation === 'function';
        if (this._customOutputActivation) {
            this.outputActivation = config.outputActivation;
            this.dOutputActivation = config.dOutputActivation ?? ((x) => 1);
        } else if (this.outputActivationName === 'clippedRelu') {
            this.outputActivation = (x) => ActivationLookup.clippedRelu[0](x, this.outputActivationCap);
            this.dOutputActivation = (x) => ActivationLookup.clippedRelu[1](x, this.outputActivationCap);
        } else if (this.outputActivationName && ActivationLookup[this.outputActivationName]) {
            this.outputActivation = ActivationLookup[this.outputActivationName][0];
            this.dOutputActivation = ActivationLookup[this.outputActivationName][1];
        } else {
            this.outputActivation = x => x;
            this.dOutputActivation = x => 1;
        }

        this.weights = [];
        this.biases = [];
        for (let i = 0; i < this.layerSizes.length - 1; i++) {
            this.weights.push(
                Array.from({ length: this.layerSizes[i + 1] },
                    () => Array.from({ length: this.layerSizes[i] }, randn))
            );
            this.biases.push(Array.from({ length: this.layerSizes[i + 1] }, () => 0));
        }

        if (this.autoInitGPU && this.preferGPU) {
            this._gpuInitializationPromise = this.initializeGPU().catch(() => false);
        }
    }

    static isWebGPUSupported() {
        return typeof navigator !== 'undefined'
            && !!navigator.gpu
            && typeof GPUBufferUsage !== 'undefined'
            && typeof GPUMapMode !== 'undefined';
    }

    static _activationNameForShader(name) {
        if (!name) return 'linear';
        return ShaderActivationLookup[name] ? name : null;
    }

    static _createActivationShader(name) {
        const shaderBody = ShaderActivationLookup[name];
        if (!shaderBody) return null;
        return `
struct Params {
    inSize: u32,
    outSize: u32,
    clipValue: f32,
    cap: f32,
};

@group(0) @binding(0) var<storage, read> inputVec: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> biases: array<f32>;
@group(0) @binding(3) var<storage, read_write> zVec: array<f32>;
@group(0) @binding(4) var<storage, read_write> aVec: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

fn activate(x: f32) -> f32 {
    ${shaderBody}
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let neuron = gid.x;
    if (neuron >= params.outSize) {
        return;
    }

    var sum = biases[neuron];
    let base = neuron * params.inSize;
    for (var feature: u32 = 0u; feature < params.inSize; feature = feature + 1u) {
        sum = sum + weights[base + feature] * inputVec[feature];
    }

    let clipped = clamp(sum, -params.clipValue, params.clipValue);
    zVec[neuron] = clipped;
    aVec[neuron] = activate(clipped);
}
`;
    }

    static _createGPUParamsBuffer(device, inSize, outSize, clipValue, cap) {
        const bytes = new ArrayBuffer(16);
        const view = new DataView(bytes);
        view.setUint32(0, inSize, true);
        view.setUint32(4, outSize, true);
        view.setFloat32(8, clipValue, true);
        view.setFloat32(12, cap, true);
        const buffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buffer, 0, bytes);
        return buffer;
    }

    async initializeGPU() {
        if (!this.preferGPU) return false;
        if (this._gpuContext) return true;
        if (this._gpuInitializationPromise) return this._gpuInitializationPromise;

        this._gpuInitializationPromise = this._initializeGPUInternal();
        try {
            return await this._gpuInitializationPromise;
        } finally {
            this._gpuInitializationPromise = null;
        }
    }

    async _initializeGPUInternal() {
        if (this._customHiddenActivation || this._customOutputActivation) {
            this._gpuUnsupportedReason = 'Custom activation functions are not supported by the WebGPU path.';
            return false;
        }

        if (!FlexibleNN.isWebGPUSupported()) {
            this._gpuUnsupportedReason = 'WebGPU is not available in this environment.';
            return false;
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            this._gpuUnsupportedReason = 'Unable to acquire a WebGPU adapter.';
            return false;
        }

        const device = await adapter.requestDevice();
        const layers = [];

        for (let index = 0; index < this.weights.length; index++) {
            const inSize = this.layerSizes[index];
            const outSize = this.layerSizes[index + 1];
            const isOutputLayer = index === this.weights.length - 1;
            const activationName = FlexibleNN._activationNameForShader(
                isOutputLayer ? this.outputActivationName : this.activationName
            );

            if (!activationName) {
                this._gpuUnsupportedReason = 'At least one activation cannot be compiled to WebGPU.';
                return false;
            }

            const shaderCode = FlexibleNN._createActivationShader(activationName);
            const shaderModule = device.createShaderModule({ code: shaderCode });
            const pipeline = device.createComputePipeline({
                layout: 'auto',
                compute: {
                    module: shaderModule,
                    entryPoint: 'main',
                },
            });

            const inputBuffer = device.createBuffer({
                size: inSize * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            const weightBuffer = device.createBuffer({
                size: inSize * outSize * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            const biasBuffer = device.createBuffer({
                size: outSize * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            const zBuffer = device.createBuffer({
                size: outSize * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            });
            const aBuffer = device.createBuffer({
                size: outSize * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            });
            const zReadBuffer = device.createBuffer({
                size: outSize * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            const aReadBuffer = device.createBuffer({
                size: outSize * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            const paramsBuffer = FlexibleNN._createGPUParamsBuffer(
                device,
                inSize,
                outSize,
                this.clipValue,
                isOutputLayer ? this.outputActivationCap : this.activationCap
            );

            const bindGroup = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: inputBuffer } },
                    { binding: 1, resource: { buffer: weightBuffer } },
                    { binding: 2, resource: { buffer: biasBuffer } },
                    { binding: 3, resource: { buffer: zBuffer } },
                    { binding: 4, resource: { buffer: aBuffer } },
                    { binding: 5, resource: { buffer: paramsBuffer } },
                ],
            });

            layers.push({
                inSize,
                outSize,
                pipeline,
                inputBuffer,
                weightBuffer,
                biasBuffer,
                zBuffer,
                aBuffer,
                zReadBuffer,
                aReadBuffer,
                paramsBuffer,
                bindGroup,
                dirty: true,
            });
        }

        this._gpuContext = { adapter, device, layers };
        this._markGPUDirty();
        await this._syncGPUWeights();
        return true;
    }

    _markGPUDirty() {
        if (!this._gpuContext) return;
        for (const layer of this._gpuContext.layers) {
            layer.dirty = true;
        }
    }

    async _syncGPUWeights() {
        if (!this._gpuContext) return;
        const { device, layers } = this._gpuContext;
        for (let index = 0; index < layers.length; index++) {
            const layer = layers[index];
            if (!layer.dirty) continue;
            device.queue.writeBuffer(layer.weightBuffer, 0, flattenMatrix(this.weights[index]));
            device.queue.writeBuffer(layer.biasBuffer, 0, toFloat32(this.biases[index]));
            layer.dirty = false;
        }
    }

    _resolveTrainingExecutionMode(mode, gpuReady) {
        if (mode === 'cpu') return 'cpu';
        if (mode === 'gpu') return gpuReady ? 'gpu' : 'cpu';

        // Auto mode prefers CPU forward during training because backprop still runs in JS.
        // Pulling every activation back from the GPU each sample is usually slower.
        return 'cpu';
    }

    _shouldLogEpoch(epochIndex, totalEpochs, logEvery) {
        if (!Number.isFinite(logEvery) || logEvery <= 0) return false;
        if (logEvery === 1) return true;
        if ((epochIndex + 1) % logEvery === 0) return true;
        return epochIndex === totalEpochs - 1;
    }

    forward(x) {
        this.zs = [];
        this.as = [x.slice()];
        for (let l = 0; l < this.weights.length; ++l) {
            const prevA = this.as[l];
            let z = this.weights[l].map((wRow, i) =>
                wRow.reduce((sum, w, j) => sum + w * prevA[j], this.biases[l][i])
            );
            z = clipArray(z, -this.clipValue, this.clipValue);
            this.zs.push(z);
            const isOutputLayer = l === this.weights.length - 1;
            const a = isOutputLayer ? z.map(this.outputActivation) : z.map(this.activation);
            this.as.push(a);
        }
        return this.as[this.as.length - 1];
    }

    async forwardGPU(x) {
        const ready = await this.initializeGPU();
        if (!ready) {
            return this.forward(x);
        }

        await this._syncGPUWeights();

        const { device, layers } = this._gpuContext;
        const zs = [];
        const as = [x.slice()];
        let current = Float32Array.from(x);

        for (const layer of layers) {
            device.queue.writeBuffer(layer.inputBuffer, 0, current);

            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(layer.pipeline);
            pass.setBindGroup(0, layer.bindGroup);
            pass.dispatchWorkgroups(Math.ceil(layer.outSize / 64));
            pass.end();

            encoder.copyBufferToBuffer(layer.zBuffer, 0, layer.zReadBuffer, 0, layer.outSize * 4);
            encoder.copyBufferToBuffer(layer.aBuffer, 0, layer.aReadBuffer, 0, layer.outSize * 4);
            device.queue.submit([encoder.finish()]);

            const [zValues, aValues] = await Promise.all([
                this._readGPUBuffer(layer.zReadBuffer, layer.outSize),
                this._readGPUBuffer(layer.aReadBuffer, layer.outSize),
            ]);

            zs.push(zValues);
            as.push(aValues);
            current = Float32Array.from(aValues);
        }

        this.zs = zs;
        this.as = as;
        return as[as.length - 1];
    }

    async _readGPUBuffer(buffer, length) {
        await buffer.mapAsync(GPUMapMode.READ);
        const values = Array.from(new Float32Array(buffer.getMappedRange()).slice(0, length));
        buffer.unmap();
        return values;
    }

    backward(y) {
        const L = this.weights.length;
        const nablaW = this.weights.map(w => w.map(row => row.map(() => 0)));
        const nablaB = this.biases.map(b => b.map(() => 0));
        let delta = this.as[L].map((a, i) => 2 * (a - y[i]) * this.dOutputActivation(this.zs[L - 1][i]));
        for (let l = L - 1; l >= 0; --l) {
            for (let i = 0; i < this.weights[l].length; ++i) {
                for (let j = 0; j < this.weights[l][i].length; ++j) {
                    nablaW[l][i][j] = delta[i] * this.as[l][j];
                }
                nablaB[l][i] = delta[i];
            }
            if (l > 0) {
                const prevDelta = [];
                for (let j = 0; j < this.layerSizes[l]; ++j) {
                    let sum = 0;
                    for (let i = 0; i < this.layerSizes[l + 1]; ++i) {
                        sum += this.weights[l][i][j] * delta[i];
                    }
                    sum *= this.dActivation(this.zs[l - 1][j]);
                    prevDelta.push(sum);
                }
                delta = prevDelta;
            }
        }
        for (let l = 0; l < this.weights.length; ++l) {
            for (let i = 0; i < this.weights[l].length; ++i) {
                for (let j = 0; j < this.weights[l][i].length; ++j) {
                    let update = this.learningRate * nablaW[l][i][j];
                    update = clip(update, -this.clipValue, this.clipValue);
                    this.weights[l][i][j] -= update;
                    this.weights[l][i][j] = clip(this.weights[l][i][j], -this.clipValue, this.clipValue);
                }
                let bUpd = this.learningRate * nablaB[l][i];
                bUpd = clip(bUpd, -this.clipValue, this.clipValue);
                this.biases[l][i] -= bUpd;
                this.biases[l][i] = clip(this.biases[l][i], -this.clipValue, this.clipValue);
            }
        }
        this._markGPUDirty();
    }

    train(X, Y, epochs = 100, options = {}) {
        const logEvery = options.logEvery ?? 1;
        for (let epoch = 0; epoch < epochs; ++epoch) {
            let totalLoss = 0;
            for (let i = 0; i < X.length; ++i) {
                const pred = this.forward(X[i]);
                const loss = pred.reduce((s, v, j) => s + (v - Y[i][j]) ** 2, 0) / pred.length;
                totalLoss += loss;
                this.backward(Y[i]);
            }
            if (this._shouldLogEpoch(epoch, epochs, logEvery)) {
                console.log(`Epoch ${epoch + 1} - Loss: ${(totalLoss / X.length).toFixed(12)}`);
            }
        }
    }

    async trainGPU(X, Y, epochs = 100, options = {}) {
        const requestedMode = options.execution ?? this.trainingExecution;
        const logEvery = options.logEvery ?? 1;
        const gpuReady = await this.initializeGPU();
        const trainingMode = this._resolveTrainingExecutionMode(requestedMode, gpuReady);
        const useGPUForward = trainingMode === 'gpu';

        for (let epoch = 0; epoch < epochs; ++epoch) {
            let totalLoss = 0;
            for (let i = 0; i < X.length; ++i) {
                const pred = useGPUForward ? await this.forwardGPU(X[i]) : this.forward(X[i]);
                const loss = pred.reduce((s, v, j) => s + (v - Y[i][j]) ** 2, 0) / pred.length;
                totalLoss += loss;
                this.backward(Y[i]);
            }
            if (this._shouldLogEpoch(epoch, epochs, logEvery)) {
                console.log(`Epoch ${epoch + 1} - Loss: ${(totalLoss / X.length).toFixed(12)}`);
            }
        }

        if (gpuReady && !useGPUForward) {
            await this._syncGPUWeights();
        }
    }

    predict(x) {
        return this.forward(x);
    }

    async predictGPU(x) {
        return this.forwardGPU(x);
    }

    getGPUStatus() {
        return {
            supported: FlexibleNN.isWebGPUSupported(),
            initialized: !!this._gpuContext,
            reason: this._gpuUnsupportedReason,
            trainingExecution: this.trainingExecution,
        };
    }

    static async loadBinModelToIndexedDB(url, key, { quantized = false, bits = 8 } = {}) {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`Failed to fetch model: ${resp.status}`);
        if (!quantized || bits === 32) {
            const text = await resp.text();
            let modelData;
            try {
                modelData = JSON.parse(text);
            } catch (err) {
                throw new Error('Invalid .bin model file: ' + err.message);
            }
            const db = await FlexibleNN._openDB();
            return new Promise((resolve, reject) => {
                const tx = db.transaction('models', 'readwrite');
                tx.objectStore('models').put(modelData, key);
                tx.oncomplete = () => { db.close(); resolve(true); };
                tx.onerror = (e) => { db.close(); reject(e); };
            });
        }

        if (bits !== 8) throw new Error('Only 8-bit quantization supported in this example');
        const arrBuf = await resp.arrayBuffer();
        const view = new DataView(arrBuf);
        let offset = 0;
        const metaLen = view.getUint32(offset, true); offset += 4;
        const metaStr = new TextDecoder().decode(new Uint8Array(arrBuf, offset, metaLen)); offset += metaLen;
        const meta = JSON.parse(metaStr);

        const weights = [];
        const biases = [];
        for (let l = 0; l < meta.layerSizes.length - 1; ++l) {
            const inSize = meta.layerSizes[l];
            const outSize = meta.layerSizes[l + 1];
            const wLen = inSize * outSize;
            const bLen = outSize;

            const wInt8 = new Int8Array(arrBuf, offset, wLen); offset += wLen;
            const [wMin, wMax] = meta.weightsMinMax[l];
            const wMat = [];
            for (let i = 0; i < outSize; ++i) {
                const row = [];
                for (let j = 0; j < inSize; ++j) {
                    const idx = i * inSize + j;
                    const q = wInt8[idx] + 128;
                    const v = wMin + (wMax - wMin) * (q / 255);
                    row.push(v);
                }
                wMat.push(row);
            }
            weights.push(wMat);

            const bInt8 = new Int8Array(arrBuf, offset, bLen); offset += bLen;
            const [bMin, bMax] = meta.biasesMinMax[l];
            const bVec = [];
            for (let i = 0; i < bLen; ++i) {
                const q = bInt8[i] + 128;
                const v = bMin + (bMax - bMin) * (q / 255);
                bVec.push(v);
            }
            biases.push(bVec);
        }

        const modelData = {
            ...meta,
            weights,
            biases,
        };
        const db = await FlexibleNN._openDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction('models', 'readwrite');
            tx.objectStore('models').put(modelData, key);
            tx.oncomplete = () => { db.close(); resolve(true); };
            tx.onerror = (e) => { db.close(); reject(e); };
        });
    }

    static async importModelFromFile(file, key) {
        const db = await FlexibleNN._openDB();
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = function (event) {
                try {
                    const modelData = JSON.parse(event.target.result);
                    const tx = db.transaction('models', 'readwrite');
                    tx.objectStore('models').put(modelData, key);
                    tx.oncomplete = () => { db.close(); resolve(true); };
                    tx.onerror = e => { db.close(); reject(e); };
                } catch (err) {
                    db.close();
                    reject(err);
                }
            };
            reader.onerror = function (e) {
                db.close();
                reject(e);
            };
            reader.readAsText(file);
        });
    }

    async saveModel(key) {
        const db = await FlexibleNN._openDB();
        const modelData = {
            layerSizes: this.layerSizes,
            learningRate: this.learningRate,
            weights: this.weights,
            biases: this.biases,
            activationName: this.activationName,
            activationCap: this.activationCap,
            outputActivationName: this.outputActivationName,
            outputActivationCap: this.outputActivationCap,
        };
        return new Promise((resolve, reject) => {
            const tx = db.transaction('models', 'readwrite');
            tx.objectStore('models').put(modelData, key);
            tx.oncomplete = () => { db.close(); resolve(true); };
            tx.onerror = (e) => { db.close(); reject(e); };
        });
    }

    static async exportModelToBinFile({ key, quantized = false, bits = 8 }) {
        const db = await FlexibleNN._openDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction('models', 'readonly');
            const req = tx.objectStore('models').get(key);
            req.onsuccess = () => {
                db.close();
                if (!req.result) return reject(new Error('Model not found for key: ' + key));
                const model = req.result;

                if (!quantized || bits === 32) {
                    const jsonStr = JSON.stringify(model);
                    const blob = new Blob([jsonStr], { type: 'application/octet-stream' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${key}.bin`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    resolve(true);
                    return;
                }

                if (bits !== 8) return reject(new Error('Only 8-bit quantization is supported'));

                const quant = { weights: [], weightsMinMax: [], biases: [], biasesMinMax: [] };
                for (let l = 0; l < model.weights.length; ++l) {
                    const wArr = model.weights[l].flat();
                    const wMin = Math.min(...wArr);
                    const wMax = Math.max(...wArr);
                    quant.weightsMinMax.push([wMin, wMax]);
                    const wQuant = wArr.map(v => Math.round((v - wMin) / (wMax - wMin || 1) * 255) - 128);
                    quant.weights.push(Int8Array.from(wQuant));

                    const bArr = model.biases[l];
                    const bMin = Math.min(...bArr);
                    const bMax = Math.max(...bArr);
                    quant.biasesMinMax.push([bMin, bMax]);
                    const bQuant = bArr.map(v => Math.round((v - bMin) / (bMax - bMin || 1) * 255) - 128);
                    quant.biases.push(Int8Array.from(bQuant));
                }

                const meta = {
                    layerSizes: model.layerSizes,
                    activationName: model.activationName,
                    activationCap: model.activationCap,
                    outputActivationName: model.outputActivationName,
                    outputActivationCap: model.outputActivationCap,
                    weightsMinMax: quant.weightsMinMax,
                    biasesMinMax: quant.biasesMinMax,
                };

                const metaStr = JSON.stringify(meta);
                const metaLen = new Uint32Array([metaStr.length]);
                const binParts = [metaLen.buffer, new TextEncoder().encode(metaStr)];
                for (let l = 0; l < quant.weights.length; ++l) {
                    binParts.push(quant.weights[l].buffer);
                    binParts.push(quant.biases[l].buffer);
                }

                const blob = new Blob(binParts, { type: 'application/octet-stream' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${key}_8bit.bin`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                resolve(true);
            };
            req.onerror = e => { db.close(); reject(e); };
        });
    }

    static async import8bitBinModelToIndexedDB(file, key) {
        const arrBuf = await file.arrayBuffer();
        const view = new DataView(arrBuf);
        let offset = 0;
        const metaLen = view.getUint32(offset, true); offset += 4;
        const metaStr = new TextDecoder().decode(new Uint8Array(arrBuf, offset, metaLen)); offset += metaLen;
        const meta = JSON.parse(metaStr);

        const weights = [];
        const biases = [];
        for (let l = 0; l < meta.layerSizes.length - 1; ++l) {
            const inSize = meta.layerSizes[l];
            const outSize = meta.layerSizes[l + 1];
            const wLen = inSize * outSize;
            const bLen = outSize;

            const wInt8 = new Int8Array(arrBuf, offset, wLen); offset += wLen;
            const [wMin, wMax] = meta.weightsMinMax[l];
            const wMat = [];
            for (let i = 0; i < outSize; ++i) {
                const row = [];
                for (let j = 0; j < inSize; ++j) {
                    const idx = i * inSize + j;
                    const q = wInt8[idx] + 128;
                    const v = wMin + (wMax - wMin) * (q / 255);
                    row.push(v);
                }
                wMat.push(row);
            }
            weights.push(wMat);

            const bInt8 = new Int8Array(arrBuf, offset, bLen); offset += bLen;
            const [bMin, bMax] = meta.biasesMinMax[l];
            const bVec = [];
            for (let i = 0; i < bLen; ++i) {
                const q = bInt8[i] + 128;
                const v = bMin + (bMax - bMin) * (q / 255);
                bVec.push(v);
            }
            biases.push(bVec);
        }

        const modelData = {
            ...meta,
            weights,
            biases,
        };
        const db = await FlexibleNN._openDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction('models', 'readwrite');
            tx.objectStore('models').put(modelData, key);
            tx.oncomplete = () => { db.close(); resolve(true); };
            tx.onerror = (e) => { db.close(); reject(e); };
        });
    }

    static async loadModel(key) {
        const db = await FlexibleNN._openDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction('models', 'readonly');
            const req = tx.objectStore('models').get(key);
            req.onsuccess = () => {
                db.close();
                if (!req.result) return reject(new Error('Model not found for key ' + key));

                const builder = new FlexibleNNBuilder();
                const model = builder
                    .withLayerSizes(req.result.layerSizes)
                    .withLearningRate(req.result.learningRate)
                    .withClipValue(req.result.clipValue)
                    .withActivation(req.result.activationName, req.result.activationCap)
                    .withOutputActivation(req.result.outputActivationName, req.result.outputActivationCap)
                    .build();

                model.weights = req.result.weights;
                model.biases = req.result.biases;
                model._markGPUDirty();
                resolve(model);
            };
            req.onerror = (e) => { db.close(); reject(e); };
        });
    }

    static _openDB() {
        return new Promise((resolve, reject) => {
            const req = indexedDB.open('SimpleNN_DB', 1);
            req.onupgradeneeded = function () {
                if (!req.result.objectStoreNames.contains('models')) {
                    req.result.createObjectStore('models');
                }
            };
            req.onsuccess = () => resolve(req.result);
            req.onerror = (e) => reject(e);
        });
    }
}

class FlexibleNNBuilder {
    constructor() {
        this.config = {};
    }

    withLayerSizes(sizes) {
        this.config.layerSizes = sizes;
        return this;
    }

    withLearningRate(lr) {
        this.config.learningRate = lr;
        return this;
    }

    withClipValue(clipVal) {
        this.config.clipValue = clipVal ?? Number.MAX_SAFE_INTEGER / 2;
        return this;
    }

    withActivation(name, cap = 1.0) {
        this.config.activationName = name;
        this.config.activationCap = cap;
        return this;
    }

    withCustomActivation(fn, dFn) {
        this.config.activation = fn;
        this.config.dActivation = dFn;
        return this;
    }

    withOutputActivation(name, cap = 1.0) {
        this.config.outputActivationName = name;
        this.config.outputActivationCap = cap;
        return this;
    }

    withCustomOutputActivation(fn, dFn) {
        this.config.outputActivation = fn;
        this.config.dOutputActivation = dFn;
        return this;
    }

    withGPUPreference(preferGPU = true) {
        this.config.preferGPU = preferGPU;
        return this;
    }

    withAutoInitGPU(autoInitGPU = true) {
        this.config.autoInitGPU = autoInitGPU;
        return this;
    }

    withTrainingExecution(mode = 'auto') {
        this.config.trainingExecution = mode;
        return this;
    }

    build() {
        if (!this.config.layerSizes) throw new Error('Must set layerSizes');
        return new FlexibleNN(this.config);
    }

    async buildGPU() {
        const model = this.build();
        await model.initializeGPU();
        return model;
    }
}

window.FlexibleNN = FlexibleNN;
window.FlexibleNNBuilder = FlexibleNNBuilder;
window.FlexibleNNWebGPU = FlexibleNN;
window.FlexibleNNWebGPUBuilder = FlexibleNNBuilder;