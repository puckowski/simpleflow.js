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
function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
}
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

class FlexibleNN {
    constructor(config) {
        this.layerSizes = config.layerSizes;
        this.learningRate = config.learningRate ?? 0.01;
        this.clipValue = config.clipValue ?? Number.MAX_SAFE_INTEGER / 2;
        this.activationCap = config.activationCap ?? 1.0;
        this.activationName = config.activationName ?? "relu";
        this.outputActivationName = config.outputActivationName ?? null;
        this.outputActivationCap = config.outputActivationCap ?? 1.0;

        if (config.activationName === "clippedRelu") {
            this.activation = (x) => ActivationLookup["clippedRelu"][0](x, this.activationCap);
            this.dActivation = (x) => ActivationLookup["clippedRelu"][1](x, this.activationCap);
        } else {
            this.activation = ActivationLookup[config.activationName][0] ?? ActivationLookup['relu'][0];
            this.dActivation = ActivationLookup[config.activationName][1] ?? ActivationLookup['relu'][1];
        }

        if (config.outputActivation) {
            this.outputActivation = config.outputActivation;
            this.dOutputActivation = config.dOutputActivation ?? ((x) => 1);
        } else if (config.outputActivationName === "clippedRelu") {
            this.outputActivation = (x) => ActivationLookup["clippedRelu"][0](x, config.outputActivationCap ?? 1.0);
            this.dOutputActivation = (x) => ActivationLookup["clippedRelu"][1](x, config.outputActivationCap ?? 1.0);
        } else if (config.outputActivationName && ActivationLookup[config.outputActivationName]) {
            this.outputActivation = ActivationLookup[config.outputActivationName][0];
            this.dOutputActivation = ActivationLookup[config.outputActivationName][1];
        } else {
            this.outputActivation = x => x; // Linear by default
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
            const isOutputLayer = (l === this.weights.length - 1);
            const a = isOutputLayer ? z.map(this.outputActivation) : z.map(this.activation);
            this.as.push(a);
        }
        return this.as[this.as.length - 1];
    }
    backward(y) {
        const L = this.weights.length;
        let nablaW = this.weights.map(w => w.map(row => row.map(_ => 0)));
        let nablaB = this.biases.map(b => b.map(_ => 0));
        let delta = this.as[L].map((a, i) => 2 * (a - y[i]));
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
    }

    train(X, Y, epochs = 100) {
        for (let epoch = 0; epoch < epochs; ++epoch) {
            let totalLoss = 0;
            for (let i = 0; i < X.length; ++i) {
                const pred = this.forward(X[i]);
                const loss = pred.reduce((s, v, j) => s + (v - Y[i][j]) ** 2, 0) / pred.length;
                totalLoss += loss;
                this.backward(Y[i]);
            }
            console.log(
                `Epoch ${epoch + 1} – Loss: ${(totalLoss / X.length).toFixed(12)}`
            );
        }
    }

    predict(x) {
        return this.forward(x);
    }

    static async loadBinModelToIndexedDB(url, key, { quantized = false, bits = 8 } = {}) {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`Failed to fetch model: ${resp.status}`);
        if (!quantized || bits === 32) {
            // JSON-based float model
            const text = await resp.text();
            let modelData;
            try {
                modelData = JSON.parse(text);
            } catch (err) {
                throw new Error("Invalid .bin model file: " + err.message);
            }
            const db = await FlexibleNN._openDB();
            return new Promise((resolve, reject) => {
                const tx = db.transaction('models', 'readwrite');
                tx.objectStore('models').put(modelData, key);
                tx.oncomplete = () => { db.close(); resolve(true); };
                tx.onerror = (e) => { db.close(); reject(e); };
            });
        } else {
            // 8-bit quantized
            if (bits !== 8) throw new Error("Only 8-bit quantization supported in this example");
            const arrBuf = await resp.arrayBuffer();
            let view = new DataView(arrBuf);
            let offset = 0;
            let metaLen = view.getUint32(offset, true); offset += 4;
            let metaStr = new TextDecoder().decode(
                new Uint8Array(arrBuf, offset, metaLen)
            ); offset += metaLen;
            const meta = JSON.parse(metaStr);

            let weights = [], biases = [];
            for (let l = 0; l < meta.layerSizes.length - 1; ++l) {
                let inSize = meta.layerSizes[l], outSize = meta.layerSizes[l + 1];
                let wLen = inSize * outSize, bLen = outSize;

                // Weights
                let wInt8 = new Int8Array(arrBuf, offset, wLen); offset += wLen;
                let [wMin, wMax] = meta.weightsMinMax[l];
                let wMat = [];
                for (let i = 0; i < outSize; ++i) {
                    let row = [];
                    for (let j = 0; j < inSize; ++j) {
                        let idx = i * inSize + j;
                        let q = wInt8[idx] + 128; // map back to 0-255
                        let v = wMin + (wMax - wMin) * (q / 255);
                        row.push(v);
                    }
                    wMat.push(row);
                }
                weights.push(wMat);

                // Biases
                let bInt8 = new Int8Array(arrBuf, offset, bLen); offset += bLen;
                let [bMin, bMax] = meta.biasesMinMax[l];
                let bVec = [];
                for (let i = 0; i < bLen; ++i) {
                    let q = bInt8[i] + 128;
                    let v = bMin + (bMax - bMin) * (q / 255);
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
    }

    // Import: load model from a .bin file into IndexedDB under a given key
    static async importModelFromFile(file, key) {
        const db = await FlexibleNN._openDB();
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = function (event) {
                try {
                    const modelData = JSON.parse(event.target.result);
                    // Optionally validate modelData structure here
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

    // --- IndexedDB Save/Load ---
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

    // Exports model as .bin, quantized if specified, otherwise as float JSON
    static async exportModelToBinFile({ key, quantized = false, bits = 8 }) {
        const db = await FlexibleNN._openDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction('models', 'readonly');
            const req = tx.objectStore('models').get(key);
            req.onsuccess = () => {
                db.close();
                if (!req.result) return reject(new Error('Model not found for key: ' + key));
                const model = req.result;

                // If not quantized or bits === 32, just export JSON blob
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

                // Only 8-bit quantization supported for now
                if (bits !== 8) return reject(new Error("Only 8-bit quantization is supported"));

                // Quantize weights and biases
                let quant = { weights: [], weightsMinMax: [], biases: [], biasesMinMax: [] };
                for (let l = 0; l < model.weights.length; ++l) {
                    // Weights
                    let wArr = model.weights[l].flat();
                    let wMin = Math.min(...wArr);
                    let wMax = Math.max(...wArr);
                    quant.weightsMinMax.push([wMin, wMax]);
                    let wQuant = wArr.map(v =>
                        Math.round((v - wMin) / (wMax - wMin || 1) * 255) - 128 // Int8
                    );
                    quant.weights.push(Int8Array.from(wQuant));
                    // Biases
                    let bArr = model.biases[l];
                    let bMin = Math.min(...bArr);
                    let bMax = Math.max(...bArr);
                    quant.biasesMinMax.push([bMin, bMax]);
                    let bQuant = bArr.map(v =>
                        Math.round((v - bMin) / (bMax - bMin || 1) * 255) - 128
                    );
                    quant.biases.push(Int8Array.from(bQuant));
                }
                // Save other config
                const meta = {
                    layerSizes: model.layerSizes,
                    activationName: model.activationName,
                    activationCap: model.activationCap,
                    outputActivationName: model.outputActivationName,
                    outputActivationCap: model.outputActivationCap,
                    weightsMinMax: quant.weightsMinMax,
                    biasesMinMax: quant.biasesMinMax,
                };
                // Serialize (meta JSON + all arrays in order)
                let metaStr = JSON.stringify(meta);
                let metaLen = new Uint32Array([metaStr.length]);
                let binParts = [metaLen.buffer, new TextEncoder().encode(metaStr)];
                for (let l = 0; l < quant.weights.length; ++l) {
                    binParts.push(quant.weights[l].buffer);
                    binParts.push(quant.biases[l].buffer);
                }
                let blob = new Blob(binParts, { type: 'application/octet-stream' });
                // Trigger download
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

    // Usage: FlexibleNN.import8bitBinModelToIndexedDB(file, key)
    static async import8bitBinModelToIndexedDB(file, key) {
        // 1. Read file as ArrayBuffer
        const arrBuf = await file.arrayBuffer();
        let view = new DataView(arrBuf);
        let offset = 0;
        let metaLen = view.getUint32(offset, /*littleEndian=*/true); offset += 4;
        let metaStr = new TextDecoder().decode(
            new Uint8Array(arrBuf, offset, metaLen)
        ); offset += metaLen;
        const meta = JSON.parse(metaStr);

        // 2. Load weights/biases, dequantize
        let weights = [], biases = [];
        for (let l = 0; l < meta.layerSizes.length - 1; ++l) {
            let inSize = meta.layerSizes[l], outSize = meta.layerSizes[l + 1];
            let wLen = inSize * outSize, bLen = outSize;
            // Weights
            let wInt8 = new Int8Array(arrBuf, offset, wLen); offset += wLen;
            let [wMin, wMax] = meta.weightsMinMax[l];
            let wMat = [];
            for (let i = 0; i < outSize; ++i) {
                let row = [];
                for (let j = 0; j < inSize; ++j) {
                    let idx = i * inSize + j;
                    let q = wInt8[idx] + 128; // map back to 0-255
                    let v = wMin + (wMax - wMin) * (q / 255);
                    row.push(v);
                }
                wMat.push(row);
            }
            weights.push(wMat);
            // Biases
            let bInt8 = new Int8Array(arrBuf, offset, bLen); offset += bLen;
            let [bMin, bMax] = meta.biasesMinMax[l];
            let bVec = [];
            for (let i = 0; i < bLen; ++i) {
                let q = bInt8[i] + 128;
                let v = bMin + (bMax - bMin) * (q / 255);
                bVec.push(v);
            }
            biases.push(bVec);
        }
        // 3. Rebuild modelData
        const modelData = {
            ...meta,
            weights,
            biases
        };
        // 4. Save to IndexedDB (identical to other methods)
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
                if (!req.result) return reject(new Error("Model not found for key " + key));
                // Recreate model and assign weights/biases
                let builder = new FlexibleNNBuilder();

                const m = builder
                    .withLayerSizes(req.result.layerSizes)
                    .withLearningRate(req.result.learningRate)
                    .withActivation(req.result.activationName, req.result.activationCap)
                    .withOutputActivation(req.result.outputActivationName, req.result.outputActivationCap)
                    .build();

                m.weights = req.result.weights;
                m.biases = req.result.biases;
                resolve(m);
            };
            req.onerror = (e) => { db.close(); reject(e); };
        });
    }

    // Helper to open or create the IndexedDB store
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
        this.config.activation = name;
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
    build() {
        if (!this.config.layerSizes) throw new Error("Must set layerSizes");
        return new FlexibleNN(this.config);
    }
}
