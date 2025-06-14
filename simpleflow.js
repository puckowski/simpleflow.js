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

    // Export: save model from IndexedDB to a .bin file for download
    static async exportModelToFile(key) {
        const db = await FlexibleNN._openDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction('models', 'readonly');
            const req = tx.objectStore('models').get(key);
            req.onsuccess = () => {
                db.close();
                if (!req.result) {
                    reject(new Error('Model not found for key: ' + key));
                    return;
                }
                // Serialize model as JSON, then create a Blob for download
                const jsonStr = JSON.stringify(req.result);
                const blob = new Blob([jsonStr], { type: 'application/octet-stream' });
                // Create a temporary <a> element to trigger the download
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${key}.bin`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                resolve(true);
            };
            req.onerror = e => { db.close(); reject(e); };
        });
    }

    static async loadBinModelToIndexedDB(url, key) {
        // 1. Fetch file as text
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`Failed to fetch model: ${resp.status}`);
        const text = await resp.text();

        // 2. Parse JSON
        let modelData;
        try {
            modelData = JSON.parse(text);
        } catch (err) {
            throw new Error("Invalid .bin model file: " + err.message);
        }

        // 3. Store in IndexedDB (using your FlexibleNN code)
        const db = await FlexibleNN._openDB();
        return new Promise((resolve, reject) => {
            const tx = db.transaction('models', 'readwrite');
            tx.objectStore('models').put(modelData, key);
            tx.oncomplete = () => { db.close(); resolve(true); };
            tx.onerror = (e) => { db.close(); reject(e); };
        });
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
