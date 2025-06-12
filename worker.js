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
                    () => Array.from({ length: this.layerSizes[i] }, randn)
                )
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
        let delta = this.as[L].map((a, i) =>
            2 * (a - y[i]) * this.dOutputActivation(this.zs[L - 1][i])
        );
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
    predict(x) { return this.forward(x); }
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

let nn = null;

onmessage = function (e) {
    const { cmd, X, Y, layerSizes, learningRate, epochs, clipValue } = e.data;
    if (cmd === "savemodel") {
        self.postMessage({
            type: "savemodel",
            modelData: {
                layerSizes: nn.layerSizes,
                learningRate: nn.learningRate,
                weights: nn.weights,
                biases: nn.biases,
                activationName: nn.activationName,
                activationCap: nn.activationCap,
                outputActivationName: nn.outputActivationName,
                outputActivationCap: nn.outputActivationCap,
            }
        });
    } else if (cmd === "train") {
        let builder = new FlexibleNNBuilder();

        let nn = builder
            .withLayerSizes(layerSizes)
            .withLearningRate(learningRate)
            .withClipValue(clipValue)
            .withActivation('clippedRelu', 1)
            .withOutputActivation('sigmoid')
            .build();

        for (let epoch = 0; epoch < epochs; ++epoch) {
            let totalLoss = 0;
            for (let i = 0; i < X.length; ++i) {
                const pred = nn.forward(X[i]);
                const loss = pred.reduce((s, v, j) => s + (v - Y[i][j]) ** 2, 0) / pred.length;
                totalLoss += loss;
                nn.backward(Y[i]);
            }
            if ((epoch + 1) % 50 === 0 || epoch === epochs - 1) {
                postMessage({
                    type: "log",
                    epoch: epoch + 1,
                    loss: (totalLoss / X.length)
                });
            }
        }
        postMessage({
            type: "model",
            model: {
                layerSizes: nn.layerSizes,
                learningRate: nn.learningRate,
                weights: nn.weights,
                biases: nn.biases,
                clipValue: nn.clipValue
            }
        });
    } else if (cmd === "predict") {
        if (!nn && e.data.model) {
            const m = e.data.model;
            let builder = new FlexibleNNBuilder();

            nn = builder
                .withLayerSizes(m.layerSizes)
                .withLearningRate(m.learningRate)
                .withClipValue(m.clipValue)
                .withActivation('clippedRelu', 1)
                .withOutputActivation('sigmoid')
                .build();
            nn.weights = m.weights;
            nn.biases = m.biases;
        }
        const result = nn.predict(e.data.input);
        postMessage({ type: "prediction", result });
    }
};
