// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { indexedDB } from 'fake-indexeddb';
import '../simpleflow.webgpu.js';

const { FlexibleNN, FlexibleNNBuilder } = window;

function deleteDb(name) {
  return new Promise((resolve) => {
    const req = indexedDB.deleteDatabase(name);
    req.onsuccess = () => resolve();
    req.onerror = () => resolve();
    req.onblocked = () => resolve();
  });
}

describe('simpleflow.webgpu.js', () => {
  beforeEach(async () => {
    window.indexedDB = indexedDB;
    globalThis.indexedDB = indexedDB;
    await deleteDb('SimpleNN_DB');
  });

  it('throws when layerSizes is not provided', () => {
    expect(() => new FlexibleNNBuilder().build()).toThrow('Must set layerSizes');
  });

  it('produces output with expected dimension on the CPU path', () => {
    const nn = new FlexibleNNBuilder()
      .withLayerSizes([3, 4, 2])
      .withActivation('relu')
      .build();

    const out = nn.predict([0.1, -0.2, 0.3]);
    expect(out).toHaveLength(2);
  });

  it('reports WebGPU support and falls back cleanly when unavailable', async () => {
    const nn = new FlexibleNNBuilder()
      .withLayerSizes([2, 3, 1])
      .withActivation('tanh')
      .withOutputActivation('sigmoid')
      .build();

    nn.weights = [
      [
        [10, -10],
        [-10, 10],
        [10, 10],
      ],
      [[5, -5, 5]],
    ];
    nn.biases = [[0, 0, 0], [0]];

    expect(FlexibleNN.isWebGPUSupported()).toBe(false);
    expect(await nn.initializeGPU()).toBe(false);

    const status = nn.getGPUStatus();
    expect(status.supported).toBe(false);
    expect(status.initialized).toBe(false);
    expect(status.reason).toContain('WebGPU');

    const gpuOut = await nn.predictGPU([1, 0]);
    const cpuOut = nn.predict([1, 0]);
    expect(gpuOut[0]).toBeCloseTo(cpuOut[0], 10);
  });

  it('updates weights and biases during trainGPU()', async () => {
    const nn = new FlexibleNNBuilder()
      .withLayerSizes([1, 1])
      .withLearningRate(0.1)
      .withActivation('relu')
      .build();

    nn.weights = [[[0]]];
    nn.biases = [[0]];

    await nn.trainGPU([[1]], [[1]], 1);

    expect(nn.weights[0][0][0]).toBeCloseTo(0.2, 8);
    expect(nn.biases[0][0]).toBeCloseTo(0.2, 8);
  });

  it('saves and reloads a model from IndexedDB', async () => {
    const nn = new FlexibleNNBuilder()
      .withLayerSizes([2, 2, 1])
      .withLearningRate(0.05)
      .withActivation('leakyRelu')
      .withOutputActivation('sigmoid')
      .build();

    nn.weights = [
      [
        [0.1, -0.2],
        [0.3, 0.4],
      ],
      [[0.5, -0.6]],
    ];
    nn.biases = [[0.01, -0.02], [0.03]];

    await nn.saveModel('roundtrip-model');
    const loaded = await FlexibleNN.loadModel('roundtrip-model');

    expect(loaded.layerSizes).toEqual([2, 2, 1]);
    expect(loaded.learningRate).toBeCloseTo(0.05, 10);
    expect(loaded.weights).toEqual(nn.weights);
    expect(loaded.biases).toEqual(nn.biases);

    const input = [0.25, -0.5];
    expect(loaded.predict(input)[0]).toBeCloseTo(nn.predict(input)[0], 10);
  });
});