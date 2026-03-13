/**
 * Integration test — validates the full Sana image generation API flow
 * from a React component through every layer:
 *
 *   React component → useCactusSana hook → CactusSana class
 *     → native Cactus bridge → Nitro HybridObject (mocked)
 *     → response JSON parsing → getLastImagePixelsRgb → writeTempPng → imageUri
 *
 * Exercises both txt2img and img2img with realistic payloads.
 */

import React from 'react';
import { render, act, screen } from '@testing-library/react';
import { useCactusSana } from '../hooks/useCactusSana';
import { CactusFileSystem } from '../native/CactusFileSystem';

// ---------------------------------------------------------------------------
// Mock the Nitro native layer with realistic responses
// ---------------------------------------------------------------------------

const getNative = () => (globalThis as any).__nativeMock;

jest.mock('react-native-nitro-modules', () => {
  const mock = {
    init: jest.fn().mockResolvedValue(undefined),
    destroy: jest.fn().mockResolvedValue(undefined),
    reset: jest.fn().mockResolvedValue(undefined),
    stop: jest.fn().mockResolvedValue(undefined),
    setTelemetryEnvironment: jest.fn().mockResolvedValue(undefined),
    generateImage: jest.fn(),
    generateImageToImage: jest.fn(),
    getLastImagePixelsRgb: jest.fn(),
  };
  (globalThis as any).__nativeMock = mock;
  return {
    NitroModules: { createHybridObject: () => mock },
  };
});

jest.mock('../native/CactusFileSystem', () => ({
  CactusFileSystem: {
    modelExists: jest.fn(),
    getModelPath: jest.fn(),
    getCactusDirectory: jest.fn(),
    downloadModel: jest.fn(),
    writeTempPng: jest.fn(),
    deleteTempFiles: jest.fn(),
  },
}));

jest.mock('../modelRegistry', () => ({
  getRegistry: jest.fn().mockResolvedValue({
    'sana-sprint-0.6b': {
      quantization: {
        int8: { sizeMb: 800, url: 'https://cdn.example.com/sana-sprint-0.6b-int8.bin' },
      },
    },
  }),
}));

const mockedFS = CactusFileSystem as jest.Mocked<typeof CactusFileSystem>;

// ---------------------------------------------------------------------------
// A real React component that exercises the full API
// ---------------------------------------------------------------------------

function SanaTestHarness({ task }: { task: 'txt2img' | 'img2img' }) {
  const sana = useCactusSana({ model: 'sana-sprint-0.6b' });

  React.useEffect(() => {
    (async () => {
      await sana.download();
      await sana.init();

      if (task === 'txt2img') {
        await sana.generateImage({
          prompt: 'a fluffy orange cat sitting on a windowsill, soft afternoon light',
          options: { width: 1024, height: 1024, steps: 4, seed: 42 },
        });
      } else {
        await sana.generateImageToImage({
          prompt: 'a watercolor painting of a mountain landscape',
          initImagePath: 'file:///photos/mountain.png',
          options: { width: 512, height: 512, strength: 0.7, steps: 4, seed: 99 },
        });
      }
    })();
  }, []);

  return (
    <div>
      <span data-testid="downloading">{String(sana.isDownloading)}</span>
      <span data-testid="downloaded">{String(sana.isDownloaded)}</span>
      <span data-testid="initializing">{String(sana.isInitializing)}</span>
      <span data-testid="generating">{String(sana.isGenerating)}</span>
      <span data-testid="imageUri">{sana.imageUri ?? 'null'}</span>
      <span data-testid="error">{sana.error ?? 'null'}</span>
      <span data-testid="step">{sana.generationStep.step}</span>
      <span data-testid="totalSteps">{sana.generationStep.total}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('Sana Integration — full API flow', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // Simulate: model not yet downloaded → download → exists
    mockedFS.modelExists
      .mockResolvedValueOnce(false)   // initial check on mount
      .mockResolvedValueOnce(false)   // check inside CactusSana.download()
      .mockResolvedValue(true);       // subsequent checks (init, etc.)

    mockedFS.downloadModel.mockImplementation(async (_model, _url, onProgress) => {
      onProgress?.(0.0);
      onProgress?.(0.25);
      onProgress?.(0.5);
      onProgress?.(0.75);
      onProgress?.(1.0);
    });

    mockedFS.getModelPath.mockResolvedValue('/data/models/sana-sprint-0.6b-int8');
    mockedFS.getCactusDirectory.mockResolvedValue('/data/cactus');

    // Simulate realistic native image generation response
    getNative().generateImage.mockResolvedValue(
      JSON.stringify({
        success: true,
        output_node: 42,
        width: 1024,
        height: 1024,
        total_time_ms: 3200.5,
      })
    );

    getNative().generateImageToImage.mockResolvedValue(
      JSON.stringify({
        success: true,
        output_node: 43,
        width: 512,
        height: 512,
        total_time_ms: 1800.3,
      })
    );

    // Return an RGB pixel buffer — sized for 1024x1024 by default
    getNative().getLastImagePixelsRgb.mockResolvedValue(
      new Array(3 * 1024 * 1024).fill(128)
    );

    mockedFS.writeTempPng.mockImplementation(async (_pixels, width, height) => {
      return `file:///tmp/cactus_sana_${width}x${height}_${Date.now()}.png`;
    });
  });

  // -----------------------------------------------------------------------
  // txt2img — full lifecycle
  // -----------------------------------------------------------------------

  it('txt2img: download → init → generateImage → imageUri', async () => {
    mockedFS.writeTempPng.mockResolvedValue(
      'file:///tmp/cactus_sana_1024x1024_cat.png'
    );

    await act(async () => {
      render(<SanaTestHarness task="txt2img" />);
    });

    // Let all async effects settle
    await act(async () => {
      await new Promise((r) => setTimeout(r, 50));
    });

    // -- Verify download happened --
    expect(mockedFS.downloadModel).toHaveBeenCalledWith(
      'sana-sprint-0.6b-int8',
      'https://cdn.example.com/sana-sprint-0.6b-int8.bin',
      expect.any(Function)
    );

    // -- Verify native init was called with correct path --
    expect(getNative().setTelemetryEnvironment).toHaveBeenCalledWith('/data/cactus');
    expect(getNative().init).toHaveBeenCalledWith(
      '/data/models/sana-sprint-0.6b-int8',
      undefined,
      false
    );

    // -- Verify generateImage was called with correct args --
    const genCall = getNative().generateImage.mock.calls[0]!;
    expect(genCall[0]).toBe('a fluffy orange cat sitting on a windowsill, soft afternoon light');
    expect(genCall[1]).toBe(1024); // width
    expect(genCall[2]).toBe(1024); // height
    const opts = JSON.parse(genCall[3]);
    expect(opts.steps).toBe(4);
    expect(opts.seed).toBe(42);

    // -- Verify pixel retrieval and PNG write --
    expect(getNative().getLastImagePixelsRgb).toHaveBeenCalled();
    expect(mockedFS.writeTempPng).toHaveBeenCalledWith(
      expect.any(Array),
      1024,
      1024
    );

    // -- Verify final UI state --
    expect(screen.getByTestId('imageUri').textContent).toBe(
      'file:///tmp/cactus_sana_1024x1024_cat.png'
    );
    expect(screen.getByTestId('generating').textContent).toBe('false');
    expect(screen.getByTestId('downloaded').textContent).toBe('true');
    expect(screen.getByTestId('error').textContent).toBe('null');
  });

  // -----------------------------------------------------------------------
  // img2img — full lifecycle
  // -----------------------------------------------------------------------

  it('img2img: download → init → generateImageToImage → imageUri', async () => {
    mockedFS.writeTempPng.mockResolvedValue(
      'file:///tmp/cactus_sana_512x512_mountain.png'
    );

    await act(async () => {
      render(<SanaTestHarness task="img2img" />);
    });

    await act(async () => {
      await new Promise((r) => setTimeout(r, 50));
    });

    // -- Verify generateImageToImage was called correctly --
    const genCall = getNative().generateImageToImage.mock.calls[0]!;
    expect(genCall[0]).toBe('a watercolor painting of a mountain landscape');
    expect(genCall[1]).toBe('/photos/mountain.png'); // file:// stripped
    expect(genCall[2]).toBe(512);  // width
    expect(genCall[3]).toBe(512);  // height
    expect(genCall[4]).toBe(0.7);  // strength
    const opts = JSON.parse(genCall[5]);
    expect(opts.steps).toBe(4);
    expect(opts.seed).toBe(99);

    // -- Verify pixels were written at correct dimensions --
    expect(mockedFS.writeTempPng).toHaveBeenCalledWith(
      expect.any(Array),
      512,
      512
    );

    // -- Verify final UI state --
    expect(screen.getByTestId('imageUri').textContent).toBe(
      'file:///tmp/cactus_sana_512x512_mountain.png'
    );
    expect(screen.getByTestId('generating').textContent).toBe('false');
    expect(screen.getByTestId('error').textContent).toBe('null');
  });

  // -----------------------------------------------------------------------
  // Verify the exact call order through all layers
  // -----------------------------------------------------------------------

  it('txt2img: calls happen in the correct order', async () => {
    const callOrder: string[] = [];

    mockedFS.downloadModel.mockImplementation(async () => { callOrder.push('download'); });
    getNative().setTelemetryEnvironment.mockImplementation(async () => { callOrder.push('telemetry'); });
    getNative().init.mockImplementation(async () => { callOrder.push('init'); });
    getNative().generateImage.mockImplementation(async () => {
      callOrder.push('generateImage');
      return JSON.stringify({ success: true, output_node: 1, width: 1024, height: 1024, total_time_ms: 100 });
    });
    getNative().getLastImagePixelsRgb.mockImplementation(async () => {
      callOrder.push('getPixels');
      return new Array(3 * 1024 * 1024).fill(0);
    });
    mockedFS.writeTempPng.mockImplementation(async () => {
      callOrder.push('writePng');
      return 'file:///tmp/out.png';
    });

    await act(async () => {
      render(<SanaTestHarness task="txt2img" />);
    });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 50));
    });

    expect(callOrder).toEqual([
      'download',
      'telemetry',
      'init',
      'generateImage',
      'getPixels',
      'writePng',
    ]);
  });
});
