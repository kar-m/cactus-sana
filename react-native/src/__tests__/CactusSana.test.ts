import { CactusSana } from '../classes/CactusSana';
import { CactusFileSystem } from '../native/CactusFileSystem';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

// Must define the mock object inside the factory to avoid TDZ issues
// (jest.mock is hoisted above const declarations)
const getMock = () => (globalThis as any).__mockHybridCactus;

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
  (globalThis as any).__mockHybridCactus = mock;
  return { NitroModules: { createHybridObject: () => mock } };
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
        int8: { sizeMb: 800, url: 'https://example.com/sana-int8.bin' },
        int4: { sizeMb: 400, url: 'https://example.com/sana-int4.bin' },
      },
    },
  }),
}));

const mockedFS = CactusFileSystem as jest.Mocked<typeof CactusFileSystem>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function mockGeneration(width = 1024, height = 1024) {
  getMock().generateImage.mockResolvedValue(
    JSON.stringify({ success: true, output_node: 7, width, height, total_time_ms: 1500 })
  );
  getMock().getLastImagePixelsRgb.mockResolvedValue(
    new Array(3 * width * height).fill(128)
  );
  mockedFS.writeTempPng.mockResolvedValue('file:///tmp/sana_out.png');
}

function mockImg2Img(width = 512, height = 512) {
  getMock().generateImageToImage.mockResolvedValue(
    JSON.stringify({ success: true, output_node: 9, width, height, total_time_ms: 800 })
  );
  getMock().getLastImagePixelsRgb.mockResolvedValue(
    new Array(3 * width * height).fill(128)
  );
  mockedFS.writeTempPng.mockResolvedValue('file:///tmp/sana_img2img.png');
}

async function initSana(): Promise<CactusSana> {
  mockedFS.modelExists.mockResolvedValue(true);
  mockedFS.getModelPath.mockResolvedValue('/models/sana-sprint-0.6b-int8');
  mockedFS.getCactusDirectory.mockResolvedValue('/cache/cactus');
  const sana = new CactusSana();
  await sana.init();
  return sana;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('CactusSana', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockedFS.modelExists.mockResolvedValue(true);
    mockedFS.getModelPath.mockResolvedValue('/models/sana-sprint-0.6b-int8');
    mockedFS.getCactusDirectory.mockResolvedValue('/cache/cactus');
  });

  // --- Constructor ---

  describe('constructor', () => {
    it('defaults to sana-sprint-0.6b-int8', () => {
      expect(new CactusSana().getModelName()).toBe('sana-sprint-0.6b-int8');
    });

    it('accepts custom model/quantization', () => {
      const sana = new CactusSana({ model: 'sana-1.0', options: { quantization: 'int4' } });
      expect(sana.getModelName()).toBe('sana-1.0-int4');
    });

    it('appends -pro', () => {
      expect(new CactusSana({ options: { pro: true } }).getModelName()).toBe('sana-sprint-0.6b-int8-pro');
    });
  });

  // --- download() ---

  describe('download()', () => {
    it('skips when model already exists', async () => {
      const onProgress = jest.fn();
      await new CactusSana().download({ onProgress });
      expect(mockedFS.downloadModel).not.toHaveBeenCalled();
      expect(onProgress).toHaveBeenCalledWith(1.0);
    });

    it('downloads from registry', async () => {
      mockedFS.modelExists.mockResolvedValue(false);
      mockedFS.downloadModel.mockResolvedValue(undefined);
      await new CactusSana().download();
      expect(mockedFS.downloadModel).toHaveBeenCalledWith(
        'sana-sprint-0.6b-int8', 'https://example.com/sana-int8.bin', undefined
      );
    });

    it('reports progress', async () => {
      mockedFS.modelExists.mockResolvedValue(false);
      mockedFS.downloadModel.mockImplementation(async (_m, _u, cb) => {
        cb?.(0); cb?.(0.5); cb?.(1.0);
      });
      const onProgress = jest.fn();
      await new CactusSana().download({ onProgress });
      expect(onProgress).toHaveBeenCalledWith(0.5);
    });

    it('throws when already downloading', async () => {
      mockedFS.modelExists.mockResolvedValue(false);
      mockedFS.downloadModel.mockImplementation(() => new Promise(() => {}));
      const sana = new CactusSana();
      sana.download();
      await new Promise((r) => setTimeout(r, 0));
      await expect(sana.download()).rejects.toThrow('already downloading');
    });

    it('skips for file:// paths', async () => {
      const onProgress = jest.fn();
      await new CactusSana({ model: 'file:///custom' }).download({ onProgress });
      expect(mockedFS.downloadModel).not.toHaveBeenCalled();
      expect(onProgress).toHaveBeenCalledWith(1.0);
    });
  });

  // --- init() ---

  describe('init()', () => {
    it('calls native init with model path', async () => {
      await new CactusSana().init();
      expect(getMock().init).toHaveBeenCalledWith(
        '/models/sana-sprint-0.6b-int8', undefined, false
      );
    });

    it('is idempotent', async () => {
      const sana = new CactusSana();
      await sana.init();
      await sana.init();
      expect(getMock().init).toHaveBeenCalledTimes(1);
    });

    it('throws when model not downloaded', async () => {
      mockedFS.modelExists.mockResolvedValue(false);
      await expect(new CactusSana().init()).rejects.toThrow('is not downloaded');
    });

    it('strips file:// from paths', async () => {
      await new CactusSana({ model: 'file:///custom/sana' }).init();
      expect(getMock().init).toHaveBeenCalledWith('/custom/sana', undefined, false);
    });
  });

  // --- generateImage() ---

  describe('generateImage()', () => {
    beforeEach(() => mockGeneration());

    it('generates and returns imageUri', async () => {
      const sana = await initSana();
      const result = await sana.generateImage({ prompt: 'a cat' });
      expect(result.success).toBe(true);
      expect(result.imageUri).toBe('file:///tmp/sana_out.png');
      expect(result.width).toBe(1024);
    });

    it('passes custom options', async () => {
      const sana = await initSana();
      await sana.generateImage({ prompt: 'a dog', options: { width: 512, height: 512, steps: 4, seed: 42 } });
      const call = getMock().generateImage.mock.calls[0];
      expect(call[1]).toBe(512);
      expect(call[2]).toBe(512);
      const opts = JSON.parse(call[3]);
      expect(opts.steps).toBe(4);
      expect(opts.seed).toBe(42);
    });

    it('writes pixels via writeTempPng', async () => {
      const sana = await initSana();
      await sana.generateImage({ prompt: 'x' });
      expect(mockedFS.writeTempPng).toHaveBeenCalledWith(expect.any(Array), 1024, 1024);
    });

    it('auto-inits if needed', async () => {
      mockGeneration();
      await new CactusSana().generateImage({ prompt: 'x' });
      expect(getMock().init).toHaveBeenCalled();
    });

    it('throws when already generating', async () => {
      getMock().generateImage.mockImplementation(() => new Promise(() => {}));
      const sana = await initSana();
      sana.generateImage({ prompt: 'a' });
      await new Promise((r) => setTimeout(r, 0));
      await expect(sana.generateImage({ prompt: 'b' })).rejects.toThrow('already generating');
    });

    it('throws on native error response', async () => {
      getMock().generateImage.mockResolvedValue(JSON.stringify({ success: false, error: 'OOM' }));
      const sana = await initSana();
      await expect(sana.generateImage({ prompt: 'x' })).rejects.toThrow('OOM');
    });

    it('handles native crash', async () => {
      getMock().generateImage.mockRejectedValue(new Error('SIGSEGV'));
      const sana = await initSana();
      await expect(sana.generateImage({ prompt: 'x' })).rejects.toThrow('SIGSEGV');
    });

    it('resets isGenerating after failure', async () => {
      getMock().generateImage.mockRejectedValueOnce(new Error('fail'));
      const sana = await initSana();
      await sana.generateImage({ prompt: 'a' }).catch(() => {});
      mockGeneration();
      const result = await sana.generateImage({ prompt: 'b' });
      expect(result.success).toBe(true);
    });
  });

  // --- generateImageToImage() ---

  describe('generateImageToImage()', () => {
    beforeEach(() => mockImg2Img());

    it('passes all params to native', async () => {
      const sana = await initSana();
      await sana.generateImageToImage({
        prompt: 'a watercolor',
        initImagePath: 'file:///input.png',
        options: { width: 512, height: 512, strength: 0.7 },
      });
      expect(getMock().generateImageToImage).toHaveBeenCalledWith(
        'a watercolor', '/input.png', 512, 512, 0.7, expect.anything()
      );
    });

    it('defaults strength to 0.6', async () => {
      const sana = await initSana();
      await sana.generateImageToImage({ prompt: 'x', initImagePath: '/in.png' });
      expect(getMock().generateImageToImage.mock.calls[0][4]).toBe(0.6);
    });

    it('returns imageUri', async () => {
      const sana = await initSana();
      const result = await sana.generateImageToImage({ prompt: 'x', initImagePath: '/in.png' });
      expect(result.imageUri).toBe('file:///tmp/sana_img2img.png');
    });
  });

  // --- destroy() ---

  describe('destroy()', () => {
    it('stops, destroys, and cleans temp files', async () => {
      const sana = await initSana();
      await sana.destroy();
      expect(getMock().stop).toHaveBeenCalled();
      expect(getMock().destroy).toHaveBeenCalled();
      expect(mockedFS.deleteTempFiles).toHaveBeenCalled();
    });

    it('no-op when not initialized', async () => {
      await new CactusSana().destroy();
      expect(getMock().destroy).not.toHaveBeenCalled();
    });

    it('allows re-init after destroy', async () => {
      const sana = await initSana();
      await sana.destroy();
      await sana.init();
      expect(getMock().init).toHaveBeenCalledTimes(2);
    });
  });

  // --- Temp files ---

  describe('temp files', () => {
    it('unique URIs per generation', async () => {
      mockGeneration(256, 256);
      mockedFS.writeTempPng
        .mockResolvedValueOnce('file:///tmp/001.png')
        .mockResolvedValueOnce('file:///tmp/002.png');
      const sana = await initSana();
      const r1 = await sana.generateImage({ prompt: 'a' });
      const r2 = await sana.generateImage({ prompt: 'b' });
      expect(r1.imageUri).not.toBe(r2.imageUri);
    });

    it('destroy cleans up', async () => {
      mockGeneration();
      const sana = await initSana();
      await sana.generateImage({ prompt: 'a' });
      await sana.destroy();
      expect(mockedFS.deleteTempFiles).toHaveBeenCalled();
    });
  });
});
