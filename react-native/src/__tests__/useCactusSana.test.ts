import { renderHook, act } from '@testing-library/react';
import { useCactusSana } from '../hooks/useCactusSana';
import { CactusSana } from '../classes/CactusSana';
import { CactusFileSystem } from '../native/CactusFileSystem';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockInstance = {
  download: jest.fn(),
  init: jest.fn(),
  generateImage: jest.fn(),
  generateImageToImage: jest.fn(),
  stop: jest.fn(),
  destroy: jest.fn(),
  getModelName: jest.fn().mockReturnValue('sana-sprint-0.6b-int8'),
};

jest.mock('../classes/CactusSana', () => ({
  CactusSana: jest.fn().mockImplementation(() => mockInstance),
}));

jest.mock('react-native-nitro-modules', () => ({
  NitroModules: { createHybridObject: () => ({}) },
}));

jest.mock('../native/CactusFileSystem', () => ({
  CactusFileSystem: {
    modelExists: jest.fn().mockResolvedValue(false),
  },
}));

jest.mock('../modelRegistry', () => ({
  getRegistry: jest.fn().mockResolvedValue({}),
}));

const mockedFS = CactusFileSystem as jest.Mocked<typeof CactusFileSystem>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('useCactusSana', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockedFS.modelExists.mockResolvedValue(false);
    mockInstance.download.mockResolvedValue(undefined);
    mockInstance.init.mockResolvedValue(undefined);
    mockInstance.generateImage.mockResolvedValue({
      success: true, imageUri: 'file:///tmp/result.png', width: 1024, height: 1024, totalTimeMs: 1500,
    });
    mockInstance.generateImageToImage.mockResolvedValue({
      success: true, imageUri: 'file:///tmp/img2img.png', width: 512, height: 512, totalTimeMs: 800,
    });
    mockInstance.stop.mockResolvedValue(undefined);
    mockInstance.destroy.mockResolvedValue(undefined);
  });

  // --- Initial state ---

  describe('initial state', () => {
    it('has correct defaults', () => {
      const { result } = renderHook(() => useCactusSana());
      expect(result.current.isDownloaded).toBe(false);
      expect(result.current.isDownloading).toBe(false);
      expect(result.current.downloadProgress).toBe(0);
      expect(result.current.isInitializing).toBe(false);
      expect(result.current.isGenerating).toBe(false);
      expect(result.current.generationStep).toEqual({ step: 0, total: 0 });
      expect(result.current.imageUri).toBeNull();
      expect(result.current.error).toBeNull();
    });

    it('creates CactusSana with params', () => {
      renderHook(() => useCactusSana({ model: 'sana-1.0', options: { quantization: 'int4' } }));
      expect(CactusSana).toHaveBeenCalledWith(
        expect.objectContaining({ model: 'sana-1.0', options: expect.objectContaining({ quantization: 'int4' }) })
      );
    });

    it('checks modelExists on mount', async () => {
      mockedFS.modelExists.mockResolvedValue(true);
      const { result } = renderHook(() => useCactusSana());
      await act(async () => {});
      expect(result.current.isDownloaded).toBe(true);
    });
  });

  // --- download() ---

  describe('download()', () => {
    it('sets isDownloading while downloading', async () => {
      let resolve!: () => void;
      mockInstance.download.mockImplementation(() => new Promise<void>((r) => { resolve = r; }));
      const { result } = renderHook(() => useCactusSana());
      act(() => { result.current.download(); });
      expect(result.current.isDownloading).toBe(true);
      await act(async () => resolve());
      expect(result.current.isDownloading).toBe(false);
      expect(result.current.isDownloaded).toBe(true);
    });

    it('reports progress', async () => {
      const progress: number[] = [];
      mockInstance.download.mockImplementation(async ({ onProgress }: any) => {
        onProgress?.(0); onProgress?.(0.5); onProgress?.(1.0);
      });
      const { result } = renderHook(() => useCactusSana());
      await act(async () => {
        await result.current.download({ onProgress: (p) => progress.push(p) });
      });
      expect(progress).toEqual([0, 0.5, 1.0]);
    });

    it('sets error on failure', async () => {
      mockInstance.download.mockRejectedValue(new Error('Network timeout'));
      const { result } = renderHook(() => useCactusSana());
      await act(async () => { await result.current.download().catch(() => {}); });
      expect(result.current.error).toBe('Network timeout');
      expect(result.current.isDownloading).toBe(false);
    });

    it('throws when already downloading', async () => {
      mockInstance.download.mockImplementation(() => new Promise(() => {}));
      const { result } = renderHook(() => useCactusSana());
      act(() => { result.current.download(); });
      await act(async () => {
        await expect(result.current.download()).rejects.toThrow('already downloading');
      });
    });

    it('no-op when already downloaded', async () => {
      mockedFS.modelExists.mockResolvedValue(true);
      const { result } = renderHook(() => useCactusSana());
      await act(async () => {});
      await act(async () => { await result.current.download(); });
      expect(mockInstance.download).not.toHaveBeenCalled();
    });
  });

  // --- init() ---

  describe('init()', () => {
    it('sets isInitializing', async () => {
      let resolve!: () => void;
      mockInstance.init.mockImplementation(() => new Promise<void>((r) => { resolve = r; }));
      const { result } = renderHook(() => useCactusSana());
      act(() => { result.current.init(); });
      expect(result.current.isInitializing).toBe(true);
      await act(async () => resolve());
      expect(result.current.isInitializing).toBe(false);
    });

    it('sets error on failure', async () => {
      mockInstance.init.mockRejectedValue(new Error('not downloaded'));
      const { result } = renderHook(() => useCactusSana());
      await act(async () => { await result.current.init().catch(() => {}); });
      expect(result.current.error).toBe('not downloaded');
    });

    it('throws when already initializing', async () => {
      mockInstance.init.mockImplementation(() => new Promise(() => {}));
      const { result } = renderHook(() => useCactusSana());
      act(() => { result.current.init(); });
      await act(async () => {
        await expect(result.current.init()).rejects.toThrow('already initializing');
      });
    });
  });

  // --- generateImage() ---

  describe('generateImage()', () => {
    it('sets isGenerating and imageUri', async () => {
      let resolve!: (v: any) => void;
      mockInstance.generateImage.mockImplementation(() => new Promise((r) => { resolve = r; }));
      const { result } = renderHook(() => useCactusSana());
      act(() => { result.current.generateImage({ prompt: 'cat' }); });
      expect(result.current.isGenerating).toBe(true);
      await act(async () => resolve({
        success: true, imageUri: 'file:///tmp/cat.png', width: 1024, height: 1024, totalTimeMs: 1000,
      }));
      expect(result.current.isGenerating).toBe(false);
      expect(result.current.imageUri).toBe('file:///tmp/cat.png');
    });

    it('passes onStep and updates generationStep', async () => {
      mockInstance.generateImage.mockImplementation(async (params: any) => {
        params.onStep?.(1, 4); params.onStep?.(4, 4);
        return { success: true, imageUri: 'file:///tmp/r.png', width: 1024, height: 1024, totalTimeMs: 1000 };
      });
      const onStep = jest.fn();
      const { result } = renderHook(() => useCactusSana());
      await act(async () => { await result.current.generateImage({ prompt: 'a', onStep }); });
      expect(onStep).toHaveBeenCalledWith(4, 4);
      expect(result.current.generationStep.total).toBe(4);
    });

    it('sets error on failure', async () => {
      mockInstance.generateImage.mockRejectedValue(new Error('GPU OOM'));
      const { result } = renderHook(() => useCactusSana());
      await act(async () => { await result.current.generateImage({ prompt: 'a' }).catch(() => {}); });
      expect(result.current.error).toBe('GPU OOM');
      expect(result.current.isGenerating).toBe(false);
    });

    it('throws when already generating', async () => {
      mockInstance.generateImage.mockImplementation(() => new Promise(() => {}));
      const { result } = renderHook(() => useCactusSana());
      act(() => { result.current.generateImage({ prompt: 'a' }); });
      await act(async () => {
        await expect(result.current.generateImage({ prompt: 'b' })).rejects.toThrow('already generating');
      });
    });
  });

  // --- generateImageToImage() ---

  describe('generateImageToImage()', () => {
    it('passes params and sets imageUri', async () => {
      const { result } = renderHook(() => useCactusSana());
      await act(async () => {
        await result.current.generateImageToImage({
          prompt: 'painting', initImagePath: 'file:///input.png', options: { strength: 0.8 },
        });
      });
      expect(mockInstance.generateImageToImage).toHaveBeenCalledWith(
        expect.objectContaining({ prompt: 'painting', initImagePath: 'file:///input.png' })
      );
      expect(result.current.imageUri).toBe('file:///tmp/img2img.png');
    });

    it('sets error on failure', async () => {
      mockInstance.generateImageToImage.mockRejectedValue(new Error('Bad image'));
      const { result } = renderHook(() => useCactusSana());
      await act(async () => {
        await result.current.generateImageToImage({ prompt: 'a', initImagePath: '/x.png' }).catch(() => {});
      });
      expect(result.current.error).toBe('Bad image');
    });
  });

  // --- stop() ---

  describe('stop()', () => {
    it('calls stop', async () => {
      const { result } = renderHook(() => useCactusSana());
      await act(async () => { await result.current.stop(); });
      expect(mockInstance.stop).toHaveBeenCalled();
    });
  });

  // --- destroy() ---

  describe('destroy()', () => {
    it('resets imageUri and generationStep', async () => {
      const { result } = renderHook(() => useCactusSana());
      await act(async () => { await result.current.generateImage({ prompt: 'a' }); });
      expect(result.current.imageUri).not.toBeNull();
      await act(async () => { await result.current.destroy(); });
      expect(result.current.imageUri).toBeNull();
      expect(result.current.generationStep).toEqual({ step: 0, total: 0 });
    });
  });

  // --- Cleanup ---

  describe('cleanup', () => {
    it('destroys on unmount', () => {
      const { unmount } = renderHook(() => useCactusSana());
      unmount();
      expect(mockInstance.destroy).toHaveBeenCalled();
    });
  });

  // --- Model change ---

  describe('model change', () => {
    it('creates new instance and resets state', () => {
      const { result, rerender } = renderHook(
        ({ model }: { model: string }) => useCactusSana({ model }),
        { initialProps: { model: 'sana-sprint-0.6b' } }
      );
      const countBefore = (CactusSana as unknown as jest.Mock).mock.calls.length;
      rerender({ model: 'sana-1.0' });
      expect((CactusSana as unknown as jest.Mock).mock.calls.length).toBeGreaterThan(countBefore);
      expect(result.current.isDownloaded).toBe(false);
      expect(result.current.imageUri).toBeNull();
    });
  });
});
