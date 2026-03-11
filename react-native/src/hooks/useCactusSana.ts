import { useCallback, useEffect, useState, useRef } from 'react';
import { CactusSana } from '../classes/CactusSana';
import { CactusFileSystem } from '../native';
import { getErrorMessage } from '../utils/error';
import type {
  CactusSanaParams,
  CactusSanaDownloadParams,
  CactusSanaGenerateImageParams,
  CactusSanaGenerateImageResult,
  CactusSanaGenerateImageToImageParams,
  CactusSanaGenerateImageToImageResult,
} from '../types/CactusSana';

export const useCactusSana = ({
  model = 'sana-sprint-0.6b',
  options: modelOptions = {
    quantization: undefined,
    pro: false,
  },
}: CactusSanaParams = {}) => {
  const [cactusSana, setCactusSana] = useState(
    () => new CactusSana({ model, options: modelOptions })
  );

  const [isGenerating, setIsGenerating] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [isDownloaded, setIsDownloaded] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [generationStep, setGenerationStep] = useState<{
    step: number;
    total: number;
  }>({ step: 0, total: 0 });
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const currentModelRef = useRef(model);
  const currentDownloadIdRef = useRef(0);

  useEffect(() => {
    currentModelRef.current = model;
  }, [model]);

  useEffect(() => {
    const newInstance = new CactusSana({
      model,
      options: {
        quantization: modelOptions.quantization,
        pro: modelOptions.pro,
      },
    });
    setCactusSana(newInstance);

    setIsGenerating(false);
    setIsInitializing(false);
    setIsDownloaded(false);
    setIsDownloading(false);
    setDownloadProgress(0);
    setGenerationStep({ step: 0, total: 0 });
    setImageUri(null);
    setError(null);

    let mounted = true;
    CactusFileSystem.modelExists(newInstance.getModelName())
      .then((exists) => {
        if (!mounted) return;
        setIsDownloaded(exists);
      })
      .catch((e) => {
        if (!mounted) return;
        setIsDownloaded(false);
        setError(getErrorMessage(e));
      });

    return () => {
      mounted = false;
    };
  }, [model, modelOptions.quantization, modelOptions.pro]);

  useEffect(() => {
    return () => {
      cactusSana.destroy().catch(() => {});
    };
  }, [cactusSana]);

  const download = useCallback(
    async ({ onProgress }: CactusSanaDownloadParams = {}) => {
      if (isDownloading) {
        const message = 'CactusSana is already downloading';
        setError(message);
        throw new Error(message);
      }

      setError(null);

      if (isDownloaded) {
        return;
      }

      const thisModel = currentModelRef.current;
      const thisDownloadId = ++currentDownloadIdRef.current;

      setDownloadProgress(0);
      setIsDownloading(true);
      try {
        await cactusSana.download({
          onProgress: (progress) => {
            if (
              currentModelRef.current !== thisModel ||
              currentDownloadIdRef.current !== thisDownloadId
            ) {
              return;
            }
            setDownloadProgress(progress);
            onProgress?.(progress);
          },
        });

        if (
          currentModelRef.current !== thisModel ||
          currentDownloadIdRef.current !== thisDownloadId
        ) {
          return;
        }

        setIsDownloaded(true);
      } catch (e) {
        if (
          currentModelRef.current !== thisModel ||
          currentDownloadIdRef.current !== thisDownloadId
        ) {
          return;
        }
        setError(getErrorMessage(e));
        throw e;
      } finally {
        if (
          currentModelRef.current !== thisModel ||
          currentDownloadIdRef.current !== thisDownloadId
        ) {
          return;
        }
        setIsDownloading(false);
        setDownloadProgress(0);
      }
    },
    [cactusSana, isDownloading, isDownloaded]
  );

  const init = useCallback(async () => {
    if (isInitializing) {
      const message = 'CactusSana is already initializing';
      setError(message);
      throw new Error(message);
    }

    setError(null);
    setIsInitializing(true);
    try {
      await cactusSana.init();
    } catch (e) {
      setError(getErrorMessage(e));
      throw e;
    } finally {
      setIsInitializing(false);
    }
  }, [cactusSana, isInitializing]);

  const generateImage = useCallback(
    async (
      params: CactusSanaGenerateImageParams
    ): Promise<CactusSanaGenerateImageResult> => {
      if (isGenerating) {
        const message = 'CactusSana is already generating';
        setError(message);
        throw new Error(message);
      }

      setError(null);
      setImageUri(null);
      setGenerationStep({ step: 0, total: 0 });
      setIsGenerating(true);
      try {
        const result = await cactusSana.generateImage({
          ...params,
          onStep: (step, total) => {
            setGenerationStep({ step, total });
            params.onStep?.(step, total);
          },
        });
        setImageUri(result.imageUri);
        return result;
      } catch (e) {
        setError(getErrorMessage(e));
        throw e;
      } finally {
        setIsGenerating(false);
      }
    },
    [cactusSana, isGenerating]
  );

  const generateImageToImage = useCallback(
    async (
      params: CactusSanaGenerateImageToImageParams
    ): Promise<CactusSanaGenerateImageToImageResult> => {
      if (isGenerating) {
        const message = 'CactusSana is already generating';
        setError(message);
        throw new Error(message);
      }

      setError(null);
      setImageUri(null);
      setGenerationStep({ step: 0, total: 0 });
      setIsGenerating(true);
      try {
        const result = await cactusSana.generateImageToImage({
          ...params,
          onStep: (step, total) => {
            setGenerationStep({ step, total });
            params.onStep?.(step, total);
          },
        });
        setImageUri(result.imageUri);
        return result;
      } catch (e) {
        setError(getErrorMessage(e));
        throw e;
      } finally {
        setIsGenerating(false);
      }
    },
    [cactusSana, isGenerating]
  );

  const stop = useCallback(async () => {
    setError(null);
    try {
      await cactusSana.stop();
    } catch (e) {
      setError(getErrorMessage(e));
      throw e;
    }
  }, [cactusSana]);

  const destroy = useCallback(async () => {
    setError(null);
    try {
      await cactusSana.destroy();
    } catch (e) {
      setError(getErrorMessage(e));
      throw e;
    } finally {
      setImageUri(null);
      setGenerationStep({ step: 0, total: 0 });
    }
  }, [cactusSana]);

  return {
    isGenerating,
    isInitializing,
    isDownloaded,
    isDownloading,
    downloadProgress,
    generationStep,
    imageUri,
    error,

    download,
    init,
    generateImage,
    generateImageToImage,
    stop,
    destroy,
  };
};
