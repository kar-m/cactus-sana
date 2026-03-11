import { Cactus, CactusFileSystem } from '../native';
import type {
  CactusSanaDownloadParams,
  CactusSanaGenerateImageParams,
  CactusSanaGenerateImageResult,
  CactusSanaGenerateImageToImageParams,
  CactusSanaGenerateImageToImageResult,
  CactusSanaParams,
} from '../types/CactusSana';
import { getRegistry } from '../modelRegistry';

export class CactusSana {
  private readonly cactus = new Cactus();

  private readonly model: string;
  private readonly options: {
    quantization: 'int4' | 'int8';
    pro: boolean;
  };

  private isDownloading = false;
  private isInitialized = false;
  private isGenerating = false;

  private static readonly defaultModel = 'sana-sprint-0.6b';
  private static readonly defaultOptions = {
    quantization: 'int8' as const,
    pro: false,
  };
  private static readonly defaultWidth = 1024;
  private static readonly defaultHeight = 1024;
  private static readonly defaultStrength = 0.6;

  constructor({ model, options }: CactusSanaParams = {}) {
    this.model = model ?? CactusSana.defaultModel;
    this.options = {
      quantization:
        options?.quantization ?? CactusSana.defaultOptions.quantization,
      pro: options?.pro ?? CactusSana.defaultOptions.pro,
    };
  }

  public async download({
    onProgress,
  }: CactusSanaDownloadParams = {}): Promise<void> {
    if (this.isModelPath(this.model)) {
      onProgress?.(1.0);
      return;
    }

    if (this.isDownloading) {
      throw new Error('CactusSana is already downloading');
    }

    if (await CactusFileSystem.modelExists(this.getModelName())) {
      onProgress?.(1.0);
      return;
    }

    this.isDownloading = true;
    try {
      const registry = await getRegistry();
      const modelConfig =
        registry[this.model]?.quantization[this.options.quantization];
      const url = this.options.pro ? modelConfig?.pro?.apple : modelConfig?.url;

      if (!url) {
        throw new Error(
          `Model ${this.model} with specified options not found`
        );
      }

      await CactusFileSystem.downloadModel(
        this.getModelName(),
        url,
        onProgress
      );
    } finally {
      this.isDownloading = false;
    }
  }

  public async init(): Promise<void> {
    if (this.isInitialized) {
      return;
    }

    let modelPath: string;
    if (this.isModelPath(this.model)) {
      modelPath = this.model.replace('file://', '');
    } else {
      if (!(await CactusFileSystem.modelExists(this.getModelName()))) {
        throw new Error(
          `Model "${this.model}" with options ${JSON.stringify(this.options)} is not downloaded`
        );
      }
      modelPath = await CactusFileSystem.getModelPath(this.getModelName());
    }

    const cacheDir = await CactusFileSystem.getCactusDirectory();
    await this.cactus.setTelemetryEnvironment(cacheDir);
    await this.cactus.init(modelPath);
    this.isInitialized = true;
  }

  public async generateImage({
    prompt,
    options,
  }: CactusSanaGenerateImageParams): Promise<CactusSanaGenerateImageResult> {
    if (this.isGenerating) {
      throw new Error('CactusSana is already generating');
    }

    const width = options?.width ?? CactusSana.defaultWidth;
    const height = options?.height ?? CactusSana.defaultHeight;

    await this.init();

    this.isGenerating = true;
    try {
      const responseJson = await this.cactus.generateImage(
        prompt,
        width,
        height,
        options
      );

      const response = JSON.parse(responseJson);
      if (!response.success) {
        throw new Error(response.error ?? 'Image generation failed');
      }

      const pixels = await this.cactus.getLastImagePixelsRgb();
      const imageUri = await CactusFileSystem.writeTempPng(
        pixels,
        response.width,
        response.height
      );

      return {
        success: true,
        imageUri,
        width: response.width,
        height: response.height,
        totalTimeMs: response.total_time_ms ?? 0,
      };
    } finally {
      this.isGenerating = false;
    }
  }

  public async generateImageToImage({
    prompt,
    initImagePath,
    options,
  }: CactusSanaGenerateImageToImageParams): Promise<CactusSanaGenerateImageToImageResult> {
    if (this.isGenerating) {
      throw new Error('CactusSana is already generating');
    }

    const width = options?.width ?? CactusSana.defaultWidth;
    const height = options?.height ?? CactusSana.defaultHeight;
    const strength = options?.strength ?? CactusSana.defaultStrength;

    await this.init();

    this.isGenerating = true;
    try {
      const responseJson = await this.cactus.generateImageToImage(
        prompt,
        initImagePath.replace('file://', ''),
        width,
        height,
        strength,
        options
      );

      const response = JSON.parse(responseJson);
      if (!response.success) {
        throw new Error(response.error ?? 'Image-to-image generation failed');
      }

      const pixels = await this.cactus.getLastImagePixelsRgb();
      const imageUri = await CactusFileSystem.writeTempPng(
        pixels,
        response.width,
        response.height
      );

      return {
        success: true,
        imageUri,
        width: response.width,
        height: response.height,
        totalTimeMs: response.total_time_ms ?? 0,
      };
    } finally {
      this.isGenerating = false;
    }
  }

  public stop(): Promise<void> {
    return this.cactus.stop();
  }

  public async destroy(): Promise<void> {
    if (!this.isInitialized) {
      return;
    }

    await this.cactus.stop();
    await this.cactus.destroy();
    await CactusFileSystem.deleteTempFiles();
    this.isInitialized = false;
  }

  private isModelPath(model: string): boolean {
    return model.startsWith('file://') || model.startsWith('/');
  }

  public getModelName(): string {
    return `${this.model}-${this.options.quantization}${this.options.pro ? '-pro' : ''}`;
  }
}
