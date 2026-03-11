import type { CactusModelOptions } from './common';

export interface CactusSanaParams {
  model?: string;
  options?: CactusModelOptions;
}

export interface CactusSanaDownloadParams {
  onProgress?: (progress: number) => void;
}

export interface CactusSanaGenerateImageOptions {
  width?: number;
  height?: number;
  steps?: number;
  seed?: number;
  guidanceScale?: number;
}

export interface CactusSanaGenerateImageParams {
  prompt: string;
  options?: CactusSanaGenerateImageOptions;
  onStep?: (step: number, totalSteps: number) => void;
}

export interface CactusSanaGenerateImageResult {
  success: boolean;
  imageUri: string;
  width: number;
  height: number;
  totalTimeMs: number;
}

export interface CactusSanaGenerateImageToImageOptions
  extends CactusSanaGenerateImageOptions {
  strength?: number;
}

export interface CactusSanaGenerateImageToImageParams {
  prompt: string;
  initImagePath: string;
  options?: CactusSanaGenerateImageToImageOptions;
  onStep?: (step: number, totalSteps: number) => void;
}

export interface CactusSanaGenerateImageToImageResult {
  success: boolean;
  imageUri: string;
  width: number;
  height: number;
  totalTimeMs: number;
}
