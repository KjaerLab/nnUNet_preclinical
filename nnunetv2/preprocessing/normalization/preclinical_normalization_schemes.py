from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from numpy import number
from scipy.ndimage import gaussian_filter1d

from nnunetv2.preprocessing.normalization.default_normalization_schemes import ImageNormalization


def find_histogram_peak(image: np.ndarray, intensity_min: float = None, intensity_max: float = None, 
                       bins: int = 256*2, sigma: float = 2.0) -> tuple:
    """
    Finds peak intensity and optionally plots the histogram (raw vs smoothed).
    """
    if intensity_min is None:
        intensity_min = np.min(image)
    if intensity_max is None:
        intensity_max = np.max(image)
    
    # 1. Create raw histogram
    hist, bin_edges = np.histogram(image, bins=bins, range=(intensity_min, intensity_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 2. Apply Gaussian Smoothing
    if sigma > 0:
        hist_smooth = gaussian_filter1d(hist, sigma=sigma)
    else:
        hist_smooth = hist
    
    # 3. Find peak
    peak_idx = np.argmax(hist_smooth)
    peak_count = hist_smooth[peak_idx]
    peak_intensity = bin_centers[peak_idx]
    
    return peak_intensity, peak_count

class CTNormalization_preclin(ImageNormalization):
    """
    CT image normalization using histogram peak-based intensity adjustment.
    
    This normalization method is more robust to poorly calibrated CT scanners by:
    - Finding the histogram peak within the valid intensity range instead of using 
      a fixed population mean
    - Using half the peak intensity as the offset, which better captures the 
      actual intensity distribution of the scan
    - Being less sensitive to outliers and scanner calibration variations
    
    The method is particularly useful when dealing with CT images from different 
    scanners or acquisition protocols with inconsistent HU (Hounsfield Unit) 
    calibration.
    
    Attributes:
        leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true (bool): 
            False - pixels outside mask are not zeroed
    
    Process:
        1. Find the peak of the intensity histogram within [lower_bound, upper_bound + std]
        2. Convert image to target dtype
        3. Subtract half the peak intensity as offset
        4. Clip intensities to [lower_bound, upper_bound]
        5. Normalize by standard deviation
    """
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"

        std_intensity = self.intensityproperties['std']
        lower_bound = self.intensityproperties['percentile_00_5']
        upper_bound = self.intensityproperties['percentile_99_5']

        peak_hist, _  = find_histogram_peak(image, intensity_min=lower_bound, intensity_max=upper_bound+std_intensity, sigma=0)

        image = image.astype(self.target_dtype, copy=False)
        image -= peak_hist/2
        np.clip(image, lower_bound, upper_bound, out=image)
        image /= max(std_intensity, 1e-8)

        # image = np.tanh(image)
        return image