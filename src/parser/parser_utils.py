import numpy as np
from typing import Optional, List
from skimage.transform import resize


def interpolate(
    input: np.ndarray,
    size: Optional[List[int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> np.ndarray:
    if mode not in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area', 'nearest-exact']:
        raise ValueError("Unsupported interpolation mode")

    if size is not None and scale_factor is not None:
        raise ValueError("Only one of size or scale_factor should be defined")

    if mode in ("nearest", "area", "nearest-exact"):
        if align_corners is not None:
            raise ValueError(
                "align_corners option can only be set with the "
                "interpolating modes: linear | bilinear | bicubic | trilinear"
            )
    else:
        if align_corners is None:
            align_corners = False

    dim = input.ndim - 2  # Number of spatial dimensions.

    if size is not None and scale_factor is None:
        output_size = size
    elif scale_factor is not None and size is None:
        output_size = [int(input.shape[i + 2] * scale_factor) for i in range(dim)]
    else:
        raise ValueError("Either size or scale_factor should be defined")

    if mode == 'nearest':
        if input.ndim == 3:
            # Implement nearest neighbor interpolation for 1D
            raise NotImplementedError("Nearest neighbor interpolation is not implemented for 1D numpy arrays")
        elif input.ndim == 4:
            # Implement nearest neighbor interpolation for 2D
            raise NotImplementedError("Nearest neighbor interpolation is not implemented for 2D numpy arrays")
        elif input.ndim == 5:
            # Implement nearest neighbor interpolation for 3D
            raise NotImplementedError("Nearest neighbor interpolation is not implemented for 3D numpy arrays")
    elif mode == 'linear':
        if input.ndim == 3:
            # Implement linear interpolation for 1D
            raise NotImplementedError("Linear interpolation is not implemented for 1D numpy arrays")
        elif input.ndim == 4:
            # Implement bilinear interpolation for 2D
            raise NotImplementedError("Bilinear interpolation is not implemented for 2D numpy arrays")
        elif input.ndim == 5:
            # Implement trilinear interpolation for 3D
            raise NotImplementedError("Trilinear interpolation is not implemented for 3D numpy arrays")
    elif mode == 'bilinear':
        if input.ndim != 3 and input.ndim != 4:
            raise ValueError(f"Unsupported input dimensions {input.ndim} for bilinear interpolation")
        # Implement bilinear interpolation for 2D or 3D
        # Use skimage's resize function for this purpose

        output = resize(input, output_size, mode='constant', order=1, anti_aliasing=antialias)
    elif mode == 'bicubic':
        if input.ndim != 4:
            raise ValueError(f"Unsupported input dimensions {input.ndim} for bicubic interpolation")
        # Implement bicubic interpolation for 2D
        # Use skimage's resize function for this purpose
        output = resize(input, output_size, mode='constant', order=3, anti_aliasing=antialias)
    elif mode == 'trilinear':
        if input.ndim != 5:
            raise ValueError(f"Unsupported input dimensions {input.ndim} for trilinear interpolation")
        # Implement trilinear interpolation for 3D
        # Use skimage's resize function for this purpose
        output = resize(input, output_size, mode='constant', order=1, anti_aliasing=antialias)
    elif mode == 'area':
        if input.ndim != 3 and input.ndim != 4:
            raise ValueError(f"Unsupported input dimensions {input.ndim} for area interpolation")
        # Implement area interpolation for 2D or 3D
        # Use skimage's resize function for this purpose
        output = resize(input, output_size, mode='reflect', order=0)
    elif mode == 'nearest-exact':
        if input.ndim != 3 and input.ndim != 4:
            raise ValueError(f"Unsupported input dimensions {input.ndim} for nearest-exact interpolation")
        # Implement nearest-exact interpolation for 2D or 3D
        # Use skimage's resize function for this purpose
        output = resize(input, output_size, mode='constant', order=0)

    return output
