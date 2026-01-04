"""
Vendored temporal_autoencoder package from Stream-DiffVSR upstream.

This package contains the TemporalAutoencoderTiny model which adds
temporal processing modules (TPM) to the standard AutoencoderTiny.

Source: https://github.com/jamichss/Stream-DiffVSR
License: Apache-2.0
"""

from .autoencoder_tiny import TemporalAutoencoderTiny, TemporalAutoencoderTinyOutput

__all__ = ["TemporalAutoencoderTiny", "TemporalAutoencoderTinyOutput"]
