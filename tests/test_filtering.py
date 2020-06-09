import pytest
import torch

from filtering import wiener

@pytest.fixture(params=[1, 2])
def nb_channels(request):
    return request.param

def test_wiener_shape(nb_channels):
    nb_frames = 10
    nb_bins = 257
    nb_sources = 1
    target_spectrograms = torch.rand(nb_frames, nb_bins, nb_channels, nb_sources)
    mix_stft = torch.rand(nb_frames, nb_bins, nb_channels, 2)
    wiener(target_spectrograms, mix_stft)
