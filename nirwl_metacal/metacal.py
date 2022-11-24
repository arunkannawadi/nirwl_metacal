import logging
from itertools import product
from typing import NamedTuple, Sequence, Union

import galsim
import ngmix
import numpy as np
import utils
from astropy.io import fits
from astropy.table import Table
from metacal_record import MetacalRecord

__all__ = [
    "MetacalRecordGenerator",
]


class MetacalRecordGenerator:
    """
    A class that makes metacal measurement for a single source.
    """

    def __init__(self, config, seed=1357, weight_fwhm=None):
        self.config = config
        self.seed = seed
        self.rng = np.random.RandomState(seed=self.seed)
        self.galsim_rng = galsim.BaseDeviate(seed=seed)
        self.boot = self._setup_metacal(self.rng, weight_fwhm)

    def measure(self, n, image, weight, noise_rms, seg_map, bbox, sep_record, psf_image) -> MetacalRecord:
        """
        Measure the metacal for a single source.

        This is the entry point to the metacal module from the main module.

        Parameters
        ----------
        image : galsim.image
            The image of the source.
        weight : galsim.image
            The weight image of the source.
        noise_rms : galsim.Image
            The noise rms of the image.
        seg_map : galsim.Image
            The segmentation map.
        bbox : galsim.Bounds
            A minimal bounding box for the source.
        sep_record : NamedTuple
            The SExtractor record corresponding to the source.
        psf_image : galsim.Image
            The PSF image at the centroid of the source.

        Notes
        -----
        The ``image``, ``weight``, ``noise_rms`` and ``seg_map`` are all views
        and shared amongst different calls of `measure`.
        A deep copy must be made before modifying any of them to avoid
        infering measurement of other sources.
        """
        ## Modfiy the bbox
        center = None
        expanded_bbox = utils.expand_bbox(
            bbox,
            mosaic_bbox=image.bounds,
            center=center,
            min_size=self.config.get("minimum_stamp_size", 32),
        )
        expanded_bbox &= image.bounds
        ## Make a deep copy of the image
        im = image[expanded_bbox].copy()
        ## Replace the pixels belonging to other sources with noise.
        noise_im = self._make_noise_image(noise_rms[expanded_bbox])
        mask = ~((seg_map[expanded_bbox].array == n) | (seg_map[expanded_bbox].array == 0))
        im.array[mask] = noise_im.array[mask]

        obs = self._make_ngmix_observation(im, weight[expanded_bbox], psf_image)
        resdict, _ = self.boot.go(obs)
        record = self._make_record(n, resdict)
        return record

    @staticmethod
    def _setup_metacal(rng, weight_fwhm=None):
        if weight_fwhm is None:
            fitter = ngmix.admom.AdmomFitter()
            psf_fitter = ngmix.admom.AdmomFitter()
            # guesser = ngmix.guessers.TFluxGuesser(rng, T=0.5, flux=100.0)
            # psf_guesser = ngmix.guessers.GMixCoellipPSF(rng, ngauss=1, guess_from_moms=True)
            guesser = ngmix.guessers.GMixPSFGuesser(rng, ngauss=1, guess_from_moms=True)
            psf_guesser = ngmix.guessers.GMixPSFGuesser(rng, ngauss=1, guess_from_moms=True)
        else:
            fitter = ngmix.gaussmom.GaussMomFitter(weight_fwhm=weight_fwhm)
            psf_fitter = ngmix.gaussmom.GaussMomFitter(weight_fwhm=weight_fwhm)
            guesser, psf_guesser = None, None

        # these "runners" run the measurement code on observations
        psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, guesser=psf_guesser, ntry=2)
        runner = ngmix.runners.Runner(fitter=fitter, guesser=guesser)

        boot = ngmix.metacal.MetacalBootstrapper(
            runner=runner,
            psf_runner=psf_runner,
            rng=rng,
            types=["noshear", "1p", "1m", "2p", "2m"],
        )

        return boot

    def _make_ngmix_observation(
        self,
        image: galsim.Image,
        weight: galsim.Image,
        psf_image: galsim.Image,
    ) -> ngmix.Observation:
        """Make an ngmix.Observation instance"""
        jac = None
        psf_obs = ngmix.Observation(psf_image.array, jacobian=jac)
        bmask = None  # get_mask_from_seg_map(seg_map, bbox, i)
        obs = ngmix.Observation(
            image.array,
            weight.array,
            bmask=bmask,
            psf=psf_obs,
            jacobian=jac,
        )
        return obs

    def _make_noise_image(self, noise_rms: galsim.Image) -> galsim.Image:
        noise_generator = galsim.GaussianNoise(self.galsim_rng, sigma=1.0)
        noise_img = galsim.Image(bounds=noise_rms.bounds)
        noise_img.addNoise(noise_generator)
        noise_img *= noise_rms
        return noise_img

    def _make_record(self, record_index, resdict) -> MetacalRecord:
        record = {"index": record_index}
        record["e1"] = resdict["noshear"]["e1"]
        record["e2"] = resdict["noshear"]["e2"]
        record["e1err"] = resdict["noshear"]["e1err"]
        record["e2err"] = resdict["noshear"]["e2err"]
        record["snr"] = resdict["noshear"]["s2n"]

        for a, b in product((1, 2), (1, 2)):
            record[f"R{a}{b}_p"] = (resdict[f"{b}p"][f"e{a}"] - resdict["noshear"][f"e{a}"]) / 0.01
            record[f"R{a}{b}_m"] = (-resdict[f"{b}m"][f"e{a}"] + resdict["noshear"][f"e{a}"]) / 0.01
            record[f"R{a}{b}"] = 0.5 * (record[f"R{a}{b}_p"] + record[f"R{a}{b}_m"])

        R11_p = (resdict["1p"]["e1"] - resdict["noshear"]["e1"]) / 0.01
        R11_m = (resdict["noshear"]["e1"] - resdict["1m"]["e1"]) / 0.01
        R11 = 0.5 * (R11_p + R11_m)
        record["R11_p"] = R11_p
        record["R11_m"] = R11_m
        record["R11"] = R11

        R22_p = (resdict["2p"]["e2"] - resdict["noshear"]["e2"]) / 0.01
        R22_m = (resdict["noshear"]["e2"] - resdict["2m"]["e2"]) / 0.01
        R22 = 0.5 * (R22_p + R22_m)
        record["R22_p"] = R22_p
        record["R22_m"] = R22_m
        record["R22"] = R22

        R12_p = (resdict["2p"]["e1"] - resdict["noshear"]["e1"]) / 0.01
        R12_m = (resdict["noshear"]["e1"] - resdict["2m"]["e1"]) / 0.01
        R12 = 0.5 * (R12_p + R12_m)
        record["R12_p"] = R12_p
        record["R12_m"] = R12_m
        record["R12"] = R12

        R21_p = (resdict["1p"]["e2"] - resdict["noshear"]["e2"]) / 0.01
        R21_m = (resdict["noshear"]["e2"] - resdict["1m"]["e2"]) / 0.01
        R21 = 0.5 * (R21_p + R21_m)
        record["R21_p"] = R21_p
        record["R21_m"] = R21_m
        record["R21"] = R21

        record["flag_noshear"] = resdict["noshear"]["flags"]
        record["flag_1p"] = resdict["1p"]["flags"]
        record["flag_1m"] = resdict["1m"]["flags"]
        record["flag_2p"] = resdict["2p"]["flags"]
        record["flag_2m"] = resdict["2m"]["flags"]

        return MetacalRecord(**record)
