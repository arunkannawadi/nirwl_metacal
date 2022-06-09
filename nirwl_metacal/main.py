import logging
import os
from typing import Any, List, Mapping, Optional

import fire
import galsim
import photutils
import yaml
from astropy.io import fits

__all__ = [
    "MetacalCatalogGenerator",
]

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


class MetacalCatalogGenerator:
    """
    Docstring
    """

    def __init__(
        self,
        config: Optional[str] = None,
        drizzle_image: Optional[str] = None,
        drizzle_weight: Optional[str] = None,
        noise_rms_map: Optional[str] = None,
        seg_map: Optional[str] = None,
        sep_cat: Optional[str] = None,
        psf_images: Optional[str] = None,
        output_cat: Optional[str] = None,
        overwrite: Optional[bool] = None,
        seed: Optional[int] = 1357,
        weight_fwhm=None,
    ):
        _empty_config: Mapping[str, Any] = {"name": None, "inputs": {}, "measurement": {}, "logging": {}}
        self.config = self._parse_config(config) if config else _empty_config
        self.config_path = config

        self.logger = logging.getLogger()  # root logger
        self.logger.setLevel(self.config.get("logging", {}).get("level", logging.INFO))

        self.drizzle_image_path = (
            drizzle_image if drizzle_image else self.config.get("inputs", {}).get("drizzle_image")
        )
        self.drizzle_weight_path = (
            drizzle_weight if drizzle_weight else self.config.get("inputs", {}).get("drizzle_weight")
        )
        self.noise_rms_map_path = (
            noise_rms_map if noise_rms_map else self.config.get("inputs", {}).get("noise_rms_map")
        )
        self.seg_map_path = seg_map if seg_map else self.config.get("inputs", {}).get("seg_map")
        self.sep_cat_path = sep_cat if sep_cat else self.config.get("inputs", {}).get("sep_cat")
        self.psf_images_path = psf_images if psf_images else self.config.get("inputs", {}).get("psf_images")

        self.output_cat_path = output_cat if output_cat else self.config.get("outputs", {}).get("output_cat")
        self.overwrite = (
            overwrite if overwrite is not None else self.config.get("outputs", {}).get("overwrite", False)
        )

        if self.output_cat_path is None:
            raise ValueError("output_cat is required")
        if self.overwrite is False:
            if os.path.exists(self.output_cat_path):
                raise FileExistsError(f"Output file {self.output_cat_path} already exists")

        for filepath in (
            "drizzle_image",
            "drizzle_weight",
            "noise_rms_map",
            "seg_map",
            "sep_cat",
            "psf_images",
        ):
            try:
                if not os.path.exists(filename := getattr(self, filepath + "_path")):
                    raise ValueError(filename + f" was specified as '{filepath}' and it was not found. ")
            except TypeError:
                self.logger.error(
                    "%s must be specified either in the config file or via command-line", filepath
                )
                raise FileNotFoundError(
                    "%s must be specified either in the config file or via command-line" % filepath
                )

        # Type hints only. These will be populated by the load_all method.
        self.drizzle_image: galsim.Image
        self.drizzle_weight: galsim.Image
        self.noise_rms_map: galsim.Image
        self.seg_map: photutils.SegmentationImage
        self.sep_cat: fits.fitsrec.FITS_rec
        self.psf_images: List[galsim.Image]

        self.seed = seed
        self.weight_fwhm = (
            weight_fwhm if weight_fwhm else self.config.get("measurement", {}).get("weight_fwhm")
        )

    @staticmethod
    def _parse_config(config_path: str) -> Mapping[str, Any]:
        """
        Parse the config file

        Parameters
        ----------
        config_path : str
            The path to the config file.

        Returns
        -------
        config : dict
            The config as a dictionary.
        """
        with open(config_path, "r") as f:
            config: Mapping[str, Any] = yaml.safe_load(f)
        return config

    def validate(self):
        """
        Validate that the inputs are sensible

        Parameters
        ----------
        drizzle_image : str
            The drizzled image

        """
        assert self.drizzle_image.array.shape == self.drizzle_weight.array.shape
        assert self.drizzle_image.array.shape == self.seg_map.shape
        assert self.drizzle_image.array.shape == self.noise_rms_map.array.shape
        assert len(self.psf_images) == self.sep_cat.size
        self.logger.info("ALl is well")

    def load_all(self):
        hdu_list = None
        for attr in (
            "drizzle_image",
            "drizzle_weight",
            "noise_rms_map",
        ):
            try:
                hdu_list = fits.open(getattr(self, attr + "_path"))
                hdu = hdu_list[0]
                assert hdu.is_image
                setattr(self, attr, galsim.Image(hdu.data))
            except (AssertionError, IndexError, IOError) as e:
                self.logger.error(e)
                raise e
            finally:
                if hdu_list is not None:
                    hdu_list.close()

        try:
            hdu_list = fits.open(self.seg_map_path)
            hdu = hdu_list[0]
            assert hdu.is_image
            self.seg_map = photutils.SegmentationImage(hdu.data)
        except (AssertionError, IndexError, IOError) as e:
            self.logger.error(e)
            raise e
        finally:
            if hdu_list is not None:
                hdu_list.close()

        try:
            self.psf_images = galsim.fits.readCube(self.psf_images_path, hdu=1)
        except Exception as e:
            self.logger.error(e)
            raise e

        try:
            hdu_list = fits.open(self.sep_cat_path)
            hdu = hdu_list[1]
            assert not hdu.is_image
            self.sep_cat = hdu.data
        except (AssertionError, IndexError, IOError) as e:
            self.logger.error(e)
            raise e
        finally:
            if hdu_list is not None:
                hdu_list.close()

    def run(self):
        """
        Run the metacal catalog generation
        """
        self.load_all()
        self.validate()


if __name__ == "__main__":
    fire.Fire(MetacalCatalogGenerator)
