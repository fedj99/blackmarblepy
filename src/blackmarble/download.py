import asyncio
import datetime
import json
import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import ClassVar, List

import backoff
import geopandas
import h5py
import httpx
import pandas as pd
from httpx import HTTPError, Timeout
from pqdm.threads import pqdm
from pydantic import BaseModel

from .tqdm_callback import ProgressCallback, tqdm_callback
from .types import Product

DEFAULT_TIMEOUT = Timeout(timeout=30.0)


def is_valid_hdf5(filename: str | Path):
    try:
        with h5py.File(filename, "r") as f:
            return True
    except (IOError, OSError):
        return False


def chunks(ls, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(ls), n):
        yield ls[i : i + n]


def safe_apply_nest_asyncio():
    try:
        import uvloop

        if isinstance(asyncio.get_event_loop(), uvloop.Loop):
            # Skip applying nest_asyncio
            return False

        import nest_asyncio

        nest_asyncio.apply()
        return True
    except ImportError:
        # Apply nest_asyncio if uvloop is not installed
        import nest_asyncio

        nest_asyncio.apply()
        return True
    except RuntimeError:
        # Not in loop, simply return (???)
        return False


@backoff.on_exception(
    backoff.expo,
    HTTPError,
)
async def get_url(client: httpx.AsyncClient, url, params):
    """

    Returns
    -------
    httpx.Response
        HTTP response
    """
    return await client.get(url, params=params, timeout=DEFAULT_TIMEOUT)


@dataclass
class BlackMarbleDownloader(BaseModel):
    """A downloader to retrieve `NASA Black Marble <https://blackmarble.gsfc.nasa.gov>`_ data.

    Attributes
    ----------
    bearer: str
        NASA EarthData bearer token

    directory: Path
        Local directory to which download
    """

    bearer: str
    directory: Path

    TILES: ClassVar[geopandas.GeoDataFrame] = geopandas.read_file(
        files("blackmarble.data").joinpath("blackmarbletiles.geojson")
    )
    URL: ClassVar[str] = "https://ladsweb.modaps.eosdis.nasa.gov"

    def __init__(self, bearer: str, directory: Path):
        safe_apply_nest_asyncio()
        super().__init__(bearer=bearer, directory=directory)

    async def get_manifest(
        self,
        gdf: geopandas.GeoDataFrame,
        product_id: Product,
        date_range: datetime.date | List[datetime.date],
        on_progress: ProgressCallback | None = None,
    ) -> pd.DataFrame:
        """Retrieve NASA Black Marble data manifest. i.d., download links.

        Parameters
        ----------
        product_id: Product
            NASA Black Marble product suite (VNP46) identifier

        date_range: datetime.date | List[datetime.date]
            Date range for which to retrieve NASA Black Marble data manifest

        Returns
        -------
        pandas.DataFrame
            NASA Black Marble data manifest (i.e., downloads links)
        """
        if isinstance(date_range, datetime.date):
            date_range = [date_range]
        if isinstance(product_id, str):
            product_id = Product(product_id)

        # Create bounding box
        gdf = pd.concat([gdf, gdf.bounds], axis="columns").round(2)
        gdf["bbox"] = gdf.round(2).apply(
            lambda row: f"x{row.minx}y{row.miny},x{row.maxx}y{row.maxy}", axis=1
        )

        async with httpx.AsyncClient(verify=False, timeout=DEFAULT_TIMEOUT) as client:
            tasks = []
            for chunk in chunks(date_range, 250):
                for _, row in gdf.iterrows():
                    url = f"{self.URL}/api/v1/files"
                    params = {
                        "product": product_id.value,
                        "collection": "5000",
                        "dateRanges": f"{min(chunk)}..{max(chunk)}",
                        "areaOfInterest": row["bbox"],
                    }
                    tasks.append(asyncio.ensure_future(get_url(client, url, params)))

            responses = [
                await f
                for f in tqdm_callback(
                    asyncio.as_completed(tasks),
                    total=len(tasks),
                    desc="GETTING MANIFEST...",
                    step_name="fetch_manifests",
                    callback=on_progress,
                )
            ]

            rs = []
            for r in responses:
                try:
                    rs.append(pd.DataFrame(r.json()).T)
                except json.decoder.JSONDecodeError:
                    continue

            return pd.concat(rs)

    @backoff.on_exception(
        backoff.expo,
        HTTPError,
    )
    def _download_file(
        self,
        name: str,
        skip_if_exists: bool = True,
        on_progress: ProgressCallback | None = None,
    ):
        """Download NASA Black Marble file

        Parameters
        ----------
        names: str
             NASA Black Marble filename

        Returns
        -------
        filename: pathlib.Path
            Filename of downloaded data file
        """
        url = f"{self.URL}{name}"
        name = name.split("/")[-1]

        filename = Path(self.directory, name)
        file_valid = filename.exists() and is_valid_hdf5(filename)

        if not skip_if_exists or not file_valid:
            with open(filename, "wb+") as f:
                with httpx.stream(
                    "GET",
                    url,
                    headers={"Authorization": f"Bearer {self.bearer}"},
                    timeout=DEFAULT_TIMEOUT,
                ) as response:
                    if response.is_error:
                        raise HTTPError(str(response.content))
                    total = int(response.headers["Content-Length"])
                    with tqdm_callback(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        leave=None,
                        step_name="download",
                        callback=on_progress,
                    ) as pbar:
                        pbar.set_description(f"Downloading {name}...")
                        for chunk in response.iter_raw():
                            f.write(chunk)
                            pbar.update(len(chunk))
        return filename

    def download(
        self,
        gdf: geopandas.GeoDataFrame,
        product_id: Product,
        date_range: List[datetime.date],
        skip_if_exists: bool = True,
        on_progress: ProgressCallback | None = None,
    ):
        """
        Downloads files asynchronously from NASA Black Marble archive.

        Parameters
        ----------
        gdf: geopandas.GeoDataFrame
             Region of Interest. Converted to EPSG:4326 and intersected with Black Mable tiles

        product: Product
            Nasa Black Marble Product Id (e.g, VNP46A1)

        date_range: List[datetime.date]
            Date range for which to download NASA Black Marble data.

        skip_if_exists: bool, default=True
            Whether to skip downloading data if file already exists

        Returns
        -------
        list: List[pathlib.Path]
            List of downloaded H5 filenames.
        """
        # Convert to EPSG:4326 and intersect with self.TILES
        gdf = geopandas.overlay(
            gdf.to_crs("EPSG:4326").dissolve(), self.TILES, how="intersection"
        )

        # Fetch manifest data asynchronously
        bm_files_df = asyncio.run(self.get_manifest(gdf, product_id, date_range))

        # Filter files to those intersecting with Black Marble tiles
        bm_files_df = bm_files_df[
            bm_files_df["name"].str.contains("|".join(gdf["TileID"]))
        ]

        # Prepare arguments for parallel download
        names = bm_files_df["fileURL"].tolist()
        args = [(name, skip_if_exists, on_progress) for name in names]
        return pqdm(
            args,
            self._download_file,
            n_jobs=os.cpu_count(),
            argument_type="args",
            desc="Downloading...",
            exception_behaviour="immediate",
        )
