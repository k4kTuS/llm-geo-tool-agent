from pydantic import BaseModel, field_validator

import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely.errors import WKTReadingError
from shapely.wkt import loads


class BoundingBox(BaseModel):
    wkt: str

    @field_validator("wkt", mode="before")
    @classmethod
    def validate_wkt(cls, value):
        """Ensure WKT is valid and represents a Polygon"""
        try:
            if not isinstance(loads(value), Polygon):
                raise ValueError("WKT must represent a Polygon")
            return value  # Return valid WKT
        except WKTReadingError:
            raise ValueError("Invalid WKT string")

    @property
    def geom(self) -> Polygon:
        return loads(self.wkt)        

    @property
    def center(self) -> Point:
        return self.geom.centroid

    @property
    def area(self) -> float:
        """Calculate the area of the bounding box in km^2 using most appropriate UTM CRS"""
        gdf = gpd.GeoDataFrame({"geometry": [self.geom]}, crs="EPSG:4326")
        utm_crs = gdf.estimate_utm_crs()
        gdf = gdf.to_crs(utm_crs)
        return gdf.iloc[0].geometry.area / 1000000

    def bounds_lonlat(self):
        """Returns bounds in (minx, miny, maxx, maxy) (default Shapely order)"""
        return self.geom.bounds

    def bounds_latlon(self):
        """Returns bounds in (miny, minx, maxy, maxx) order"""
        minx, miny, maxx, maxy = self.geom.bounds
        return (miny, minx, maxy, maxx)

    def as_envelope(self):
        return self.geom.envelope

    def to_string_latlon(self) -> str:
        """Returns the bounding box as a string in the format miny,minx,maxy,maxx"""
        return ','.join(map(str, self.bounds_latlon()))

    def to_string_lonlat(self) -> str:
        """Returns the bounding box as a string in the format minx,miny,maxx,maxy"""
        return ','.join(map(str, self.bounds_lonlat()))


class PointMarker(BaseModel):
    wkt: str

    @field_validator("wkt", mode="before")
    @classmethod
    def validate_wkt(cls, value):
        """Ensure WKT is valid and represents a Point"""
        try:
            if not isinstance(loads(value), Point):
                raise ValueError("WKT must represent a Point")
            return value  # Return valid WKT
        except WKTReadingError:
            raise ValueError("Invalid WKT string")

    @property
    def geom(self) -> Point:
        return loads(self.wkt)

    def as_point(self):
        return self.geom

    def to_string_lonlat(self) -> str:
        """Returns the point as a string in the format x,y"""
        return ','.join(map(str, self.geom.coords[0]))
    
    def to_string_latlon(self) -> str:
        """Returns the point as a string in the format y,x"""
        return ','.join(map(reversed, map(str, self.geom.coords[0])))
