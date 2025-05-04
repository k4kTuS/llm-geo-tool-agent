from pydantic import BaseModel, field_validator
from typing import ClassVar

import geopandas as gpd
from shapely.geometry import Polygon, Point, box
from shapely.errors import ShapelyError
from shapely.wkt import loads
import pyproj


class GeoModel(BaseModel):
    wkt: str
    crs: str = "EPSG:4326"
    _geometry_type: ClassVar[type] = None

    @property
    def geom(self):
        """Convert WKT to Shapely geometry"""
        return loads(self.wkt)

    @property
    def gdf(self):
        """Return a GeoDataFrame with this geometry"""
        return gpd.GeoDataFrame(geometry=[self.geom], crs=self.crs)

    @classmethod
    def from_geometry(cls, geometry, crs: str = "EPSG:4326"):
        """Create an instance from a Shapely geometry"""
        if not isinstance(geometry, cls._geometry_type):
            raise TypeError(f"Expected {cls._geometry_type.__name__}, got {type(geometry)}")
        return cls(wkt=geometry.wkt, crs=crs)
    
    @classmethod
    def _validate_wkt(cls, value, expected_type):
        """Validate WKT string and check geometry type"""
        try:
            geom = loads(value)
            if not isinstance(geom, expected_type):
                raise ValueError(f"WKT must represent a {expected_type.__name__}")
            return value
        except ShapelyError:
            raise ValueError("Invalid WKT string")
    
    def is_geograpgic(self):
        """Check if the geometry is in a geographic CRS"""
        return pyproj.CRS(self.crs).is_geographic

    def to_crs(self, target_crs: str):
        """Transform geometry to a different CRS"""
        gdf = gpd.GeoDataFrame({"geometry": [self.geom]}, crs=self.crs)
        gdf = gdf.to_crs(target_crs)
        return self.from_geometry(gdf.geometry.iloc[0], target_crs)

class BoundingBox(GeoModel):
    _geometry_type: ClassVar[type] = Polygon

    @field_validator("wkt", mode="before")
    @classmethod
    def validate_wkt(cls, value):
        """Ensure WKT is valid and represents a Polygon"""
        return cls._validate_wkt(value, Polygon)
    
    @classmethod
    def from_bounds(cls, minx: float, miny: float, maxx: float, maxy: float, crs: str = "EPSG:4326"):
        """Create a BoundingBox from bounds coordinates
        
        Args:
            minx: Minimum x coordinate
            miny: Minimum y coordinate
            maxx: Maximum x coordinate
            maxy: Maximum y coordinate
            crs: Coordinate Reference System (default: WGS84)
        """
        polygon = box(minx, miny, maxx, maxy)
        return cls(wkt=polygon.wkt, crs=crs)    

    @property
    def center(self) -> Point:
        """Returns the centroid of the bounding box"""
        return self.geom.centroid

    @property
    def area(self) -> float:
        """Returns area in square kilometers
        
        If the geometry is in a geographic CRS, it will be projected to UTM for area calculation.
        If the geometry is in a projected CRS, the area will be calculated directly.
        """
        if self.is_geograpgic():
            gdf = self.gdf
            utm_crs = gdf.estimate_utm_crs()
            gdf = gdf.to_crs(utm_crs)
            return gdf.iloc[0].geometry.area / 1_000_000
        else:
            return self.geom.area / 1_000_000

    def as_envelope(self):
        return self.geom.envelope

    def bounds(self):
        """Returns bounds in (minx, miny, maxx, maxy) order"""
        return self.geom.bounds
    
    def bounds_yx(self):
        """Returns bounds in (miny, minx, maxy, maxx) order"""
        minx, miny, maxx, maxy = self.geom.bounds
        return (miny, minx, maxy, maxx)

    def to_string(self) -> str:
        """Returns the bounding box as a string in the format minx,miny,maxx,maxy"""
        return ','.join(map(str, self.bounds()))

    def to_string_yx(self) -> str:
        """Returns the bounding box as a string in the format miny,minx,maxy,maxx"""
        return ','.join(map(str, self.bounds_yx()))

    def to_string_wms(self, wms_version: str = "1.3.0") -> str:
        """Returns the bounding box as a string with the correct order for the CRS and WMS version"""
        if wms_version == "1.3.0":
            if pyproj.CRS(self.crs).axis_info[0].direction == "east":
                return self.to_string()
        return self.to_string_yx()
    
class PointMarker(GeoModel):
    _geometry_type: ClassVar[type] = Point

    @field_validator("wkt", mode="before")
    @classmethod
    def validate_wkt(cls, value):
        """Ensure WKT is valid and represents a Point"""
        return cls._validate_wkt(value, Point)

    @classmethod
    def from_coordinates(cls, x: float, y: float, crs: str = "EPSG:4326"):
        """Create a PointMarker from x and y coordinates
        
        Args:
            x: The x coordinate
            y: The y coordinate
            crs: Coordinate Reference System (default: WGS84)
        """
        point = Point(x, y)
        return cls(wkt=point.wkt, crs=crs)

    @property
    def coordinates(self):
        """Returns the coordinates as (x, y)"""
        return self.geom.x, self.geom.y

    @property
    def x(self) -> float:
        """Returns the x coordinate (longitude in WGS84)"""
        return self.geom.x
    
    @property
    def y(self) -> float:
        """Returns the y coordinate (latitude in WGS84)"""
        return self.geom.y

    def to_string(self) -> str:
        """Returns the point as a string in the format x,y"""
        return f"{self.geom.x},{self.geom.y}"
    
    def to_string_yx(self) -> str:
        """Returns the point as a string in the format y,x"""
        return f"{self.geom.y},{self.geom.x}"
