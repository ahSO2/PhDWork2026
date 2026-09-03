import geonum

p1 = geonum.GeoPoint(latitude=0, longitude=0, altitude=0)
p2 = p1.offset(azimuth=45, dist_hor=5, dist_vert=3000)

v = p1 - p2
print(v.elevation)
