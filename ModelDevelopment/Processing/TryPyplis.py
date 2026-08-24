import pyplis
from geonum import GeoPoint

#https://pyplis.readthedocs.io/en/latest/examples.html#advanced-examples-for-emission-rate-analysis

IMG_DIR = "C:/Users/ggp24ash/Documents/Main Datasets/Pyplis/pyplis_etna_testdata/images/"

# Define camera (here the default ecII type is used)
cam_id = "ecII" #TODO does PiCam exist as one of the std camera classes

# the camera filter setup
filters = [pyplis.utils.Filter(type="on", acronym="F01"),
            pyplis.utils.Filter(type="off", acronym="F02")]

# camera location and viewing direction (altitude will be retrieved
# automatically)
geom_cam = {"lon": 15.1129, #TODO Edit all this, this is for Etna! Ask for acess to sheet
            "lat": 37.73122,
            "elev": 20.0,
            "elev_err": 5.0,
            "azim": 270.0,
            "azim_err": 10.0,
            "alt_offset": 15.0,
            "focal_length": 25e-3}  # altitude offset (above topography)

cam = pyplis.setupclasses.Camera(cam_id, filter_list=filters, **geom_cam)

source = pyplis.setupclasses.Source("etna")

# Create BaseSetup object (which creates the MeasGeometry object)
stp = pyplis.setupclasses.MeasSetup(
    base_dir=IMG_DIR,
    camera=cam,
    source=source
    )

ds = pyplis.Dataset(stp)

# get on-band image list
on_list = ds.get_list("on")
on_list.goto_next()
off_list = ds.get_list("off")

# activate dark correction in both lists. Dark and offset image lists are
# automatically assigned to plume on and off-band image lists on initiation
# of the dataset object
on_list.darkcorr_mode = False
off_list.darkcorr_mode = False

print("On-band list contains %d images, current image index: %d"% (on_list.nof, on_list.cfn))

img = on_list.current_img()

# plume distance image retrieved from MeasGeometry class...
plume_dists = on_list.plume_dists

print("Testing")

###########Measurment Geometry Example Script

# Position of SE crater in the image (x, y)
se_crater_img_pos = [806, 736]

# Geographic position of SE crater (extracted from Google Earth)
# The GeoPoint object (geonum library) automatically retrieves the altitude
# using SRTM data
se_crater = GeoPoint(37.747757, 15.002643, name="SE crater", auto_topo_access=True)

print("Retrieved altitude SE crater (SRTM): %s" % se_crater.altitude)

# The following method finds the camera viewing direction based on the
# position of the south east crater.
new_elev, new_azim, _, basemap =\
    meas_geometry.find_viewing_direction(pix_x=se_crater_img_pos[0],
                                        pix_y=se_crater_img_pos[1],
                                        # for uncertainty estimate
                                        pix_pos_err=100,
                                        geo_point=se_crater,
                                        draw_result=True,
                                        update=True)  # overwrite settings

print("Updated camera azimuth and elevation in MeasGeometry, new values: "
        f"elev = {new_elev:.1f}, azim = {new_azim:.1f}")