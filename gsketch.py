from argparse import ArgumentParser                                                                                     # command-line argument parsing
from matplotlib.colors import LinearSegmentedColormap                                                                   # custom colourmaps for matplotlib
from pathlib import Path                                                                                                # path manipulation tools
from PIL import Image                                                                                                   # image manipulation tools
from tqdm import tqdm                                                                                                   # progress bar management

import matplotlib.pyplot as plt
import numpy as np

###############################################################################

# known 3D-printer profiles, add your own settings as reported by your slicer
profiles = {"prusa-mk3s+": {"max_acc":  {"x": 1000.0, "y": 1000.0, "z": 200.0, "p": 1250.0, "t": 1250.0},               # maximum acceleration
                            "max_feed": {"x":  200.0, "y":  200.0, "z":  12.0},                                         # maximum feed
                            "max_jerk": {"x":    8.0, "y":    8.0, "z":   0.4},                                         # maximum jerk
                            "max_size": {"x":  250.0, "y":  210.0, "z": 210.0},                                         # print volume limits
                            "feed"    : {"xy": 200.0 * 60,         "z": 12.0 * 60}}}                                    # move feed to use (in minutes)
    
###############################################################################

def generate_preamble(profile):
    """
    Generates the G-code preamble: printer homing, applying settings and paper loading
    ----------
    args:
        profile     dict {str: {str: float}}    settings for selected printer profile
    ----------
    returns:
        preamble    list [str]                  list of G-code strings, one string per line
    """
    return ["M73 P0",                                                                                                   # sets build percentage [%]
            "M201 X{x} Y{y} Z{z}".format(**profile["max_acc"]),                                                         # sets maximum accelerations [mm/sec^2]
            "M203 X{x} Y{y} Z{z}".format(**profile["max_feed"]),                                                        # sets maximum feedrates [mm/sec]
            "M205 X{x} Y{y} Z{z}".format(**profile["max_jerk"]),                                                        # sets jerk limits [mm/sec]
            "M204 P{p} T{t}".format(**profile["max_acc"]),                                                              # sets maximum print and travel acceleration [mm/sec^2]
            "G90",                                                                                                      # use absolute coordinates
            "G21",                                                                                                      # all units in mm
            "G28 W",                                                                                                    # home all without mesh bed level
            "G0 Z100",                                                                                                  # raise z for paper load
            "M117 Place paper on bed",                                                                                  # display message
            "M300 S1200 P400",                                                                                          # beep start
            "M300 S0 P400",                                                                                             # beep stop
            "M0"]                                                                                                       # pause print
            
def generate_postamble(profile):
    """
    Generates the G-code postable: move pen away from bed, power off motors and signal 100% completion
    ----------
    args:
        profile     dict {str: {str: float}}    settings for selected printer profile
    ----------
    returns:
        postamble   list [str]                  list of G-code strings, one string per line
    """
    return ["G0 Y{y} Z100".format(**profile["max_size"]),                                                               # move pen up and bed towards user
            "M84",                                                                                                      # power off motors
            "M73 P100"]                                                                                                 # set build percentage [%]
            
###############################################################################
            
def process_image(bits, colour, home, image, nudge, offset, resolution, size, zsafe, zdraw, feed):
    """
    Generates the main bulk of the G-code by processing image into a list of "dot" movements
    ----------
    args:
        bits        int                         number of bits per channel that will represent up top 2**{bits} shades per channel
        colour      str                         colour mode that determines which channels get printed, one of ['grayscale', 'cmy', 'cmyk']
        home        dict {str: float}           absolute coordinates in mm of the home position for the pen
        image       str                         string representing the absolute or relative path of the image to process
        nudge       float                       move the pen home of each channel away from the first channel's home by {nudge} mm
        offset      dict {str: float}           offset of image from pen home in mm
        resolution  float                       defines the minimum feature size of the printed image
        size        list [float]                temporary x and y size of the final image in mm, if one of them is zero it will be inferred from the aspect ratio
        zsafe       float                       safe movement height above pen home for non-draw pen moves in mm
        zdraw       float                       pen draw height below home in mm
        feed        dict {str: float}           movement feed values for xy and z moves
    ----------
    returns:
        content     list [str]                  list of G-code strings, one string per line
    """
    print(f"{'='*20} SETTINGS {'='*20}")                                                                                # print summary of settings: header
    print(f"BITS:          {bits} (up to {2**bits - 1:.0f} dots per channel per chunk)")                                # ... number of bits for representing colour, 2**bits gives total possible shades per channel
    print(f"COLOUR MODE:   {colour}")                                                                                   # ... colour mode, one of ['grayscale', 'cmy', 'cmyk']
    print(f"PEN HOME:      {home}")                                                                                     # ... coordinates of pen home in {'x': x, 'y': y, 'z': z} format
    print(f"COLOUR NUDGE:  {nudge} mm")                                                                                 # ... nudge of inidividual colour homes in mm
    print(f"IMAGE OFFSET:  {offset} from home")                                                                         # ... offset of image from pen home in {'x': x, 'y': y} format
    print(f"RESOLUTION:    {resolution} ({resolution}mm x {resolution}mm chunk size)")                                  # ... resolution - defines side size of square chunks
    print(f"IMAGE SIZE:    {size}")                                                                                     # ... image size in [x, y] format (one of those may be zero)
    print(f"Z SAFE HEIGHT: {zsafe}mm from home")                                                                        # ... safe z height above home for pen movement
    print(f"Z DRAW HEIGHT: {zdraw}mm from home")                                                                        # ... z height below home for draw moves
    print(f"FEED RATE:     {feed}")                                                                                     # ... feed rate in {'xy': xy, 'z': z} format
    print(f"{'='*50}")                                                                                                  # ... summary footer
    
    with Image.open(image) as img:                                                                                      # open image using PIL
        print(f">> Loading image {image}")
        
        print( ">> Extracting temporary image array in grayscale mode")
        arr = np.asarray(img.convert('L'))                                                                              # load image in grayscale mode to get 2D array size
        size = {"x": size[0] if size[0] > 0 else (size[1] * (arr.shape[1] / arr.shape[0])),                             # update size: compute x size from image aspect ratio and y requirement if x=0
                "y": size[1] if size[1] > 0 else (size[0] * (arr.shape[0] / arr.shape[1]))}                             # update size: compute y size from image aspecr ratio and x requirement if y=0
        print(f"<< Loaded temporary array with shape {arr.shape}")
        print(f"<< Updated image size constraint to {size} mm")
        
        x_step = int(arr.shape[1] // (size['x'] / resolution))                                                          # compute the number of pixels in each x-step to achieve the target resolution
        y_step = int(arr.shape[0] // (size['y'] / resolution))                                                          # as above for y (and should be equal to x-step), separated out in case different x and y resolutions are wanted in the future
        
        print(f"<< Pixels per chunk set to {x_step}x{y_step} ({x_step * y_step:.0f} ppc)")
        print(f">> Resizing image to {x_step * (arr.shape[1] // x_step)}x{y_step * (arr.shape[0] // y_step)} px")
        print( ">> Downscaling image to pixel per chunk")
        
        img_downscaled = img.resize(size=(int(size['x'] / resolution), int(size['y'] / resolution)),                    # resize the image, new size is per chunk (defined by x- and y-step) not per pixel
                                    resample=Image.Resampling.LANCZOS,                                                  # resample image using the Lanczos method (higest quality)
                                    box=(0, 0, x_step * (arr.shape[1] // x_step), y_step * (arr.shape[0] // y_step)))   # first, crop image to largest multiple of x- and y-step, removing a few pixels from the edges
                                    
    print( ">> Extracting final image array in CMYK mode")
    arr_downscaled = np.asarray(img_downscaled.convert('CMYK'))                                                         # extract array from image in CMYK mode, array will be 3D and of shape (y, x, c) where c is one of the CMYK channels
    arr_downscaled[:, :, 3] = 255 - np.asarray(img_downscaled.convert('L'))                                             # as CMYK mode leaves K empty, fill in K channel from grayscale image mode
    print( "<< Per-channel limits identified as: C={} M={} Y={} K={}".format(*[(arr_downscaled[:, :, chan].min(), arr_downscaled[:, :, chan].max()) for chan in range(arr_downscaled.shape[2])]))
    print(f"<< Updated image array size to {arr_downscaled.shape}")
    
    print(f">> Updating upper array limit (255 -> {bits**2 - 1})")
    arr_nbit = np.floor(arr_downscaled / (256 / 2**bits)).astype(int)                                                   # lower the per-chanel bit-depth to that specifed by {bits}
    print( "<< Updated image per-channel limits to C={} M={} Y={} K={}".format(*[(arr_nbit[:, :, chan].min(), arr_nbit[:, :, chan].max()) for chan in range(arr_nbit.shape[2])]))

    print( ">> Plotting individual CMYK channels")
    fig, axes = plt.subplots(ncols=4, figsize=(13, 3.5))                                                                # create figure with 4 axis that will contain one CMYK channel each
    for chan_id, (rgb, key, lbl, stat) in enumerate(zip([[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0]],                   # iterate over ... CMYK colours in RGB representated as a list of floats (chan_id represents the index of iteration)
                                                        ["c",       "m",       "y",       "k"],                         # ... dictionary keys for each channel
                                                        ["Cyan",    "Magenta", "Yellow",  "Key"],                       # ... colours names as labels, and below, ... status of whether each channel is enabled or disabled with respect to colour requirements
                                                        ["ENABLED" if "cmy" in colour else "DISABLED"]*3 + ["ENABLED" if colour in ["grayscale", "cmyk"] else "DISABLED"])):
        colours = [rgb + [0], rgb + [1]]                                                                                # convert RGB to two RGBA points of maximum and minimum transparency
        cmap = LinearSegmentedColormap.from_list(lbl, colours)                                                          # construct matplotlib colourmap with the two RGBA points at each end
        axes[chan_id].imshow(arr_downscaled[:, :, chan_id], cmap=cmap, clim=(0, 255))                                   # plot image of a single colour channel, using a colourmap of colours representing that channel
        axes[chan_id].set_title(f"{lbl} ({stat})")                                                                      # set title of the axis as colour name and enabled/disabled status
        axes[chan_id].set_xticks([])                                                                                    # remove all x-ticks, they are simply chunk numbers which are not informative to the user
        axes[chan_id].set_yticks([])                                                                                    # also remove all y-ticks
    plt.tight_layout()                                                                                                  # minimize border space in the figure
    plt.show()                                                                                                          # show figure to user

    x_pts = {"c": [], "m": [], "y": [], "k": []}                                                                        # create a dict that will contain the x-coordinates in real space of each point for each channel
    y_pts = {"c": [], "m": [], "y": [], "k": []}                                                                        # create a matching dict of y-points, these are separated for ease of plotting
    reverse = True                                                                                                      # as the arrays are read top-down and left-to-right, every second left-to-right list needs to be reversed to minimize pen move distance
    
    with tqdm(total=arr_nbit.shape[0]*arr_nbit.shape[1], desc=">> Generating dots for per chunk", ascii=True) as progress:  # using a progress bar that tracks the percentage of the array that has been traversed
        for y in range(0, arr_nbit.shape[0]):                                                                           # iterate top-down over the y-axis
            reverse = not reverse                                                                                       # every second left-to-right list will reversed to minimise pen move distance
            x_pts_tmp = {"c": [], "m": [], "y": [], "k": []}                                                            # create a temporary dict for per-channel x-points, this will later be reversed if necessary and joined with the main x_pts dict
            y_pts_tmp = {"c": [], "m": [], "y": [], "k": []}                                                            # as above but for y-coordinates of each dot
            for x in range(0, arr_nbit.shape[1]):                                                                       # now iterate left-to-right over each horizontal chunk
                for chan_id, chan in enumerate(["c", "m", "y", "k"]):                                                   # iterate over strings that represent dict keys for each channel
                    for dim, target in zip([x, y], [x_pts_tmp[chan], y_pts_tmp[chan]]):                                 # iterate over the x and y dimension, each iteration contains the current index of that dimension as well as the target list for that dimension
                        target += list(np.random.uniform(dim * resolution, (dim + 1) * resolution, arr_nbit[y, x, chan_id]))    # generate as many points in real space of the corresponding dimension (x, then y) between the current and next index as the number stored at the current array coordinate
            for chan in ["c", "m", "y", "k"]:                                                                           # iterate again over each CMYK dict key
                for pts, pts_tmp in zip([x_pts, y_pts], [x_pts_tmp, y_pts_tmp]):                                        # iterate also over each dimension, where each iteration contains the final and temporary array of the corresponding dimension
                    if reverse: pts_tmp[chan].reverse()                                                                 # if necessary, reverse the list of x or y points
                    pts[chan] += pts_tmp[chan]                                                                          # merge list in temporary array to the corresponding list of the final array
            progress.update(arr_nbit.shape[1])                                                                          # update progress bar after each row has been processed
    print( "<< Generated point count per channel: C={} M={} Y={} K={}".format(*[len(x_pts[chan]) for chan in ["c", "m", "y", "k"]]))
    
    print( ">> Plotting per-channel reconstruction")
    fig, axes = plt.subplots(ncols=4, nrows=3, figsize=(13, 10))                                                        # create figure with 4x3=12 axis, 4 for each colour and 3 for each representation of that channel
    # set row and column labels, colums labelled by colour name and whether that colour channel is enabled or not, and rows labelled by three representations of the image
    axes[0, 0].xaxis.set_label_position('top'); axes[0, 0].set_xlabel(f"Cyan ({'ENABLED' if 'cmy' in colour else 'DISABLED'})")
    axes[0, 1].xaxis.set_label_position('top'); axes[0, 1].set_xlabel(f"Magenta ({'ENABLED' if 'cmy' in colour else 'DISABLED'})")
    axes[0, 2].xaxis.set_label_position('top'); axes[0, 2].set_xlabel(f"Yellow ({'ENABLED' if 'cmy' in colour else 'DISABLED'})")
    axes[0, 3].xaxis.set_label_position('top'); axes[0, 3].set_xlabel(f"Key ({'ENABLED' if colour in ['grayscale', 'cmyk'] else 'DISABLED'})")
    axes[0, 0].set_ylabel("Downscaled")                                                                                 # downscaled represents the image after it has been been scaled down to per-chunk instead of per-pixel
    axes[1, 0].set_ylabel("Adjusted depth")                                                                             # adjusted depth represents the image after its bit-depth is reduced to {bits}
    axes[2, 0].set_ylabel("Reconstruction")                                                                             # finally, the reconstruction represents the dots drawn per chunk per channel in order to reconstruct the image
    for chan_id, chan in enumerate(["c", "m", "y", "k"]):                                                               # iterate again over each channel key string
        axes[0, chan_id].imshow(arr_downscaled[:, :, chan_id], cmap="gist_yarg", clim=(0, 255))                         # the top row of axis will show each channel of the downscaled image array
        axes[1, chan_id].imshow(arr_nbit[:, :, chan_id], cmap="gist_yarg", clim=(0, 2**bits - 1))                       # the middle row will instead display the bit-depth-reduced arrays
        axes[2, chan_id].plot(x_pts[chan], y_pts[chan], marker=".", c="k", ms=1, alpha=0.25, ls=None, lw=0)             # for the final row, each computed point is drawn as a small circle with transparency
        axes[2, chan_id].set_xlim(0, size['x'])                                                                         # x-limits for the final row are adjusted to match those produced by imshow
        axes[2, chan_id].set_ylim(size['y'], 0)                                                                         # ... same for the y-limits
        axes[2, chan_id].set_aspect("equal")                                                                            # enforce equal aspect ratio so the dot-reconstructed image ins't stretched
    for row in axes:                                                                                                    # iterate over all axis (by first iterating over the outer array of the 2D arrays of axes)
        for ax in row:                                                                                                  # ... and then by the inner arrays that contain individual axis
            ax.set_xticks([])                                                                                           # disable the ticks as they only show chunk numbers, which are not useful to the user
            ax.set_yticks([])                                                                                           # also disable ticks on the y-axis
    plt.tight_layout()                                                                                                  # recompute borders for a tigther layout
    plt.show()                                                                                                          # show image to user
    
    print( ">> Plotting full image reconstruction")
    fig, ax = plt.subplots(figsize=(8, 8))                                                                              # create yet another figure, this time contianing only a single axis which will in turn contain points for all colours for a final reconstruction of the image
    for chan, rgb in zip(["y",       "m",       "c",       "k"],                                                        # iterate over channel keys ...
                         ["#FFFF00", "#FF00FF", "#00FFFF", "#000000"]):                                                 # ... as well as their corresponding RGB strings
        ax.plot(x_pts[chan], y_pts[chan], marker=".", c=rgb, ms=1, alpha=0.5, ls=None, lw=0)                            # for each channel, draw points of that colour
    ax.set_title("Full image reconstruction")                                                                           # set figure title
    ax.set_xlim(0, size['x'])                                                                                           # set x-axis limits so that the resulting image is the right way up
    ax.set_ylim(size['y'], 0)                                                                                           # ... same for the y-axis
    ax.set_aspect("equal")                                                                                              # enforce equal aspect ratio so image isn't shown stretched
    ax.set_xticks([])                                                                                                   # disable x-ticks as mentioned in the above plotting routine
    ax.set_yticks([])                                                                                                   # ... also disable the y-ticks
    plt.tight_layout()                                                                                                  # recompute borders for a tigther layout
    plt.show()                                                                                                          # show image to user
    
    gcode = []                                                                                                          # empty list will contain a string for each line of G-code
    channels = (["y", "m", "c"] if 'cmy' in colour else []) + (["k"] if colour in ['grayscale', 'cmyk'] else [])        # compute a list of all active channels as per the colour requirements, and this time iterate over only these keys
    total_points = sum(len(x_pts[chan]) for chan in channels)                                                           # total points to be converted to gcode is simply the sum of all x (or y) points for each channel
    completed_points = 0                                                                                                # here we will keep track of the total number of points processed so far
    last_percentage = 0                                                                                                 # this is the last percentage update sent to the printer, needed so that redundant percentage updates are not sent
    with tqdm(total=total_points, desc=">> Generating G-code", ascii=True) as progress:                                 # create a progress bar for this point to G-code conversion, with progress tracked by the number of points processed
        for chan, nudge_x, nudge_y in zip(channels,                                                                     # iterate over each active channel, and for each channel also add successive nudges so that pen home colours aren't contaminated
                                          [0,   nudge,     0, nudge],                                                   # nudges in the x-direction
                                          [0,   0,     nudge, nudge]):                                                  # ... as well as the y-direction, together they form a grid of 4 colours separated by {nudge} mm in the horizontal and vertical direction
            if len(x_pts[chan]) == 0: continue                                                                          # if the channel to be processed is empty, ignore it and skip that channel altogether (inclding colour change movements)
            gcode += ["G0 X{} Y{}".format(home['x'] + nudge_x, home['y'] + nudge_y),                                    # preamble: move to pen home xy + calibration nudge
                      "G0 Z{}".format(home['z']),                                                                       # ... move pen to home z for calibration
                      "M117 Insert {} pen".format(chan.upper()),                                                        # ... display message
                      "M300 S1200 P400",                                                                                # ... beep start
                      "M300 S0 P400",                                                                                   # ... beep stop
                      "M0",                                                                                             # ... pause print
                      "G0 Z{}".format(home["z"] + zsafe)]                                                               # ... move pen to safety zone
            for x, y in zip(x_pts[chan], y_pts[chan]):                                                                  # finally, after the correct colour has been loaded, iterate over all x and y points for this colour
                percentage = np.floor((completed_points / total_points) * 100)                                          # compute the current percentage
                if percentage != last_percentage:                                                                       # if the percentage is not equal to the previously printed percentage
                    gcode.append(f"M73 P{percentage}")                                                                  # ... add G-code to update printer with current percentage
                    last_percentage = percentage                                                                        # ... and locally update this percentage so it is not sent to the printer repetitively
                gcode.append(f"G0 X{home['x'] + offset['x'] + x:.2f} Y{home['y'] + offset['y'] + y:.2f} F{feed['xy']}") # draw section: move pen to the (x, y) point
                gcode.append(f"G0 Z{home['z'] - zdraw} F{feed['z']}")                                                   # ... drop pen to draw height
                gcode.append(f"G0 Z{home['z'] + zsafe} F{feed['z']}")                                                   # ... raise pen to safe height
                completed_points += 1                                                                                   # increment the number of processed points
                progress.update()                                                                                       # increment the progress bar
            gcode += ["G0 X{} Y{}".format(home['x'] + nudge_x, home['y'] + nudge_y),                                    # postamble: move to pen home xy + calibration nudge
                      "M117 Remove {} pen".format(chan.upper()),                                                        # ... display message
                      "M300 S1200 P400",                                                                                # ... beep start
                      "M300 S0 P400",                                                                                   # ... beep stop
                      "M0"]                                                                                             # ... pause print
    print(f"<< Generated {len(gcode)} G-code lines")

    return gcode                                                                                                        # return the generated G-code list
    
###############################################################################

def setup_argparse():
    """
    Sets up argument parser
    ----------
    args:
        None
    ----------
    returns:
        parser      argparse.ArgumentParser     argument parser object
    """
    parser = ArgumentParser(description="Script that converts images into gcode sketches")
    parser.add_argument('-0', '--home',       action="store", type=float, nargs=3, required=False, default=[40.0, 48.0, 2.0], help="Pen home coordinates [x, y, z]")
    parser.add_argument('-b', '--bits',       action="store", type=int,            required=False, default=3,                 help="Printed image bit-depth (draw up to 2**{bits} dots per chunk)")
    parser.add_argument('-c', '--colour',     action="store", type=str,            required=False, default="grayscale",       help="Colour mode in which image will be processed (one of ['grayscale', 'cmy', 'cmyk'])")
    parser.add_argument('-f', '--feed',       action="store", type=float,          required=False, default=1.0,               help="Move feed as a multiplier of max feed (0.5 -> 50% of max feed)")
    parser.add_argument('-F', '--feed_mult',  action="store", type=int,            required=False, default=60,                help="Number of seconds in unit of feed (60 -> feed specified per minute)")
    parser.add_argument('-i', '--image',      action="store", type=str,            required=True,                             help="Path to image file")
    parser.add_argument('-n', '--nudge',      action="store", type=float,          required=False, default=5,                 help="Offset of individual home colour markers in mm")
    parser.add_argument('-o', '--offset',     action="store", type=float, nargs=2, required=False, default=[20.0, 0.0],       help="Image offset from home in mm")
    parser.add_argument('-p', '--profile',    action="store", type=str,            required=False, default="prusa-mk3s+",     help="Printer profile to load")
    parser.add_argument('-r', '--resolution', action="store", type=float,          required=False, default=1.0,               help="Discretise image into square chunks with sides of {resolution} mm")
    parser.add_argument('-s', '--size',       action="store", type=float, nargs=2, required=False, default=[150.0, 0.0],      help="Final size of image in mm (0 to infer missing size by preserving aspect ratio)")
    parser.add_argument('-z', '--zsafe',      action="store", type=float,          required=False, default=1.0,               help="Safe travel z-distance above home")
    parser.add_argument('-Z', '--zdraw',      action="store", type=float,          required=False, default=0.2,               help="Draw depth below z-home")
    return parser

def parse_args(parser):
    """
    Parses arguments and verifies them for validity
    ----------
    args:
        parser      argparse.ArgumentParser     argument parser object
    ----------
    returns:
        args        argparse.Namespace          namespace object of parsed args
    """
    args = parser.parse_args()

    if not Path(args.image).is_file() or Path(args.image).suffix.lower() not in [".jpg", ".png"]:
        print("The specified file does not exist or is not an image")
        return
        
    if args.profile not in ["prusa-mk3s+", "custom"]:
        print("The specified profile does not exist")
        return
    
    args.home = {coord: val for coord, val in zip(["x", "y", "z"], args.home)}
    if not all([0 <= args.home[coord] < profiles[args.profile]["max_size"][coord] for coord in ["x", "y", "z"]]):
        print("Pen home outside of machine limites")
        return
        
    if not 0 < args.zsafe < profiles[args.profile]["max_size"]["z"] - args.zsafe:
        print("Invalid z-safe distance above home")
        return
        
    if not 0 <= args.zdraw <= 1:
        print("Draw z-offset must be negative (no less than -1mm) or zero")
        return
        
    if not 1 <= args.bits <= 8:
        print("Only bit-depths between 1 and 8 bits per chunk are supported")
        return
        
    if args.colour not in ['grayscale', 'cmy', 'cmyk']:
        print("Unsupported colour mode selected")
        return
        
    if not 0 < args.feed <= 2:
        print("Feed must be non-zero and no greater than 2.0 (200%)")
        return
    
    if not 0 < args.feed_mult <= (60*60):
        print("Feed multiplier must be non-zero and no greater than an hour (3600 seconds)")
        return
        
    if not 0 <= args.nudge <= 10:
        print("Nudge must be positive and no larger than 10mm")
        return
    
    args.offset = {coord: val for coord, val in zip(["x", "y"], args.offset)}
    if not all([0 <= args.home[coord] + args.offset[coord] < profiles[args.profile]["max_size"][coord] for coord in ["x", "y"]]):
        print("Image offset outside of machine limits")
        return
    
    if all([arg <= 0 for arg in args.size]):
        print("At least one image dimension needs to be specified")
        return
        
    if not 0 < args.resolution <= (max(args.size) / 2):
        print("Resolution must be non-zero and smaller than half the largest size of the image")
        return
    
    return args                                                                                                         # if arguments were parsed successfully, return them
    
###############################################################################
    
if __name__ == "__main__":
    args = parse_args(setup_argparse())                                                                                 # parse arguments
    if args is not None:                                                                                                # proceed only if parsing was successful
        profile = profiles[args.profile]                                                                                # load the current printer profile
        preamble = generate_preamble(profile)                                                                           # compute the G-code preamble
        postamble = generate_postamble(profile)                                                                         # compute the G-code postamble
        
        feed = {"xy": min(profile["max_feed"]["x"], profile["max_feed"]["y"]) * args.feed * args.feed_mult,             # compute the xy move feed according to the specified machine limits, requested feed and multiplier
                "z": profile["max_feed"]["z"] * args.feed * args.feed_mult}                                             # ... also compute the z move feed
        content = process_image(args.bits, args.colour, args.home, args.image, args.nudge, args.offset, args.resolution, args.size, args.zsafe, args.zdraw, feed)   # process image and generate main bulk of G-code
        
        with open("sketch.gcode", "w") as gcode:                                                                        # open output file for writing
            gcode.write("\n".join(preamble))                                                                            # write the preamble, separating each string by a newline
            gcode.write("\n"*2)                                                                                         # write empty line separating preamble and main bulk of code
            gcode.write("\n".join(content))                                                                             # write the main bulk of code, separating each string by a newline
            gcode.write("\n"*2)                                                                                         # write empty line separating main bulk of code and postamble
            gcode.write("\n".join(postamble))                                                                           # write the postamble, separating each string by a newline