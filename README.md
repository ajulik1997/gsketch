# gsketch.py - Convert images to G-code sketches

This Python script converts image files into G-code that can be sent to a 3D-printer with a pen attachment. The image is reconstructed by drawing dots of varying density onto paper. Multiple colour modes, including greyscale and CMYK, are supported

~ INSERT EXAMPLES HERE ~

## Prerequisites

- Python 3.9+ (earlier version may work but have not been tested)
- Required packages (`pip install -r requirements.txt`)
- A 3D printer pen attachment of your choice
- Card paper (normal paper works, but 180+ GSM paper is recommended to prevent paper warping and ink bleed)
- A pen of your choice (greyscale mode)
- (optional, CMYK mode) Cyan, magenta, yellow and black pens (or closest available colours)

## Quick start

1. If not using a supported printer profile, identify the necessary limits of your 3D printer and add them to the `profiles` dictionary. These limits can usually be found on the settings page of your preferred slicer. _(Feel free to contribute the limits profile of your 3D printer by submitting a pull request!)_
2. Attach your pen mount and home your printer on all 3 axes.
3. Identify your pen home coordinates by attaching your pen to the pen holder a few millimetres above the surface and manually moving the print head until the pen hovers above the bottom left corner of the print bed.
4. Run `python gsketch.py --image my_image.jpg --home x y z` with an image of your choice and home coordinates obtained from the previous step.
5. Send the generated `sketch.gcode` file to be printed.
6. The printer will automatically home itself on all axes, move up the print head, and beep to signal it is ready for paper loading. Attach a piece of paper to the print bed (masking tape works well for me, but you may experiment with clips or magnetic mounts) and press the action button. _(Low GSM papers have the tendency to bleed ink or warp if too much ink is deposited, therefore 180+ GSM paper is recommended.)_
7. The printer will then move to pen home position and beep to signal it is ready for pen loading. Insert the pen to the pen holder such that it gently touches the paper and secure the pen in place. When ready, press the action button to unpause the printer, which will begin recreating the image.
8. If multiple colours are to be used, the printer will return to pen home after the current colour has finished printing and beep to signal it is ready for pen unloading. Remove the pen from the holder and press the action button. The printer will then move to a slightly offset pen home ready for loading of the next pen colour.
9. After completion, the printer will move the print head up and the bed towards the user. The printing process has finished.

## Options

The following table summarizes the available options, their default values, limits, and useful comments:

| &#160;&#160;&#160;&#160;Argument&#160;&#160;&#160;&#160; | Short | &#160;&#160;&#160;&#160;&#160;&#160;&#160;Default&#160;&#160;&#160;&#160;&#160;&#160;&#160; | Limits                                   | Comments |
|----------------------------------------------------------|-------|---------------------------------------------------------------------------------------------|------------------------------------------|----------|
| `--home`                                                 | `-0`  | 40.0, 48.0, 2.0                                                                             | Print volume                             | Defines the home coordinates (in mm) of the pen. This is a reasonable estimate for most left-mounted pen holders, but it is recommended to use your own pen home to maximize image print area. To calculate your pen home, home your printer on all axes, insert the pen into your holder, and manually move the print head until the tip of the pen hovers above the bottom left corner of the bed. The current head position can now be used as the pen home. |
| `--bits`                                                 | `-b`  | 3                                                                                           | 0<`b`≤8                                  | The per-channel colour bit-depth of the final image, where the maximum number of representable colour shades is given by `b`^2, and the maximum dot-per-chunk density is given by `b`^2 - 1. The default setting gives 8 possible shades and up to 7 dots per chunk, and is suited for "0.4" fine-tipped pens when using a chunk resolution of 1 mm. Change this value in tandem with `--resolution` to optimize the dot density for your use-case. |
| `--brightness`                                           | `-B`  | 1.0                                                                                         | 0≤`B`≤2                                  | Defines a value that will be used adjust the image brightness, with 1.0 returning the orignal image and 0.0 resulting in a fully black image. Internlally, this value is passed to PIL's `ImageEnhance` brightness enhancement routine. |
| `--colour`                                               | `-c`  | `"grayscale"`                                                                               | one of: `"grayscale"`, `"cmy"`, `"cmyk"` | The colour mode filters which image colour channels will be printed. `"grayscale"` prints exclusively the key/black channel, which can be printed using any pen colour. `"cmy"` prints the image in CMYK mode without the key channel. `"cmyk"` is an alternative to `"cmy"` which additionally prints the key channel, and is useful if your ink of choice is particularly transparent. |
| `--feed`                                                 | `-f`  | 1.0                                                                                         | 0<`f`≤2                                  | Fraction of the maximum feed value to use for pen movement. The default value corresponds to 100% of the maximum supported move feed. Some printers support feed rates of up to 200%. |
| `--feed_mult`                                            | `-F`  | 60                                                                                          | 0<`F`≤3600                               | Per-second feed multiplier for printers that don't quote their feed in seconds. The default value corresponds to a move rate per minute. |
| `--image`                                                | `-i`  | `None`                                                                                      | Supported formats: `.jpg`, `.png`        | Relative or absolute path to the image that will be processed. More image formats may be supported, but have not been tested. |
| `--k_bits`                                               | `-k`  | 1                                                                                           | 0<`k`≤8                                  | The colour bit-depth of the K channel when printing image in `cmyk` colour mode (if any other colour mode is used, the K channel is either ommitied or defaults to `--bits`). Useful if your inks are semi-transparent, where a K channel with `--bits` depth would make the image too dark but ommiting it entirely would degrade image details and contrast. See `--bits` and `--colour` for more info. |
| `--nudge`                                                | `-n`  | 5.0                                                                                         | 0≤`n`≤10                                 | Distance to move successive pen colour homes from the original pen home, in mm. Useful so that pen tips are not contaminated with the previous colour when loading and unloading pens. |
| `--offset`                                               | `-o`  | 20.0, 0.0                                                                                   | Print area                               | The distance by which the printed image will be offset from the pen home, in mm. The default is a reasonable choice alongside the default pen home and an image size of 150 mm, but may need to be adjusted to suit your use-case. |
| `--profile`                                              | `-p`  | `"prusa-mk3s+"`                                                                             | Available profiles                       | String representing one of the available printer profiles. If your profile is not listed, you will need to add your own custom profile to the `profiles` dictionary. Pull requests are welcome in order to build a comprehensive list of printer profiles. |
| `--resolution`                                           | `-r`  | 1.0                                                                                         | Image size                               | The image to be printed will be scaled down to "chunks", the amount of which is equivalent to the image size (in mm) divided by the image resolution (in mm). This defines the size of the smallest visible feature in the final image, and should be over double the width of your pen tip. This value should be adjusted in tandem with `--bits` in order to fine-tune the desired dot density. |
| `--size`                                                 | `-s`  | 150.0, 0.0                                                                                  | Print area                               | Defines the final size of the image in mm. One value may be left as 0.0, in which case it will be inferred from the image aspect ratio. |
| `--zsafe`                                                | `-z`  | 1.0                                                                                         | Print volume                             | Defines the height above pen home in mm at which it is safe to move the pen without drawing on the paper. If this value is set too low and your paper warps, the pen will never leave the paper and will draw lines instead of dots. If this value is set too high, the total printing time increases dramatically. |
| `--zdraw`                                                | `-Z`  | 0.2                                                                                         | 0≤`z`≤1                                  | Defines the height below pen home in mm at which the pen is guaranteed to leave a mark on the paper. High values are suitable for brush pens, while lower values are more suitable for ball-point pens. Be aware that too high value may break your pen or damage your bed if it breaks through the paper, while too low values may result in no mark being left on the paper, especially if your paper warps. |

## Known limitations

The following section summarizes the known limitations of the script:

- Only the Marlin G-code flavour is supported so far
- A limited number of printer profiles are available as-is, and users will likely need to add their own
