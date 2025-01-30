#!/usr/bin/env python
#
## This script converts the .xml(ParaView compatible format) colormaps into Matplotlib or MATLAB format
import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from lxml import etree

## load source xml file
def _subfn_load_xml_colormap(xml):
    try:
        xmldoc = etree.parse(xml)
    except IOError as e:
        print('The input file is invalid. It must be a colormap xml file. Go to https://sciviscolor.org/home/colormaps/ for some good options')
        print('Go to https://sciviscolor.org/matlab-matplotlib-pv44/ for an example use of this script.')
        sys.exit()
    data_vals=[]
    color_vals=[]
    for s in xmldoc.getroot().findall('.//Point'):
        data_vals.append(float(s.attrib['x']))
        color_vals.append((float(s.attrib['r']),float(s.attrib['g']),float(s.attrib['b'])))
    return {'color_vals':color_vals, 'data_vals':data_vals}

## source of this function: http://schubert.atmos.colostate.edu/~cslocum/custom_cmap.html#code
def load_cmap_from_xml_file(xml) -> mpl.colors.LinearSegmentedColormap:
    """ 
    from neuropy.utils.external.cm_xml_to_matplotlib import load_cmap_from_xml_file
    
    mycmap = load_cmap_from_xml_file(args.path)
    """
    vals = _subfn_load_xml_colormap(xml)
    colors = vals['color_vals']
    position = vals['data_vals']
    assert len(position) == len(colors), f"position length must be the same as colors but len(position): {len(position)} and len(colors): {len(colors)}"
    cdict = {'red':[], 'green':[], 'blue':[]}
    # the first position must be 0.0
    if position[0] != 0:
        cdict['red'].append((0, colors[0][0], colors[0][0]))
        cdict['green'].append((0, colors[0][1], colors[0][1]))
        cdict['blue'].append((0, colors[0][2], colors[0][2]))
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))
    # the last position must be 1.0
    if position[-1] != 1:
        cdict['red'].append((1, colors[-1][0], colors[-1][0]))
        cdict['green'].append((1, colors[-1][1], colors[-1][1]))
        cdict['blue'].append((1, colors[-1][2], colors[-1][2]))
    # end for pos, color ...
    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap


# def make_cmap(xml):
#     vals = load_xml(xml)
#     colors = vals['color_vals']
#     position = vals['data_vals']
#     if len(position) != len(colors):
#         sys.exit('position length must be the same as colors')
#     elif position[0] != 0 or position[-1] != 1:
#         sys.exit('position must start with 0 and end with 1')
#     cdict = {'red':[], 'green':[], 'blue':[]}
#     if position[0] != 0:
#         cdict['red'].append((0, colors[0][0], colors[0][0]))
#         cdict['green'].append((0, colors[0][1], colors[0][1]))
#         cdict['blue'].append((0, colors[0][2], colors[0][2]))
#     for pos, color in zip(position, colors):
#         cdict['red'].append((pos, color[0], color[0]))
#         cdict['green'].append((pos, color[1], color[1]))
#         cdict['blue'].append((pos, color[2], color[2]))
#         if position[-1] != 1:
#             cdict['red'].append((1, colors[-1][0], colors[-1][0]))
#             cdict['green'].append((1, colors[-1][1], colors[-1][1]))
#             cdict['blue'].append((1, colors[-1][2], colors[-1][2]))
#     cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
#     return cmap


## This is a quick example plotting the 8 by 1 gradient of the colormap 
## with Matplotlib
def plot_cmap(colormap):
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    fig=plt.figure(figsize=(8,1))
    map=fig.add_subplot(111)
    map.set_frame_on(False)
    map.get_xaxis().set_visible(False)
    map.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0)
    map.imshow(gradient, aspect='auto', cmap=plt.get_cmap(colormap))
    plt.show(fig)



# ## check for correct file type
# def is_xml(string):
# 	if os.path.isfile(string):
# 		return string
# 	else:
# 		raise argparse.ArgumentTypeError('The file %s does not exist!' % string)


# ## Example usage
# if __name__ == '__main__':
# 	import argparse
    
# 	parser = argparse.ArgumentParser(description='Convert ParaView compatible colormaps to Matplotlib or MATLAB compatible colormaps.')
# 	parser.add_argument('-f', '--file-path', dest='path', required=True, type=lambda s: is_xml(s), help='Input file path of .xml colormap with position starting at 0 and ending at 1.')
# 	parser.add_argument('-m', '--make-matrix', dest='matrix', action='store_true', required=False, help='Print a 3xn matrix of rgb values to copy and paste into MATLAB.')
    
# 	args = parser.parse_args()

# 	## construct the colormap
# 	mycmap = load_cmap_from_xml_file(args.path)
# 	print 'converted successfully!'

# 	## mycmap is matplotlib compatible object. to query color value out of it:
# 	print 'example rgba value for data value 0 is: ' + str(mycmap(0.0))

# 	## MATLAB Users: This fuction will output a RGB matrix to use in MATLAB
# 	if args.matrix == True:
# 		print_cmap_matrix(mycmap)

# 	## Plotting the colormap to test the conversion
# 	plot_cmap(mycmap)


