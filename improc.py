from imageutils import ImTools
import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Image processing tools')
    parser.add_argument('-i', '--image', 
        help = 'file path to source image to process',
        required = True)
    
    parser.add_argument('-k', '--kmeans',
        help = 'Quantize image using K-Means into a given number of clusters',
        type = int)
    
    parser.add_argument('-t', '--triangulate',
        help = 'Triangulate image with a given complexity',
        type = int)
    
    parser.add_argument('-s', '--show',
        help = 'Show image after processing',
        action = 'store_true')
    
    parser.add_argument('-o', '--output',
        help = 'Output transformed image to a file')
    
    parser.add_argument('-std', '--stdout',
        help = 'Output transformed image to standard out using a specified format (jpg, png, etc...)')
    
    parser.add_argument('-wi', '--width',
        help = 'Resize image width by removing low energy seams',
        type = int)
    
    parser.add_argument('-he', '--height',
        help = 'Resize image height by removing low energy seams',
        type = int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    if (args.kmeans is not None) and (args.triangulate is not None):
        print('Only one transformation can be performed!')
        sys.exit()


    imtool = ImTools(args.image)
    new_im = imtool.mImg

    if args.kmeans is not None:
        features = imtool.get_rgb_xy_features(1, .4)
        new_im = imtool.kmeans_cluster(args.kmeans, features)

    if args.triangulate is not None:
        new_im = imtool.triangulate(args.triangulate)

    if args.width is not None:
        new_im = imtool.seam_carve('x', args.width)
    
    if args.height is not None:
        imtool2 = ImTools(None)
        imtool2.mImg = new_im
        new_im = imtool2.seam_carve('y', args.height)

    if args.show is True:
        imtool.show(new_im)

    if args.output is not None:
        imtool.write_image(args.output, new_im)

    if args.stdout is not None:
        imtool.write_image_stdout(args.stdout, new_im)

