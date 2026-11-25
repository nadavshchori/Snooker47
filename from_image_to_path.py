#import np_image_processing
#import full_image_processing
import image_processing_corner_with_numpy
import simulation.find_path 
import simulation.vis
import simulation.path_planner
import time


if __name__ == '__main__':
    while True:
        ball_pixels = image_processing_corner_with_numpy.main()
        #ball_pixels = np_image_processing.main()
        #ball_pixels = full_image_processing.main()
        input_positions, chosen, path = simulation.find_path.main(ball_pixels)
        simulation.vis.main(input_positions, chosen, path)


