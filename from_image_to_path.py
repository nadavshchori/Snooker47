import np_image_processing
import full_image_processing
import simulation.find_path 
import simulation.vis
import simulation.path_planner


if __name__ == '__main__':
    # ball_pixels = np_image_processing.main()
    ball_pixels = full_image_processing.main()
    input_positions, chosen, path = simulation.find_path.main(ball_pixels)
    simulation.vis.main(input_positions, chosen, path)

