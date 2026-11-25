import full_image_processing 
import simulation.find_path 
import simulation.vis
import simulation.path_planner


if __name__ == '__main__':
    ball_pixeles = full_image_processing.main()
    input_positions, chosen, path = simulation.find_path.main(ball_pixeles)
    simulation.vis.main(input_positions, chosen, path)

