
import abc
import logging
import math
import time
import numpy as np

from ..carla.client import VehicleControl
from ..carla.client import make_carla_client
from ..carla.planner.planner import Planner
from ..carla.settings import CarlaSettings
from ..carla.tcp import TCPConnectionError

from . import results_printer
from .recording import Recording
from .metrics import Metrics

def get_vec_dist(x_dst, y_dst, x_src, y_src):
    vec = np.array([x_dst, y_dst] - np.array([x_src, y_src]))
    dist = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    return vec / dist, dist


def sldist(c1, c2):
    return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)


class BenchmarkRunner(object):
    """
    The Benchmark class, controls the execution of the benchmark interfacing
    an Agent class with a set Suite.
    The benchmark class must be inherited with a class that defines the
    all the experiments to be run by the agent
    """

    def __init__(
            self,
            city_name='Town01',
            name_to_save='Test',
            continue_experiment=False,
            save_images=False,
            distance_for_success=2.0
    ):
        """
        Args
            city_name:
            name_to_save:
            continue_experiment:
            save_images:
            distance_for_success:
            collisions_as_failure: if this flag is set to true, episodes will terminate as failure, when the car collides.
        """

        self.__metaclass__ = abc.ABCMeta

        self._city_name = city_name
        self._base_name = name_to_save
        # The minimum distance for arriving into the goal point in
        # order to consider ir a success
        self._distance_for_success = distance_for_success
        # The object used to record the benchmark and to able to continue after
        self._recording = Recording(name_to_save=name_to_save,
                                    continue_experiment=continue_experiment,
                                    save_images=save_images
                                    )

        # We have a default planner instantiated that produces high level commands
        self._planner = Planner(city_name)
        self._map = self._planner._city_track.get_map()

        # TO keep track of the previous collisions
        self._previous_pedestrian_collision = 0
        self._previous_vehicle_collision = 0
        self._previous_other_collision = 0

    def get_path(self):
        """
        Returns the path were the log was saved.
        """
        return self._recording.path

    def _get_directions(self, current_point, end_point):
        """
        Class that should return the directions to reach a certain goal
        """

        directions = self._planner.get_next_command(
            (current_point.location.x,
             current_point.location.y, 0.22),
            (current_point.orientation.x,
             current_point.orientation.y,
             current_point.orientation.z),
            (end_point.location.x, end_point.location.y, 0.22),
            (end_point.orientation.x, end_point.orientation.y, end_point.orientation.z))
        return directions

    def _get_shortest_path(self, start_point, end_point):
        """
        Calculates the shortest path between two points considering the road netowrk
        """

        return self._planner.get_shortest_path_distance(
            [
                start_point.location.x, start_point.location.y, 0.22], [
                start_point.orientation.x, start_point.orientation.y, 0.22], [
                end_point.location.x, end_point.location.y, end_point.location.z], [
                end_point.orientation.x, end_point.orientation.y, end_point.orientation.z])

    def _has_agent_collided(self, measurement, metrics_parameters):

        """
            This function must have a certain state and only look to one measurement.
        """
        collided_veh = 0
        collided_ped = 0
        collided_oth = 0

        if (measurement.collision_vehicles - self._previous_vehicle_collision) \
                > metrics_parameters['collision_vehicles']['threshold']/2.0:
            collided_veh = 1
        if (measurement.collision_pedestrians - self._previous_pedestrian_collision) \
                > metrics_parameters['collision_pedestrians']['threshold']/2.0:
            collided_ped = 1
        if (measurement.collision_other - self._previous_other_collision) \
                > metrics_parameters['collision_other']['threshold']/2.0:
            collided_oth = 1

        self._previous_pedestrian_collision = measurement.collision_pedestrians
        self._previous_vehicle_collision = measurement.collision_vehicles
        self._previous_other_collision = measurement.collision_other

        return collided_ped, collided_veh, collided_oth


    def _run_navigation_episode(
            self,
            agent,
            client,
            time_out,
            target,
            episode_name,
            metrics_parameters,
            collision_as_failure,
            traffic_light_as_failure):
        """
         Run one episode of the benchmark (Pose) for a certain agent.
        Args:
            agent: the agent object
            client: an object of the carla client to communicate
            with the CARLA simulator
            time_out: the time limit to complete this episode
            target: the target to reach
            episode_name: The name for saving images of this episode
            metrics_object: The metrics object to check for collisions
        """

        # Send an initial command.
        measurements, sensor_data = client.read_data()
        client.send_control(VehicleControl())

        initial_timestamp = measurements.game_timestamp
        current_timestamp = initial_timestamp

        # The vector containing all measurements produced on this episode
        measurement_vec = []
        # The vector containing all controls produced on this episode
        control_vec = []
        frame = 0
        distance = 10000
        col_ped, col_veh, col_oth = 0, 0, 0
        traffic_light_state, number_red_lights, number_green_lights = None, 0, 0
        fail = False
        success = False
        not_count = 0

        while not fail and not success:

            # Read data from server with the client
            measurements, sensor_data = client.read_data()
            # The directions to reach the goal are calculated.
            directions = self._get_directions(measurements.player_measurements.transform, target)
            # Agent process the data.
            # control = agent.run_step(measurements, sensor_data, directions, target)
            # # Send the control commands to the vehicle
            # client.send_control(control)

            # save images if the flag is activated
            #self._recording.save_images(sensor_data, episode_name, frame)

            current_x = measurements.player_measurements.transform.location.x
            current_y = measurements.player_measurements.transform.location.y

            # logging.info("Controller is Inputting:")
            # logging.info('Steer = %f Throttle = %f Brake = %f ',
            #              control.steer, control.throttle, control.brake)
            #
            # current_timestamp = measurements.game_timestamp
            # logging.info('Timestamp %f', current_timestamp)
            # # Get the distance travelled until now

            distance = sldist([current_x, current_y],
                              [target.location.x, target.location.y])
        #     # Write status of the run on verbose mode
        #     logging.info('Status:')
        #     logging.info(
        #         '[d=%f] c_x = %f, c_y = %f ---> t_x = %f, t_y = %f',
        #         float(distance), current_x, current_y, target.location.x,
        #         target.location.y)
        #     # Check if reach the target
        #     col_ped, col_veh, col_oth = self._has_agent_collided(measurements.player_measurements,
        #                                                          metrics_parameters)
        #
        #     if distance < self._distance_for_success:
        #         success = True
        #     elif (current_timestamp - initial_timestamp) > (time_out * 1000):
        #         fail = True
        #     elif collision_as_failure and (col_ped or col_veh or col_oth):
        #         fail = True
        #
        #     # Increment the vectors and append the measurements and controls.
        #     frame += 1
        #     measurement_vec.append(measurements.player_measurements)
        #     control_vec.append(control)
        #
        # if success:
        #     return 1, measurement_vec, control_vec, float(
        #         current_timestamp - initial_timestamp) / 1000.0, distance,  col_ped, col_veh, col_oth, \
        #            number_red_lights, number_green_lights
        # return 0, measurement_vec, control_vec, time_out, distance, col_ped, col_veh, col_oth, \
        #     number_red_lights, number_green_lights
