import math
from matplotlib.path import Path
import random
import gym
from gym import spaces
import sys
from . import MxsEnv
import Box2D as b2
import wandb
from earcut import earcut
import pygame
from pygame.locals import K_p, KEYDOWN, QUIT
import numpy as np
sys.path.insert(1, '/home/tu18537/dev/mxs/pymxs/models')

from pyaerso import AffectedBody, AeroBody, Body
from gym_mxs.model import Combined, calc_state, inertia, trim_points

from scipy.spatial.transform import Rotation

# Constants for rendering
# PPM = 5  # Pixels per meter (scaling factor)
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1200
SCALE = 5
GEOM_SCALE = 16.187 /2

# GEOMETRY_SCALE = 7.5

# aircaft dimensions
AIRCRAFT_HEIGHT = 270 #mm
AIRCRAFT_LENGTH = 1000 #mm

# obstacle course params
MAX_WALL_HEIGHT = 75 * GEOM_SCALE   # size of walls
N_OBSTACLE_WALLS = 5 # number of walls in obstacle course
MIN_GAP_HEIGHT = 3*(GEOM_SCALE)  # distance between wall halves
MAX_GAP_HEIGHT =5*(GEOM_SCALE)
MIN_WALL_DISTANCE = 30*GEOM_SCALE  # distance between walls
MAX_WALL_DISTANCE = 40*GEOM_SCALE  # distance between walls
MIN_GAP_WIDTH = 0.05*GEOM_SCALE   # width of walls 
MAX_GAP_WIDTH = 1.0*GEOM_SCALE   
MIN_FIRST_WALL_DISTANCE = 30*GEOM_SCALE # distance betwee(n plane and first wall
MAX_FIRST_WALL_DISTANCE = 40 * GEOM_SCALE
MIN_WALL_SPACING = 0  
MIN_GAP_OFFSET = -3 * GEOM_SCALE
MAX_GAP_OFFSET = 3  * GEOM_SCALE
INITIAL_WALL_OFFSET = 0
MIDDLE_Y = (WINDOW_HEIGHT / 2) / SCALE
LIDAR_OFFSET_VECTOR = b2.b2Vec2(4, 0)

# LIDAR params
N_LIDAR_RAYS = 10
LIDAR_RANGE = 95*GEOM_SCALE

LIDAR_FOV = 20
ANGLE_INCREMENT = LIDAR_FOV / (N_LIDAR_RAYS - 1)

#reward params
K1 = 1.0
K2 = 0.25

# aircraft gemotry
PLANE_OUTLINE_PATH = "M -8.4344006,0.8833226 L -3.6174367,1.4545926 C -2.6957014,1.5861425 -1.2977255,1.7000225 -0.44895008,0.98453256 C 0.97534922,0.9358126 2.1554971,0.9295626 3.4694746,0.8473026 C 3.4694746,0.8473026 4.1040207,0.8167026 4.1204559,0.5018026 C 4.1306045,0.3072626 4.2764544,-1.2268074 1.7485665,-1.3031174 L 1.7604066,-1.0355474 L 1.3209316,-1.0233574 L 1.3822972,-1.7538274 C 1.9074643,-1.7412074 2.0141441,-2.5891474 1.4111688,-2.6446878 C 0.80819248,-2.7002378 0.8023354,-1.8387774 1.1839183,-1.7720774 L 1.0908357,-1.0522274 L -5.2189818,-0.91913738 L -12.198397,-0.80283738 C -12.198397,-0.80283738 -12.820582,-0.84082738 -12.643322,-0.31380735 C -12.466063,0.2132026 -11.622877,3.1026526 -11.622877,3.1026526 L -10.120232,3.1500026 C -10.120232,3.1500026 -9.8463164,3.1552526 -9.6753635,2.8748926 C -9.5044154,2.5944926 -8.4343678,0.8834126 -8.4343678,0.8834126 Z"
MAIN_WING_PATH="M 0.32346345,0.1815526 C 1.8962199,0.1638926 1.9691414,-0.33848735 0.34369001,-0.39724735 C -2.0368286,-0.46197735 -3.4920188,-0.15280735 -3.3975903,-0.13907735 C -1.5720135,0.1264326 -0.81500941,0.1943226 0.32346345,0.1815526 Z"
TAIL_PLANE_PATH="M -8.9838929,0.4470726 C -7.9395132,0.4475726 -7.8954225,0.0758826 -8.975461,0.01829265 C -10.557021,-0.05024735 -11.520801,0.1663226 -11.457966,0.1773326 C -10.24323,0.3898926 -9.739887,0.4467426 -8.9838897,0.4471126 Z"

class MxsEnvBox2D(MxsEnv):


    def __init__(self, render_mode=None, reward_func=lambda obs: 0.5, timestep_limit=100, scenario=None, context=np.array([3, 5]), **kwargs):
        self.render_mode = render_mode
        super().__init__(render_mode= render_mode,**kwargs)
        
        self.steps = 0

        self.controls_high = np.array([np.radians(60), 1])
        self.controls_low = np.array([np.radians(-60), 0])

        self.reward_func = reward_func
        self.reward_state = None

        self.use_lidar = False

        if self.use_lidar is True:
            n_obs = 8 + N_LIDAR_RAYS
        else:
            n_obs = 11

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.scenario = scenario

        self.paused = False
        self.n_lidar_rays = N_LIDAR_RAYS
        self.context = context
        # print("context", self.context)
        # print(self.context[0] * GEOM_SCALE)
        self.min_gap_height = self.context * GEOM_SCALE
        # self.max_gap_height = self.context[1] * GEOM_SCALE

        self.old_obs_shape = self.observation_space.shape[0]

        new_obs_shape = self.old_obs_shape + self.n_lidar_rays + self.action_space.shape[0]


        # create bx2d world 
    
        # reset world
        # x z u w qy qw q l1 l2 l3 l4 l5 l6 l7 l8 l9 l10 elevator throttle
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1,1], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        # print(f"action space: {self.action_space}")
        # breakpoint()

        self.initial_ac_pos = (0, (WINDOW_HEIGHT / 2) / SCALE)
        self.render_screen = None
        self.clock = None
        self.x_pos_prev = 0
        self.last_obstacle_x = 0
        self.termination = False

    def _get_obs(self):
        return np.array(self.vehicle.statevector)
    
    def _get_full_obs(self, obs):
        # obs = self.long_obs(obs)
        position = (self.aircraft.position - self.initial_ac_pos) / GEOM_SCALE
        velocity = self.aircraft.linear_velocity / GEOM_SCALE
        angle = self.aircraft.angle
        angular_velocity = self.aircraft.angular_velocity

        if self.use_lidar:
            wall_obs = [l.fraction for l in self.lidars]
        else:

            wall_pos = self.calc_wall_position_delta()
            wall_height = self.get_next_wall_gap_height()
            wall_obs = [*wall_pos, wall_height]
        observation = np.array([*position, *velocity, angle, angular_velocity, *wall_obs, self.elevator, self.throttle], dtype=np.float32)
        return observation
    
    def long_obs(self, obs):
    #           x       z       u       w      qy      qw        q
        return [obs[0], obs[2], obs[3], obs[5], obs[7], obs[9], obs[11]]

    def reset(self, seed=None, return_info=False, options=None):
        self.vehicle.statevector = [
            *self.initial_state[0],
            *self.initial_state[1],
            *self.initial_state[2],
            *self.initial_state[3]
        ]

        # print("context", self.context)
        self.min_gap_height = self.context * GEOM_SCALE
        wandb.log({"context": self.min_gap_height})
        self.max_gap_height = 0
        # print("gap height", self.min_gap_height)
        # self.max_gap_height = self.context[1] * GEOM_SCALE

        self.elevator = np.radians(trim_points[self.selected_trim_point][1])
        self.throttle = trim_points[self.selected_trim_point][2]

        observation = self._get_obs()

        self.aircraft_initial_angle = self.get_pitch(*self.initial_state[2])
        self.aircraft_initial_velocity = self.get_velocity(observation)[0]
        self.wall_i = 0
        self.reset_world()
        observation = self._get_full_obs(observation)

        

        self.steps = 0
        self.reward_state = None
        self.scroll_offset = 0

        
        self.termination = False

        info = {}

        if self.render_mode == "human":
            self._render_frame()
        return (observation,info) if return_info else observation

    def step(self, action):
        # print(f"action: {action}")

        controls = self.action_mapper(
            action, 
            self.action_space.low, self.action_space.high,
            self.controls_low, self.controls_high)

        # print(f"controls: {controls}")
        # get actions
        self.elevator = np.clip(self.elevator + controls[0] * self.dT, np.radians(-30), np.radians(30))
        self.throttle = controls[1]

        # print(f"elevator: {self.elevator}")
        # print(f"throttle: {self.throttle}")
        # apply actions
        # breakpoint()
        
        self.vehicle.step(self.dT,[0,self.elevator,self.throttle,0])
        #get observation from mxs
        observation = self._get_obs()
        # print(observation)

        #[x,y,z, u,v,w, qx,qy,qz,qw, p,q,r] 

        linear_velocities, angular_velocity = self.get_velocity(observation)


        self.aircraft.linear_velocity = linear_velocities
 
        self.aircraft.angular_velocity = angular_velocity
        
        # step box2d
        # print(f"linear vel: {linear_velocities}")
        self.world.Step(self.dT, 8 , 3) #TODO: what are these numbers?
        # print(f"box2d vel: {self.aircraft.linear_velocity}")
        # print(linear_velocities[0]/GEOM_SCALE)
        self.pitch = math.degrees(self.get_pitch(*observation[6:10]))


        # print(f"Aircraft pos: {self.aircraft.position / GEOM_SCALE}")
        # print(f"Next wall position {self.get_next_wall_position()}")
        # print(f"Get wall position delta {self.calc_wall_position_delta()}")
        # print(f"Get next wall height {self.get_next_wall_gap_height()}")

        # breakpoint()
        # print(f"mxs pos: {observation[0]}")
        # print("box2d pos: ", (self.aircraft.position[0]/GEOM_SCALE)) 
        # print(f"MXS pitch angle {math.degrees(self.get_pitch(*observation[6:10]))}")
        # print(f"box2d pitch angle {math.degrees(self.aircraft.angle)}")
        # print(f" MXS vel: {observation[3], observation[5]}")
        # print(f" set vel: {observation[3]*GEOM_SCALE}")
        # print(f"box2d vel: {self.aircraft.linear_velocity/GEOM_SCALE}")
        # print(f" box2d body frame velocity {self.world_to_body_frame(self.aircraft.linear_velocity/GEOM_SCALE, self.aircraft.angle)}")

        self.update_lidar()

        observation = self._get_full_obs(observation)
        # print(f"observation: {observation}")

        reward = self.obstacle_avoid_box2d(observation)
        # print(f"reward: {reward}")
        self.steps += 1
        # done = ep_done or self.steps >= self.timestep_limit
        done = self.termination 

        if self.aircraft.position.x * SCALE > WINDOW_WIDTH / 2:
            self.scroll_offset = self.aircraft.position.x * SCALE - WINDOW_WIDTH / 2

        if self.render_mode == "human":
            self._render_frame()
        # print(observation)
        return observation, reward, done, {}
    
    def get_velocity(self, state):
        """ Returns the u and w velocities and q angular velocity
        of the aircraft, based on whether the env has standard or longitunda only
        observation space."""
        return (state[3]*GEOM_SCALE, state[5]*GEOM_SCALE), state[11]
        
    def action_mapper(self, action, input_low, input_high, output_low, output_high):
    
        return (action - input_low) * (output_high - output_low) / (input_high - input_low) + output_low
    
    def update_lidar(self):
        pos = self.aircraft.position
        angle = self.aircraft.angle
        offset_vector = b2.b2Vec2(LIDAR_OFFSET_VECTOR)  # Define the offset vector for the front of the aircraft

        # Rotate the offset vector according to the body's angle
        rotated_offset_vector = b2.b2Vec2(
            offset_vector.x * math.cos(angle) - offset_vector.y * math.sin(angle),
            offset_vector.x * math.sin(angle) + offset_vector.y * math.cos(angle),
        )

        # Calculate the starting point of the lidar beams
        lidar_start_point = pos + rotated_offset_vector

        for i in range(N_LIDAR_RAYS):
            self.lidars[i].fraction = 1.0
            angle_offset = (0.5 * i / 10.0) - 5.0  # Adjust the starting angle of the lidar beams
            # angle_offset = (1 * i / N_LIDAR_RAYS) - 5.25
            # angle_offset = (ANGLE_INCREMENT * i) - (LIDAR_FOV / 2)
            start_angle = angle + angle_offset
            # end_angle = angle + 1.5 * i / 10.0 # why these magic numbers?

            self.lidars[i].p1 = lidar_start_point
            self.lidars[i].p2 = (
                pos[0] + math.sin(start_angle) * LIDAR_RANGE,
                pos[1] - math.cos(start_angle) * LIDAR_RANGE,
            )
            self.world.RayCast(self.lidars[i], self.lidars[i].p1, self.lidars[i].p2)
    
    def game_over(self):
        # print("Contact")
        if self.contact_listener.game_over_collision:
            self.contact_listener.game_over_collision = False
            return True
        return False
    
    def obstacle_avoid_box2d(self, obs):

        # print(f"obs: {obs}")
        x_pos = obs[0]
        alt = obs[1]
        u_vel = obs[2]
        
        desired_alt = 0
        x_pos_prev = self.x_pos_prev
        # pitch = self.pitch
        pitch = self.aircraft.angle

        # print(f"X pos: {x_pos}")
        # print(f"X pos prev: {x_pos_prev}")
        # print(f"Alt: {alt}")
        # print(f"Desired Alt: {desired_alt}")
        # print(f"self.last_obstacle_x: {self.last_obstacle_x}")
        # print(f"wall max height: {MAX_WALL_HEIGHT/GEOM_SCALE}")

    

        # intitial_position = env.initial_state[0]

        # # print(f"z_error: {z_error}, pitch_error: {pitch_error}, x_error: {x_error}")
        # pitch_r = rewards.tolerance(pitch_error, bounds = (-5, 5), margin = 90)
        # alt_r = rewards.tolerance(z_error, bounds = (-2.5, 2.5), margin = 15)
        # x_r = rewards.tolerance(x_error, bounds = (-2.5, 2.5), margin =20.0)

        # reward = (pitch_r * 0.4)  + (alt_r * 0.4) + (x_r * 0.2)

        rw_1 = K1* ((x_pos - x_pos_prev)/self.dT)
        # print(f"rw_1: {rw_1}")
        reward = rw_1
        self.x_pos_prev = x_pos

        # alitude maintenance
        rw2 = -K2 * abs(desired_alt - alt)

        # print(f"rw2: {rw2}")
        # reward += rw2

        if self.game_over() or abs(alt) > MAX_WALL_HEIGHT/GEOM_SCALE:
            # reward = -100
            self.termination = True


        if x_pos > self.last_obstacle_x:
            print(f"Passed obstacle at {x_pos}")
            self.termination = True
            print("Finished!")
        if abs(math.degrees(pitch)) > 85:
            reward = -100
            self.termination = True
        # print(f"Reward {reward}")
        if u_vel > 24.5:
            reward = -100
            self.termination = True
        # if reward < 0:
            # print(obs)
            # # print(f"altitude: {alt}")
            # # print(f"alt error: {desired_alt - alt}")
            # print(f" x pos: {x_pos}")
            # print(f" x pos prev: {x_pos_prev}")
            # print(f"reward: {reward}")
            # print(f"rw_1: {rw_1}")
            # print(f"rw2: {rw2}")
        
        return reward
    
    def body_to_world_frame(self, body_velocity, aircraft_angle):
    # Calculate the rotation matrix for the aircraft's orientation
        rot_matrix = [
            [math.cos(aircraft_angle), -math.sin(aircraft_angle)],
            [math.sin(aircraft_angle), math.cos(aircraft_angle)]
        ]

        # Transform the body frame linear velocity to the world frame
        world_velocity_x = rot_matrix[0][0] * body_velocity[0] + rot_matrix[0][1] * body_velocity[1]
        world_velocity_y = rot_matrix[1][0] * body_velocity[0] + rot_matrix[1][1] * body_velocity[1]

        return world_velocity_x, world_velocity_y
    
    def reset_world(self):
        self.world = None

        self.world = b2.b2World(gravity=(0, 0))

        self.create_aircraft_and_obstacles()

        # body_def = b2.b2BodyDef()
        # body_def.type = b2.b2_kinematicBody
        # body_def.position = b2.b2Vec2(15.5, MIDDLE_Y-1)

        # top_wall_body = self.world.CreateBody(body_def)
        # wall_shape = b2.b2PolygonShape()
        # wall_shape.SetAsBox(SCALE*AIRCRAFT_LENGTH/1000, SCALE*AIRCRAFT_HEIGHT/1000)
        # top_wall_body.CreateFixture(shape=wall_shape)

        self.contact_listener = self.ContactListener(self.aircraft.aircraft_body)
        self.world.contactListener = self.contact_listener

        class LidarCallback(b2.b2RayCastCallback):
                def ReportFixture(self, fixture, point, normal, fraction):
                    if (fixture.filterData.categoryBits & 1) == 0:
                        return -1
                    self.p2 = point
                    self.fraction = fraction
                    return fraction

        self.lidars = [LidarCallback() for _ in range(N_LIDAR_RAYS)]
        self.update_lidar()

    def create_aircraft_and_obstacles(self):
        self.aircraft = MXSGeometry(self.world, initial_position=self.initial_ac_pos, initial_angle=self.aircraft_initial_angle, initial_velocity=self.aircraft_initial_velocity)
        obstacle_geom = ObstacleGeometry(self.world, initial_position=self.initial_ac_pos, min_gap_height=self.min_gap_height, max_gap_height=self.max_gap_height)
        self.last_obstacle_x = obstacle_geom.last_wall_x
        # print(f"Wall positions: {obstacle_geom.wall_positions}")
        # print(f"Wall heights: {obstacle_geom.wall_heights}")
        # print(f"Wall widths: {obstacle_geom.wall_widths}")
        self.wall_positions = obstacle_geom.wall_positions
        self.wall_heights = obstacle_geom.wall_heights
        self.wall_widths = obstacle_geom.wall_widths
        self.x_pos_prev = 0

    # def get_next_obstacle_position(self):
    #     """ A function to get the absolute position of the next obstacle """
    #     obstacle_i = self.obstacle_i
    #     return self.obstacle_positions[obstacle_i]
    
    def calc_wall_position_delta(self):
        """ A function to calculate the position delta between the aircraft and the next obstacle """
        aircraft_position = self.aircraft.position/GEOM_SCALE 
        next_obstacle_position = self.get_next_wall_position()
        delta_x = next_obstacle_position[0] - aircraft_position[0]
        delta_y = next_obstacle_position[1] - aircraft_position[1]

        return (delta_x, delta_y)
  
    def get_next_wall_position(self):
        """A method to return the position of the next obstacle relative to the aircraft,
        based on the aicraft's current position"""
        aircraft_position = self.aircraft.position/GEOM_SCALE
        if aircraft_position[0] > (self.wall_positions[self.wall_i][0] + self.wall_widths[self.wall_i]):
            if self.wall_i < (len(self.wall_positions) - 1):
            # breakpoint()
                self.wall_i += 1
            
        if self.wall_i == len(self.wall_positions) - 1:
            return (self.last_obstacle_x, MIDDLE_Y)
      

        # wall_centre_x = self.wall_positions[self.wall_i][0]
        # wall_centre_y = self.wall_positions[self.wall_i][1]
        
        return self.wall_positions[self.wall_i]
    
    def get_next_wall_gap_height(self):
        """A method to return the height of the next obstacle relative to the aircraft,
        based on the aicraft's current position"""
        return self.wall_heights[self.wall_i]


    class ContactListener(b2.b2ContactListener):
        def __init__(self, aircraft_body):
            self.aircraft_body = aircraft_body
            super().__init__()
            self.game_over_collision = False

        def BeginContact(self, contact):
            body_a, body_b = contact.fixtureA.body, contact.fixtureB.body
            if (body_a == self.aircraft_body and body_b.type == b2.b2_staticBody) or (body_a.type == b2.b2_staticBody and body_b == self.aircraft_body):
                self.game_over_collision = True
                # pass

    def render(self):
        if self.render_mode == "ansi" or self.render_mode == "human":
            elements = ", ".join([f"{v:.{4}f}" for v in self._get_obs()])
            return f"[{elements},{self.vehicle.airstate[0]},{self.vehicle.airstate[2]},{self.elevator},{self.throttle}]"
        
    def _render_frame(self):
        if self.render_screen is None and self.render_mode == "human":
            print("Initializing pygame")
            pygame.init()
            self.scroll = 0
            self.scroll_offset = 0
            self.render_screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), 0, 32)
            pygame.display.set_caption("Box2D World Rendering")
            
            self.surf = pygame.Surface(
                (WINDOW_HEIGHT + max(0.0, self.scroll) * SCALE, WINDOW_WIDTH)
                )
            pygame.transform.scale(self.surf, (SCALE, SCALE))
        # self.clock = pygame.time.Clock()
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        pos = self.aircraft.position
        # print(f"{pos.x} {pos.y}")
        self.scroll = pos.x - WINDOW_WIDTH / SCALE / 5
        self.render_screen.fill((135, 206, 235))
    # Draw the world first
        self.draw_world()

        pygame.display.flip()
        self.clock.tick(240)

    def close(self):
        if self.render_screen is not None:
            # pygame.display.quit()
            pygame.quit

    def draw_world(self):
    # Draw bodies
        for body in self.world.bodies:
            for fixture in body.fixtures:
                if type(fixture.shape) == b2.b2PolygonShape:
                    color = (255, 0, 0)  # Default color: blue
                    if body == self.aircraft.aircraft_body:  # If the body is the aircraft, set the color to green
                        color = (0, 255, 0)
                    self.draw_polygon(fixture.shape, body, color)

            # Draw lidar rays
        for lidar in self.lidars:
            if hasattr(lidar, "p1") and hasattr(lidar, "p2"):
                start_screen = self.world_to_screen(lidar.p1)
                end_screen = self.world_to_screen(lidar.p2)
                pygame.draw.line(self.render_screen, (255, 0, 0), start_screen, end_screen, 1)

        pygame.display.flip()

    def get_pitch(self, qx,qy,qz,qw):
        if True:
            # Can't use scipy if 'jit'ting
        #     rot = Rotation.from_quat(np.array([qx,qy,qz,qw]).T)
        #     [yaw, pitch, roll] = rot.as_euler('zyx', degrees=False)
        #     # print([yaw,pitch,roll])
        #     if yaw != 0:
        #         if pitch > 0:
        #             pitch = np.pi/2 + (np.pi/2 - pitch)
        #         else:
        #             pitch = -np.pi/2 + (-np.pi/2 - pitch)
        # if False:
            sinp = np.sqrt(1 + 2 * (qw*qy - qx*qz))
            cosp = np.sqrt(1 - 2 * (qw*qy - qx*qz))
            pitch = 2 * np.arctan2(sinp, cosp) - np.pi / 2
        return pitch
    
    def draw_polygon(self, polygon, body, color):
        vertices = [(body.transform * v) * SCALE for v in polygon.vertices]
        vertices = [(v[0] - self.scroll_offset, WINDOW_HEIGHT - v[1]) for v in vertices]
        pygame.draw.polygon(self.render_screen, color, vertices)

    def world_to_screen(self, world_point):
        screen_x = int(world_point[0] * SCALE - self.scroll_offset)
        screen_y = int(WINDOW_HEIGHT - world_point[1] * SCALE)
        return (screen_x, screen_y)
        
class MXSGeometry():
    """A class to store and process the geometry of the MXS aircraft"""

    PLANE_OUTLINE_PATH = "M -8.4344006,0.8833226 L -3.6174367,1.4545926 C -2.6957014,1.5861425 -1.2977255,1.7000225 -0.44895008,0.98453256 C 0.97534922,0.9358126 2.1554971,0.9295626 3.4694746,0.8473026 C 3.4694746,0.8473026 4.1040207,0.8167026 4.1204559,0.5018026 C 4.1306045,0.3072626 4.2764544,-1.2268074 1.7485665,-1.3031174 L 1.7604066,-1.0355474 L 1.3209316,-1.0233574 L 1.3822972,-1.7538274 C 1.9074643,-1.7412074 2.0141441,-2.5891474 1.4111688,-2.6446878 C 0.80819248,-2.7002378 0.8023354,-1.8387774 1.1839183,-1.7720774 L 1.0908357,-1.0522274 L -5.2189818,-0.91913738 L -12.198397,-0.80283738 C -12.198397,-0.80283738 -12.820582,-0.84082738 -12.643322,-0.31380735 C -12.466063,0.2132026 -11.622877,3.1026526 -11.622877,3.1026526 L -10.120232,3.1500026 C -10.120232,3.1500026 -9.8463164,3.1552526 -9.6753635,2.8748926 C -9.5044154,2.5944926 -8.4343678,0.8834126 -8.4343678,0.8834126 Z"
    MAIN_WING_PATH="M 0.32346345,0.1815526 C 1.8962199,0.1638926 1.9691414,-0.33848735 0.34369001,-0.39724735 C -2.0368286,-0.46197735 -3.4920188,-0.15280735 -3.3975903,-0.13907735 C -1.5720135,0.1264326 -0.81500941,0.1943226 0.32346345,0.1815526 Z"
    TAIL_PLANE_PATH="M -8.9838929,0.4470726 C -7.9395132,0.4475726 -7.8954225,0.0758826 -8.975461,0.01829265 C -10.557021,-0.05024735 -11.520801,0.1663226 -11.457966,0.1773326 C -10.24323,0.3898926 -9.739887,0.4467426 -8.9838897,0.4471126 Z"    

    def __init__(self, world=None, render_mode="human", initial_position=(0, 0), initial_angle=0, initial_velocity=(0, 0)):
        self.world = world
        self.create_aircraft_body(initial_position, initial_angle, initial_velocity)

    def _parse_path(self, path):
        vertices = []
        codes = []
        parts = path.split()
        code_map = {
            'M': Path.MOVETO,
            'C': Path.CURVE4,
            'L': Path.LINETO,
        }
        i = 0
        while i < len(parts) - 1:
            if parts[i] in code_map:
                path_code = code_map[parts[i]]
                code_len = 1
            else:
                path_code = code_map['l']
                code_len = 0
            npoints = Path.NUM_VERTICES_FOR_CODE[path_code]
            codes.extend([path_code] * npoints)
            vertices.extend([[*map(float, y.split(','))]
                            for y in parts[i+code_len:][:npoints]])
            i += npoints + code_len
        return vertices, codes

    def _path_to_vertices(self, path):
        vertices, _ = self._parse_path(path)
        vertices = self.scaled_vertices(vertices, scale=0.5)

        return [b2.b2Vec2(x, y) for x, y in vertices]

    def update_velocities(self, linear_velocity, angular_velocity):
        self.aircraft_body.linearVelocity = linear_velocity
        self.aircraft_body.angularVelocity = angular_velocity

    @property
    def position(self):
        return self.aircraft_body.position
    
    @property
    def angle(self):
        return self.aircraft_body.angle
    
    @property
    def linear_velocity(self):
        return self.aircraft_body.linearVelocity
    
    @property
    def angular_velocity(self):
        return self.aircraft_body.angularVelocity
    
    @property
    def angle(self):
        return self.aircraft_body.angle


    @linear_velocity.setter
    def linear_velocity(self, linear_velocity):
        self.aircraft_body.linearVelocity = linear_velocity

    @angular_velocity.setter
    def angular_velocity(self, angular_velocity):
        self.aircraft_body.angularVelocity = angular_velocity


    def scaled_vertices(self, vertices, scale=1):
        return [(x * scale, y * scale) for x, y in vertices]

    def create_aircraft_body(self, initial_position, initial_angle, initial_velocity):
   

        fuselage_vertices = self._path_to_vertices(PLANE_OUTLINE_PATH)
        main_wing_vertices = self._path_to_vertices(MAIN_WING_PATH)
        tail_plane_vertices = self._path_to_vertices(TAIL_PLANE_PATH)
        aircraft_vertices = fuselage_vertices + main_wing_vertices + tail_plane_vertices
        aircraft_vertices_flat = [coord for vertex in aircraft_vertices for coord in (vertex.x, vertex.y)]
        

        aircraft_body_def = b2.b2BodyDef()
        aircraft_body_def.type = b2.b2_dynamicBody
        aircraft_body_def.position = b2.b2Vec2(0, MIDDLE_Y )
        aircraft_body_def.angle = initial_angle
        aircraft_body_def.linearVelocity = initial_velocity
        aircraft_body_def.linear_damping = 0.0

        # Create the aircraft body first
        self.aircraft_body = self.world.CreateBody(aircraft_body_def)

        triangles_indices = earcut.earcut(aircraft_vertices_flat)

        for i in range(0, len(triangles_indices), 3):
            triangle_shape = b2.b2PolygonShape()
            triangle_shape.vertices = [
                    aircraft_vertices[triangles_indices[i]],
                    aircraft_vertices[triangles_indices[i + 1]],
                    aircraft_vertices[triangles_indices[i + 2]],
                ]
            

            # Create the fixture on the aircraft body inside the loop
            self.aircraft_body.CreateFixture(shape=triangle_shape)



class ObstacleGeometry():
    """A class to store and process the geometry of the obstacles"""

    def __init__(self, world=None, render_mode="human", initial_position=(0, 0), min_gap_height=0, max_gap_height=0):
        self.world = world
        self.initial_ac_pos = MIDDLE_Y
        self.min_gap_height = min_gap_height
        self.max_gap_height = max_gap_height
        self._wall_positions = []
        self._wall_heights = []
        self._wall_widths = []
        self.create_obstacles()
       


    def define_placement(self):
        first_wall_distance = random.uniform(MIN_FIRST_WALL_DISTANCE, MAX_FIRST_WALL_DISTANCE)
        wall_positions = [first_wall_distance] + [random.uniform(MIN_WALL_DISTANCE + MIN_WALL_SPACING, MAX_WALL_DISTANCE) for _ in range(N_OBSTACLE_WALLS - 1)]
        cumulative_wall_positions = [sum(wall_positions[:i+1]) for i in range(len(wall_positions))]

        # print(cumulative_wall_positions)

        return cumulative_wall_positions, first_wall_distance
    
    def create_obstacles(self):
        cumulative_wall_positions, first_wall_distance = self.define_placement()
        for i in range(N_OBSTACLE_WALLS):
            self.create_wall(cumulative_wall_positions, first_wall_distance, i)
    
    def create_wall(self, cumulative_wall_positions, first_wall_distance, i):
        # gap_height = random.uniform(self.min_gap_height, self.max_gap_height)

        gap_height = self.min_gap_height[0]
        wall_gap_width = random.uniform(MIN_GAP_WIDTH, MAX_GAP_WIDTH)
        gap_offset = random.uniform(MIN_GAP_OFFSET, MAX_GAP_OFFSET)
        gap_center_y = self.initial_ac_pos + gap_offset

        # print(gap_height, wall_gap_width, gap_offset, gap_center_y)

        top_wall_y = gap_center_y + gap_height / 2
        wall_body_def = b2.b2BodyDef()
        wall_body_def.type = b2.b2_staticBody
        wall_body_def.position = b2.b2Vec2(INITIAL_WALL_OFFSET + cumulative_wall_positions[i], top_wall_y + (MAX_WALL_HEIGHT - gap_height) / 2)

        self._wall_positions.append((wall_body_def.position.x/GEOM_SCALE, gap_center_y/GEOM_SCALE))
        self._wall_heights.append(gap_height/GEOM_SCALE)
        self._wall_widths.append(wall_gap_width/GEOM_SCALE)

        top_wall_body = self.world.CreateBody(wall_body_def)
        wall_shape = b2.b2PolygonShape()
        wall_shape.SetAsBox(wall_gap_width / 2, (MAX_WALL_HEIGHT - gap_height) / 2)
        top_wall_body.CreateFixture(shape=wall_shape)

        # Bottom part of the wall
        bottom_wall_y = gap_center_y - gap_height / 2
        wall_body_def = b2.b2BodyDef()
        wall_body_def.type = b2.b2_staticBody
        wall_body_def.position = b2.b2Vec2(INITIAL_WALL_OFFSET + cumulative_wall_positions[i], bottom_wall_y - (MAX_WALL_HEIGHT - gap_height) / 2)

        bottom_wall_body = self.world.CreateBody(wall_body_def)
        wall_shape = b2.b2PolygonShape()
        wall_shape.SetAsBox(wall_gap_width / 2, (MAX_WALL_HEIGHT - gap_height) / 2)
        bottom_wall_body.CreateFixture(shape=wall_shape)

        # if i is last i
        if i == N_OBSTACLE_WALLS - 1:
            # print(f"final wall{INITIAL_WALL_OFFSET + cumulative_wall_positions[i] + wall_gap_width}")
            # get last wall x + a little bit extra to account for plane fully passing through
            self._last_wall_x = (INITIAL_WALL_OFFSET + cumulative_wall_positions[i] + wall_gap_width + 10*GEOM_SCALE)/GEOM_SCALE
            # print(self._last_wall_x)

    @property
    def last_wall_x(self):
        return self._last_wall_x
    
    @property
    def wall_positions(self):
        return self._wall_positions
    
    @property
    def wall_heights(self):
        return self._wall_heights
    
    @property
    def wall_widths(self):
        return self._wall_widths
    


