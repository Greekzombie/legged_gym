import numpy as np
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO, FEET_ORIGIN, AVERAGE_MEASUREMENT
from legged_gym.envs.base.base_config import Default

# NAMES OF JOINTS IN URDF FILE
# front left leg
FL_HAA = 'FL_hip_joint' # hip AA (abduction/adduction)
FL_HFE = 'FL_thigh_joint' # hip FE (flexion/extension)
FL_KFE = 'FL_calf_joint' # knee FE

FR_HAA = 'FR_hip_joint' # front right test test test
FR_HFE = 'FR_thigh_joint'
FR_KFE = 'FR_calf_joint'

HL_HAA = 'RL_hip_joint'  # hind (back) left
HL_HFE = 'RL_thigh_joint'
HL_KFE = 'RL_calf_joint'

HR_HAA = 'RR_hip_joint'  # hind (back) right
HR_HFE = 'RR_thigh_joint'
HR_KFE = 'RR_calf_joint'

INVERT_HIND = True
MEASURE_HEIGHTS = True # this impacts several params

class Solo12Cfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_actions = 12    # Number of actions robot can perform. In this case, the SOLO robot has 12 DOF, one in each leg
        num_envs = 4096     # Number of robots present at the same time

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'    # plane or trimesh
        steps_height_scale = 0.5
        curriculum = True
        measure_heights = MEASURE_HEIGHTS
        horizontal_scale = 0.05 # [m]
        horizontal_difficulty_scale = 0.6 
        # We can modify terrain types depending on what we want to train robot on.
       #  terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gap, pit]
        terrain_proportions = [0.05,          0,        0.1,        0.1,          0,        0,          0.7,   0.05]
      
        # The measured points of terrain serve as input to the robot. 
        measured_points_x = np.arange(-1, 1.005, 0.05).tolist() # 0.8mx1.2m rectangle (without center line)
        measured_points_y = np.arange(-0.5, 0.505, 0.05).tolist()

    class init_state( LeggedRobotCfg.init_state ):
        default_joint_angles = { # = target angles [rad] when action = 0.0
            
            FL_HAA: 0.,
            FL_HFE: 0.9,
            FL_KFE: -1.64,

            FR_HAA: 0.,
            FR_HFE: 0.9,
            FR_KFE: -1.64,

            HL_HAA: 0.,
            HL_HFE: -0.9 * -1 if INVERT_HIND else 1,
            HL_KFE: 1.64 * -1 if INVERT_HIND else 1,

            HR_HAA: 0.,
            HR_HFE: -0.9 * -1 if INVERT_HIND else 1,
            HR_KFE: 1.64 * -1 if INVERT_HIND else 1

        }
        pos = [0.0, 0.0, 0.25]

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters: (paper page 5)
        # "joint" will apply to all DoF since all joint names contain "joint" 
        stiffness = { "joint": 3. } # {'HAA': 3., 'HFE': 3., 'KFE': 3.}  # K_p [N*m/rad]
        damping = { "joint": 0.2 }  # {'HAA': .2, 'HFE': .2, 'KFE': .2}     # K_d [N*m*s/rad]

        action_scale = 0.3 # paper (page 6)
        feet_height_target = 0.06 # p_z^max [m]

        decimation = Default()

    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = False
        base_height_target = 0.215
        tracking_sigma = 0.25

        height_estimation = FEET_ORIGIN

        class curriculum ( LeggedRobotCfg.rewards.curriculum ):
            """
            P. Class for negative rewards. All treated equally. More precisely, negative rewards are allowed at the end of training.
            Rewards with an associated negative scale are the ones subjected to a curriculum.
            """
            enabled = True     #P Let's see what happens if we turn curriculum off. #PP DON'T Things won't work well. 
            delay = 500
            duration = 3500
            interpolation = 1.5

        # you can use specific curriculum (even for positive rewards) this way:
        #class <reward_name>_curriculum(CurriculumConfig):
        #   your specific config here...

        class test_curriculum( LeggedRobotCfg.rewards.curriculum ):
            """P
            Define a curriculum for the a specific reward, be it positive or negative.
            """
            enabled = True     
            delay = 500
            duration = 3500
            interpolation = 1.5

        
        class scales( ):
            """P
            Defines the weight of each reward we have introduced at the end of the "solo.py" file. If we look at the "cassie_config.py"
            file, scales() inherits from "LeggedRobotCfg.rewards.scales". Here, we are not doing that. What this means is that the scales
            introduced in "legged_robot" are not used. 

            Given the available scales the "_prepare_reward_function()" matches the scales to defined reward functions. Even in the case
            where scales doesn't inherit from "LeggedRobotCfg.rewards.scales", Solo12 class can still access these reward functions as it
            was defined as
            
                                class Solo12(LeggedRobot):
            
            """

            tracking_lin_vel = 6. # c_vel
            tracking_ang_vel = 6.
            #go_to_spot = 50
            #exploration = 30
    
            foot_clearance = -20. # -c_clear -20
            foot_slip = -2. # -c_slip
            roll_pitch = -4. # -c_orn
            vel_z = -2 # -c_vz
            joint_pose = -0.5 # -c_q
            power_loss = -0.1 # -c_E
            smoothness_1 = -2.5 # -c_a1
            smoothness_2 = -1.5 # -c_a2

            collision = -1.
            base_height = -2. 

            termination = -1
            step_forecast = -0

            # I have modified these reward functions.
            test = 0.0
            torques = -0
            base_height = -0.  #P Positive base_height makes robots crouch. Negative base_height makes robots stand tall but not jump
            going_forward = 0
            vel_x = 0

    class commands( LeggedRobotCfg.commands ):
        class curriculum( LeggedRobotCfg.commands.curriculum ):
            enabled = False   #P This originaly was set false
            duration = 1500
            interpolation = 2

        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-1.5, 1.5]
            lin_vel_y = [-1, 1]
            ang_vel_yaw = [-1., 1.]
            heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/solo12/solo12_isaac.urdf'
        name = "solo"
        foot_name = 'foot'
        shoulder_name = 'hip'
        collapse_fixed_joints = False # otherwise feet are collapsed, and robot.feet_indices is not populated

        flip_visual_attachments = False # fix visual problem with meshes
        terminate_after_contacts_on = ["base", "trunk"] # TODO: why are contacts on base not terminating the episode?
        penalize_contacts_on = ["thigh"]
        self_collisions = 1

        feet_offset = 0.015

    class normalization( LeggedRobotCfg.normalization ):
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            lin_vel = Default() if MEASURE_HEIGHTS else 0 # not available on real robot
        
        clip_measurements = 0.25
    

    class noise( LeggedRobotCfg.noise ):
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            lin_vel = Default() if MEASURE_HEIGHTS else 0 # not available on real robot
            gravity = Default() if MEASURE_HEIGHTS else 0.08 # increase noise for robustness on real robot
    
    class sim( LeggedRobotCfg.sim ):
        dt = Default()
    def eval(self):
        super().eval()
        self.viewer.follow_env = True
        # IF I UNCOMMENT THESE COMMANDS ARE NOT ALLOWED. PLAY.PY DOES NOT WORK WELL WITH THIS
        #self.commands.ranges.lin_vel_x = [0.,0.]
        #self.commands.ranges.lin_vel_y = [0,0]
        #self.env.num_envs = 1
        #self.commands.ranges.ang_vel_yaw = [0,0]
        #self.commands.ranges.heading = [0,0]

class Solo12CfgPPO( LeggedRobotCfgPPO ):

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'solo12'
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        max_iterations = 100000   #P Max iterations. Might affect the curriculum.

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        learning_rate = Default() #0.005 #requested in the paper, but not working at all...

        train_step_estimator = False
        