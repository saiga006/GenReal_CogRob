import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedEnvCfg, ViewerCfg
from isaaclab.managers import ActionTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.envs import mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import KINOVA_GEN3_N7_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import os

# Get the base path for assets
KITCHEN_ASSETS_DIR = os.path.dirname(__file__) + "/../assets"

@configclass
class KitchenTeleopSceneCfg(InteractiveSceneCfg):
    """Kitchen scene configuration for teleoperation."""
    
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    
    # Robot table
    table = RigidObjectCfg(
        prim_path="/World/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 1.2, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.2, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.0, 1.75, 0.4),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # Kinova robot with gripper
    robot = KINOVA_GEN3_N7_CFG.replace(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{KITCHEN_ASSETS_DIR}/Kinova/kinova_gen3_robotiq_2f_85_working.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(2.4, 1.25, 0.8),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "joint_1": 0.0,
                "joint_2": -0.3,
                "joint_3": 0.0,
                "joint_4": -1.0,
                "joint_5": 0.0,
                "joint_6": 0.7,
                "joint_7": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-7]"],
                velocity_limit=100.0,
                effort_limit={
                    "joint_[1-4]": 39.0,
                    "joint_[5-7]": 9.0,
                },
                stiffness={
                    "joint_[1-4]": 40.0,
                    "joint_[5-7]": 15.0,
                },
                damping={
                    "joint_[1-4]": 1.0,
                    "joint_[5-7]": 0.5,
                },
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["robotiq_.*"],
                velocity_limit=0.2,
                effort_limit=200.0,
                stiffness=2000.0,
                damping=100.0,
            ),
        },
    )
    
    # Kitchen objects (using your existing configurations)
    insulated_shelf = ArticulationCfg(
        prim_path="/World/InsulatedShelf",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{KITCHEN_ASSETS_DIR}/Kitchen_InsularShelf/Kitchen_InsularShelf.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(2.5, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "shelf_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=50.0,
                velocity_limit=0.5,
                stiffness=1000.0,
                damping=100.0,
            ),
        },
    )
    
    microwave = ArticulationCfg(
        prim_path="/World/Microwave",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{KITCHEN_ASSETS_DIR}/Microwave052/Microwave052.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(2.3, 0.3, 1.25),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "microwave_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*door.*|.*hinge.*"],
                effort_limit=20.0,
                velocity_limit=0.3,
                stiffness=500.0,
                damping=50.0,
            ),
        },
    )
    
    fridge = ArticulationCfg(
        prim_path="/World/Fridge",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{KITCHEN_ASSETS_DIR}/Refrigerator036/Refrigerator036.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.5, 0.0, 0.9),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "fridge_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*door.*|.*hinge.*"],
                effort_limit=30.0,
                velocity_limit=0.2,
                stiffness=800.0,
                damping=80.0,
            ),
        },
    )
    
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
    )

@configclass
class ActionsCfg:
    """Action configuration for teleoperation."""
    
    # Robot arm control using differential inverse kinematics
    arm_action = ActionTermCfg(
        class_type=mdp.DifferentialInverseKinematicsAction,
        asset_name="robot",
        body_name="tool0",  # End effector body name (adjust based on your robot)
        controller=mdp.DifferentialIKControllerCfg(
            command_type="pose", 
            use_relative_mode=True,
            ik_method="dls"
        ),
    )
    
    # Gripper control
    gripper_action = ActionTermCfg(
        class_type=mdp.BinaryJointPositionAction,
        asset_name="robot",
        joint_names=["robotiq_85_left_finger_joint"],  # Adjust based on actual gripper joint names
        open_command_expr={"robotiq_85_left_finger_joint": 0.0},
        close_command_expr={"robotiq_85_left_finger_joint": 0.8},
    )

@configclass
class ObservationsCfg:
    """Observation configuration for teleoperation."""
    
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy observations."""
        
        # Robot joint positions and velocities
        arm_joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-7]"])},
        )
        arm_joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-7]"])},
        )
        
        # End effector pose
        ee_pose = ObservationTermCfg(
            func=mdp.body_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["tool0"])},
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()

@configclass
class KitchenTeleopEnvCfg(ManagerBasedEnvCfg):
    """Kitchen teleoperation environment configuration."""
    
    # Scene configuration
    scene = KitchenTeleopSceneCfg(num_envs=1, env_spacing=5.0)
    
    # Manager configurations
    actions = ActionsCfg()
    observations = ObservationsCfg()
    
    # Environment settings
    episode_length_s = 300.0  # 5 minutes for teleoperation
    decimation = 2
    
    # Viewer settings
    viewer = ViewerCfg(eye=(4.0, 2.0, 3.0), lookat=(2.0, 0.0, 1.0))
