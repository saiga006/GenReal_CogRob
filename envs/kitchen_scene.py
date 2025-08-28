import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import FRANKA_PANDA_CFG
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import spawners

import os
import math


# Import the pre-configured Kinova Gen3 configuration
from isaaclab_assets import KINOVA_GEN3_N7_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Get the base path for assets
KITCHEN_ASSETS_DIR = os.path.dirname(__file__) + "/assets"

# Tweakable offset: estimated vertical offset from shelf base to microwave base.
# Adjust this if the microwave appears intersecting or floating.
MICROWAVE_HEIGHT_OFFSET = 0.75  # increased to raise microwave above shelf (tune if needed)


@configclass
class MinimalKitchenSceneCfg(InteractiveSceneCfg):
    """Kitchen scene with proper grounding and rotation."""
    
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    
    # Robot table - properly grounded
    table = RigidObjectCfg(
        prim_path="/World/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 1.2, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,  # Ensure gravity works
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.2, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.0, 1.75, 0.4),  # Height/2 for proper grounding
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # Kinova robot on table - fixed actuators and positioning
    robot = KINOVA_GEN3_N7_CFG.replace(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            #usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Kinova/Gen3/gen3n7_instanceable.usd",
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
            pos=(2.4, 1.25, 0.8),  # On table surface
            rot=(1.0, 0.0, 0.0, 0.0),  # Facing forward
            joint_pos={
                "joint_1": 0.0,       # No base rotation
                "joint_2": -0.3,      # Shoulder down
                "joint_3": 0.0,       # No roll
                "joint_4": -1.0,      # Elbow bent
                "joint_5": 0.0,       # Wrist straight
                "joint_6": 0.7,       # Wrist up
                "joint_7": 0.0,       # No hand rotation
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
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )


    
    # Insulated shelf - FIXED: Now properly positioned above ground
    insulated_shelf = ArticulationCfg(
        prim_path="/World/InsulatedShelf",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{KITCHEN_ASSETS_DIR}/Kitchen_InsularShelf/Kitchen_InsularShelf.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(2.5, 0.0, 0.5),  # FIXED: Z=0.5 to properly sit above ground
            rot=(1.0, 0.0, 0.0, 0.0),  # No rotation
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
    
    # Microwave - ON SHELF with 180° rotation
    microwave = ArticulationCfg(
        prim_path="/World/Microwave",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{KITCHEN_ASSETS_DIR}/Microwave052/Microwave052.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # Center microwave above the insulated_shelf: match X/Y and place Z so it sits on the shelf top.
            # Assumption: shelf top is near Z ~= 1.0 in this scene; adjust if the asset has a different height.
            # Centered on shelf (X same as insulated_shelf). Z = insulated_shelf Z + offset
            pos=(2.3, 0.3, 0.5 + MICROWAVE_HEIGHT_OFFSET),  # tuned via MICROWAVE_HEIGHT_OFFSET
            rot=(0.0, 0.0, 0.0, 1.0),  # Identity quat (no rotation) - change if a 180° yaw is required
        ),
        actuators={
            "microwave_joints": ImplicitActuatorCfg(
                # Match prim/joint names seen in the USD: Microwave052_door, microjoint, Disc01_joint, etc.
                joint_names_expr=[".*door.*|.*joint.*|.*micro.*|.*Disc.*"],
                effort_limit=20.0,
                velocity_limit=0.3,
                stiffness=500.0,
                damping=50.0,
            ),
        },
    )
    
    # Fridge - BESIDE SHELF with 180° rotation
    fridge = ArticulationCfg(
        prim_path="/World/Fridge",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{KITCHEN_ASSETS_DIR}/Refrigerator036/Refrigerator036.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.5, 0.0, 0.9),  # Moved closer to insulated_shelf (Y reduced from -1.5 to -0.6)
            rot=(0.0, 0.0, 0.0, 1.0),  # 180° rotation in yaw - door faces robot (corrected)
        ),
        actuators={
            "fridge_joints": ImplicitActuatorCfg(
                # Match freezer/fridge door and drawer joint names from the USD (e.g. freezer0_door_joint,
                # fridge_door_joint, fridge_drawer0_joint)
                joint_names_expr=[".*door.*|.*joint.*|.*freezer.*|.*fridge.*"],
                effort_limit=30.0,
                velocity_limit=0.2,
                stiffness=800.0,
                damping=80.0,
            ),
        },
    )

    CONTAINER_CFG = RigidObjectCfg(
        prim_path="/World/Bowl",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{KITCHEN_ASSETS_DIR}/Lightwheel_bowl/bowl.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.6534, 0.20906, 1.29814),  # Updated to user-specified coordinates
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Enhanced lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
    )

@configclass 
class MinimalKitchenEnvCfg(ManagerBasedEnvCfg):
    """Minimal environment configuration."""
    scene: MinimalKitchenSceneCfg = MinimalKitchenSceneCfg(num_envs=1, env_spacing=5.0)
    episode_length_s = 120.0
    decimation = 2
    viewer = ViewerCfg(eye=(3.0, 3.0, 2.5), lookat=(0.0, 0.0, 1.0))