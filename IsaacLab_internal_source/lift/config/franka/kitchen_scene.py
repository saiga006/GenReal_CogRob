import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import spawners

import os
import math

# Import the pre-configured Franka configuration
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Get the base path for assets (using the Kinova assets for kitchen items)
KITCHEN_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "kinova", "assets")

# Tweakable offset: estimated vertical offset from shelf base to microwave base.
# Adjust this if the microwave appears intersecting or floating.
MICROWAVE_HEIGHT_OFFSET = 0.75  # increased to raise microwave above shelf (tune if needed)


@configclass
class MinimalKitchenSceneCfg(InteractiveSceneCfg):
    """Kitchen scene with proper grounding and rotation for Franka robot."""
    
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
            pos=(2.75, 1.75, 0.4),  # Height/2 for proper grounding
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # Franka robot on table - fixed actuators and positioning for IK movements
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(2.15, 1.25, 0.8),  # On table surface
            rot=(1.0, 0.0, 0.0, 0.0),  # Facing forward
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.8,
                "panda_joint3": 0.0,
                "panda_joint4": -2.3,
                "panda_joint5": 0.0,
                "panda_joint6": 1.5,
                "panda_joint7": 0.8,
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
        )
    )
    
    # Insulated shelf - positioned above ground
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
            pos=(2.5, 0.5, 0.5),  # Z=0.5 to properly sit above ground
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
    
    # Microwave - ON SHELF
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
            # Center microwave above the insulated_shelf
            pos=(2.3, 0.5, 0.5 + MICROWAVE_HEIGHT_OFFSET),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "microwave_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*door.*|.*joint.*|.*micro.*|.*Disc.*"],
                effort_limit=20.0,
                velocity_limit=0.3,
                stiffness=500.0,
                damping=50.0,
            ),
        },
    )
    
    # Fridge - BESIDE SHELF
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
            pos=(1.5, 0.75, 0.9),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "fridge_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*door.*|.*joint.*|.*freezer.*|.*fridge.*"],
                effort_limit=30.0,
                velocity_limit=0.2,
                stiffness=800.0,
                damping=80.0,
            ),
        },
    )

    CONTAINER_CFG = RigidObjectCfg(
        prim_path="/World/TomatoSoupCan",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{KITCHEN_ASSETS_DIR}/Lightwheel_tomato_soup_can/tomato_soup_can.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.6534, 0.85, 1.4814),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Enhanced lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
    )
