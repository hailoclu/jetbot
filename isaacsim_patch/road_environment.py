import carb
import omni
import numpy as np
from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server

from pxr import UsdGeom, Gf, Sdf, PhysxSchema, PhysicsSchema, PhysicsSchemaTools
from jetbot_city.road_map import *
from jetbot_city.road_map_path_helper import *
from jetbot_city.road_map_generator import *

from omni.isaac.synthetic_utils import DomainRandomization

import math


class Environment:
    def __init__(self, omni_kit, z_height=0):
        self.omni_kit = omni_kit
        result, nucleus_server = find_nucleus_server()
        if result is False:
            carb.log_error(
                "Could not find nucleus server with /Isaac folder. Please specify the correct nucleus server in experiences/isaac-sim-python.json"
            )
            return
        result, nucleus_server = find_nucleus_server("/Library/Props/Road_Tiles/Parts/")
        if result is False:
            carb.log_error(
                "Could not find nucleus server with /Library/Props/Road_Tiles/Parts/ folder. Please refer to the documentation to aquire the road tile assets"
            )
            return
        # 1=I 2=L 3=T, 4=X
        self.tile_usd = {
            0: None,
            1: {"asset": nucleus_server + "/Library/Props/Road_Tiles/Parts/p4336p01.usd", "offset": 180},
            2: {"asset": nucleus_server + "/Library/Props/Road_Tiles/Parts/p4342p01.usd", "offset": 180},
            3: {"asset": nucleus_server + "/Library/Props/Road_Tiles/Parts/p4341p01.usd", "offset": 180},
            4: {"asset": nucleus_server + "/Library/Props/Road_Tiles/Parts/p4343p01.usd", "offset": 180},
        }  # list of tiles that can be spawned

        self.texture_list = [
            nucleus_server + "/Isaac/Samples/DR/Materials/Textures/checkered.png",
            nucleus_server + "/Isaac/Samples/DR/Materials/Textures/marble_tile.png",
            nucleus_server + "/Isaac/Samples/DR/Materials/Textures/picture_a.png",
            nucleus_server + "/Isaac/Samples/DR/Materials/Textures/picture_b.png",
            nucleus_server + "/Isaac/Samples/DR/Materials/Textures/textured_wall.png",
            nucleus_server + "/Isaac/Samples/DR/Materials/Textures/checkered_color.png",
        ]
        self.tile_size = [25.0, 25.0]

        # 1=UP, 2 = DOWN, 3 = LEFT, 4= RIGHT
        self.direction_map = {1: 180, 2: 0, 3: -90, 4: 90}

        self.prims = []  # list of spawned tiles
        self.height = z_height  # height of the ground tiles
        self.tiles = None
        self.state = None
        # because the ground plane is what the robot drives on, we only do this once. We can then re-generate the road as often as we need without impacting physics
        self.setup_physics()
        self.road_map = None
        self.road_path_helper = None
        self.map_generator = LoopRoadMapGenerator()

        contents = omni.client.list(nucleus_server + "/Isaac/Props/Sortbot_Housing/Materials/Textures/")[1]
        for entry in contents:
            self.texture_list.append(
                nucleus_server + "/Isaac/Props/Sortbot_Housing/Materials/Textures/" + entry.relative_path
            )

        contents = omni.client.list(nucleus_server + "/Isaac/Props/YCB/Axis_Aligned/")[1]
        names = []
        loaded_paths = []

        for entry in contents:
            if not entry.flags & omni.client.Flags.CAN_HAVE_CHILDREN:
                names.append(nucleus_server + "/Isaac/Props/YCB/Axis_Aligned/" + entry.relative_path)
                loaded_paths.append("/World/DR/mesh_component/mesh_" + entry.relative_path[0:-4])
        print(loaded_paths)

        self.omni_kit.create_prim("/World/Floor", "Xform")

        stage = omni.usd.get_context().get_stage()
        cubeGeom = UsdGeom.Cube.Define(stage, "/World/Floor/thefloor")
        cubeGeom.CreateSizeAttr(300)
        offset = Gf.Vec3f(75, 75, -150.1)
        cubeGeom.AddTranslateOp().Set(offset)

        # Create a sphere room so the world is not black
        self.omni_kit.create_prim("/World/Room", "Sphere", attributes={"radius": 1e3})

        prims = []
        self.dr = DomainRandomization()
        self.dr.toggle_manual_mode()
        self.dr.create_mesh_comp(prim_paths=prims, mesh_list=names, mesh_range=[1, 1])
        self.omni_kit.update(1 / 60.0)
        print("waiting for materials to load...")

        while self.omni_kit.is_loading():
            self.omni_kit.update(1 / 60.0)

        lights = []
        for i in range(5):
            prim_path = "/World/Lights/light_" + str(i)
            self.omni_kit.create_prim(
                prim_path,
                "SphereLight",
                translation=(0, 0, 200),
                rotation=(0, 0, 0),
                attributes={"radius": 10, "intensity": 1000.0, "color": (1.0, 1.0, 1.0)},
            )
            lights.append(prim_path)

        frames = 1

        # enable randomization for environment
        self.dr.create_movement_comp(prim_paths=loaded_paths, min_range=(0, 0, 15), max_range=(150, 150, 15))
        self.dr.create_rotation_comp(prim_paths=loaded_paths)
        self.dr.create_visibility_comp(prim_paths=loaded_paths, num_visible_range=(15, 15))

        self.dr.create_light_comp(light_paths=lights)
        self.dr.create_movement_comp(prim_paths=lights, min_range=(0, 0, 30), max_range=(150, 150, 30))

        self.dr.create_texture_comp(
            prim_paths=["/World/Floor"], enable_project_uvw=True, texture_list=self.texture_list
        )

        self.dr.create_color_comp(prim_paths=["/World/Room"])

    def generate_lights(self):
        prim_path = omni.kit.utils.get_stage_next_free_path(self.omni_kit.get_stage(), "/World/Env/Light", False)
        self.prims.append(prim_path)
        self.omni_kit.create_prim(
            prim_path,
            "RectLight",
            translation=(75, 75, 100),
            rotation=(0, 0, 0),
            attributes={"height": 150, "width": 150, "intensity": 2000.0, "color": (1.0, 1.0, 1.0)},
        )

    def reset(self, shape):
        # print(self.prims)
        # cmd = omni.kit.builtin.init.DeletePrimsCommand(self.prims)
        # cmd.do()
        stage = omni.usd.get_context().get_stage()
        for layer in stage.GetLayerStack():
            edit = Sdf.BatchNamespaceEdit()
            for path in self.prims:
                prim_spec = layer.GetPrimAtPath(path)
                if prim_spec is None:
                    continue
                parent_spec = prim_spec.realNameParent
                if parent_spec is not None:
                    edit.Add(path, Sdf.Path.emptyPath)
            layer.Apply(edit)

        self.prims = []
        self.generate_road(shape)
        self.dr.randomize_once()

    def generate_road(self, shape):
        self.tiles, self.state, self.road_map = self.map_generator.generate(shape)
        tiles = self.tiles
        state = self.state

        self.road_path_helper = RoadMapPathHelper(self.road_map)

        if tiles.shape != state.shape:
            print("tiles and state sizes don't match")
            return
        stage = self.omni_kit.get_stage()
        rows, cols = tiles.shape

        self.valid_tiles = []
        for x in range(0, rows):
            for y in range(0, cols):
                if tiles[x, y] != 0:
                    pos_x = x * self.tile_size[0] + 12.5
                    pos_y = y * self.tile_size[1] + 12.5
                    self.create_tile(
                        stage,
                        self.tile_usd[tiles[x, y]]["asset"],
                        Gf.Vec3d(pos_x, pos_y, self.height),
                        self.direction_map[state[x, y]] + self.tile_usd[tiles[x, y]]["offset"],
                    )

        for x in range(0, rows):
            for y in range(0, cols):
                # print(paths[x,y])
                if tiles[x, y] != 0:
                    self.valid_tiles.append([x, y])

    def generate_road_from_numpy(self, tiles, state):
        self.tiles = tiles
        self.state = state
        self.road_map = RoadMap.create_from_numpy(self.tiles, self.state)
        self.road_path_helper = RoadMapPathHelper(self.road_map)

        if tiles.shape != state.shape:
            print("tiles and state sizes don't match")
            return
        stage = self.omni_kit.get_stage()
        rows, cols = tiles.shape

        self.valid_tiles = []
        for x in range(0, rows):
            for y in range(0, cols):
                if tiles[x, y] != 0:
                    pos_x = x * self.tile_size[0] + 12.5
                    pos_y = y * self.tile_size[1] + 12.5
                    self.create_tile(
                        stage,
                        self.tile_usd[tiles[x, y]]["asset"],
                        Gf.Vec3d(pos_x, pos_y, self.height),
                        self.direction_map[state[x, y]] + self.tile_usd[tiles[x, y]]["offset"],
                    )

        for x in range(0, rows):
            for y in range(0, cols):
                # print(paths[x,y])
                if tiles[x, y] != 0:
                    self.valid_tiles.append([x, y])

    def create_tile(self, stage, path, location, rotation):
        prefix = "/World/Env/Tiles/Tile"
        prim_path = omni.kit.utils.get_stage_next_free_path(stage, prefix, False)
        self.prims.append(prim_path)
        tile_prim = stage.DefinePrim(prim_path, "Xform")
        tile_prim.GetReferences().AddReference(path)
        xform = UsdGeom.Xformable(tile_prim)
        xform_op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
        mat = Gf.Matrix4d().SetTranslate(location)
        mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation))
        xform_op.Set(mat)

    def setup_physics(self):
        stage = self.omni_kit.get_stage()
        # Add physics scene
        scene = PhysicsSchema.PhysicsScene.Define(stage, Sdf.Path("/World/Env/PhysicsScene"))
        # Set gravity vector
        scene.CreateGravityAttr().Set(Gf.Vec3f(0.0, 0.0, -981.0))
        # Set physics scene to use cpu physics
        PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/Env/PhysicsScene"))
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/World/Env/PhysicsScene")
        physxSceneAPI.CreatePhysxSceneEnableCCDAttr(True)
        physxSceneAPI.CreatePhysxSceneEnableStabilizationAttr(True)
        physxSceneAPI.CreatePhysxSceneEnableGPUDynamicsAttr(False)
        physxSceneAPI.CreatePhysxSceneBroadphaseTypeAttr("MBP")
        physxSceneAPI.CreatePhysxSceneSolverTypeAttr("TGS")
        # Create physics plane for the ground
        PhysicsSchemaTools.addGroundPlane(
            stage, "/World/Env/GroundPlane", "Z", 100.0, Gf.Vec3f(0, 0, self.height), Gf.Vec3f(1.0)
        )
        # Hide the visual geometry
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath("/World/Env/GroundPlane/geom"))
        if imageable:
            imageable.MakeInvisible()

    def get_valid_location(self):
        if self.tiles is None:
            print("cannot provide valid location until road is generated")
            return (0, 0)
        i = np.random.choice(len(self.valid_tiles), 1)[0]
        dist, point = self.road_path_helper.distance_to_path(self.valid_tiles[i])
        x, y = point
        return (x * self.tile_size[0], y * self.tile_size[1])

    # Computes an approximate forward vector based on the current spawn point and nearby valid path point
    def get_forward_direction(self, loc):
        if self.road_path_helper is not None:
            k = 100
            dists, pts = self.road_path_helper.get_k_nearest_path_points(np.array([self.get_tile_from_pose(loc)]), k)
            pointa = pts[0][0]
            pointb = pts[0][k - 1]

            if random.choice([False, True]):
                pointa, pointb = pointb, pointa
            return math.degrees(math.atan2(pointb[1] - pointa[1], pointb[0] - pointa[0]))

    # Compute the x,y tile location from the robot pose
    def get_tile_from_pose(self, pose):
        return (pose[0] / self.tile_size[0], pose[1] / self.tile_size[1])

    def distance_to_path(self, robot_pose):
        if self.road_path_helper is not None:
            distance, point = self.road_path_helper.distance_to_path(self.get_tile_from_pose(robot_pose))
            return distance * self.tile_size[0]

    def distance_to_path_in_tiles(self, robot_pose):
        if self.road_path_helper is not None:
            distance, point = self.road_path_helper.distance_to_path(self.get_tile_from_pose(robot_pose))
            return distance

    def distance_to_boundary(self, robot_pose):
        if self.road_path_helper is not None:
            distance = self.road_path_helper.distance_to_boundary(self.get_tile_from_pose(robot_pose))
            return distance * self.tile_size[0]

    def is_inside_path_boundary(self, robot_pose):
        if self.road_path_helper is not None:
            return self.road_path_helper.is_inside_path_boundary(self.get_tile_from_pose(robot_pose))
