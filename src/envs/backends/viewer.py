"""Real-time rendering and multi-robot visualization for envs.

Provides a `ViewerMixin` that adds viewer management, multi-robot XML
scene construction, and offscreen trajectory rendering to any env.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Sequence

import mujoco
import numpy as np

from envs.state import State


class ViewerMixin:
    """Mixin that adds real-time rendering, multi-robot view, and offscreen rendering.

    Hosts the following attributes on the host class:
    - _enable_viewer, _viewer
    - _num_render_envs, _render_model, _render_data
    - _render_grid_cols, _render_spacing, _model_path (optional)
    - _mj_model, _mj_data, _dt (provided by env)

    Public API:
    - enable_viewer(enable=True, num_render_envs=1)
    - render(state=None, states=None, sync_frame_time=True)
    - render_trajectory(trajectory, ...)
    - close_viewer()
    """

    def enable_viewer(self, enable: bool = True, num_render_envs: int = 1):
        """Enable or disable real-time viewer."""
        self._enable_viewer = enable
        self._num_render_envs = num_render_envs

        if enable and self._viewer is None:
            try:
                from mujoco import viewer as mj_viewer
            except ImportError:
                print("Warning: mujoco.viewer not available.")
                print("Please upgrade MuJoCo: pip install mujoco>=3.0.0")
                self._enable_viewer = False
                return

            if num_render_envs > 1:
                self._create_multi_robot_scene(num_render_envs)

            render_model = getattr(self, "_render_model", self._mj_model)
            render_data = getattr(self, "_render_data", self._mj_data)

            try:
                self._viewer = mj_viewer.launch_passive(
                    render_model,
                    render_data,
                    show_left_ui=True,
                    show_right_ui=True,
                )
                actual_num = num_render_envs if hasattr(self, "_render_model") \
                    and self._render_model != self._mj_model else 1
                print(f"Viewer launched with {actual_num} robot(s). "
                      f"Use mouse to rotate view.")
            except Exception as e:
                print(f"Warning: Failed to launch viewer: {e}")
                self._enable_viewer = False

    def _create_multi_robot_scene(self, num_envs: int):
        """Create a multi-robot MuJoCo scene via XML duplication."""
        import tempfile
        import xml.etree.ElementTree as ET

        num_cols = int(np.ceil(np.sqrt(num_envs)))
        num_rows = int(np.ceil(num_envs / num_cols))
        spacing = 2.0

        if hasattr(self, "_model_path") and self._model_path:
            model_path = self._model_path
        else:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../resources/xg/xgb_all_sphere_0128_v2.xml",
            )
            model_path = os.path.abspath(model_path)

        print(f"Creating multi-robot scene: {num_cols}x{num_rows} = {num_envs} robots")

        tree = ET.parse(model_path)
        root = tree.getroot()

        # Fix mesh path → absolute
        compiler = root.find("compiler")
        if compiler is not None and compiler.get("meshdir"):
            relative_meshdir = compiler.get("meshdir")
            xml_dir = os.path.dirname(model_path)
            absolute_meshdir = os.path.abspath(os.path.join(xml_dir, relative_meshdir))
            compiler.set("meshdir", absolute_meshdir)
            print(f"  meshdir: {absolute_meshdir}")

        worldbody = root.find("worldbody")

        for light in worldbody.findall("light"):
            if light.get("mode") == "targetbodycom":
                light.set("mode", "fixed")
                light.attrib.pop("target", None)
                light.set("dir", "0 0.5 -1")
                print(f"  Fixed light '{light.get('name')}' to use fixed direction")

        robot_body = None
        for child in worldbody:
            if child.tag == "body" and child.get("name") != "floor":
                robot_body = child
                break

        if robot_body is None:
            print("Warning: Could not find robot body in XML, using single robot")
            self._render_model = self._mj_model
            self._render_data = self._mj_data
            return

        original_robot = robot_body
        robot_name = original_robot.get("name", "robot")
        worldbody.remove(original_robot)

        for i in range(num_envs):
            row = i // num_cols
            col = i % num_cols
            x_offset = col * spacing - (num_cols - 1) * spacing / 2
            y_offset = row * spacing - (num_rows - 1) * spacing / 2

            import copy
            new_robot = copy.deepcopy(original_robot)
            new_robot.set("name", f"{robot_name}_{i}")

            pos = new_robot.get("pos", "0 0 0.45").split()
            new_pos = f"{float(pos[0]) + x_offset} {float(pos[1]) + y_offset} {pos[2]}"
            new_robot.set("pos", new_pos)

            self._rename_elements(new_robot, f"_{i}")
            worldbody.append(new_robot)

        actuator = root.find("actuator")
        if actuator is not None:
            original_motors = list(actuator)
            for motor in original_motors:
                actuator.remove(motor)

            for i in range(num_envs):
                for motor in original_motors:
                    import copy
                    new_motor = copy.deepcopy(motor)
                    name = new_motor.get("name", "motor")
                    new_motor.set("name", f"{name}_{i}")
                    joint = new_motor.get("joint", "")
                    if joint:
                        new_motor.set("joint", f"{joint}_{i}")
                    actuator.append(new_motor)

        sensor = root.find("sensor")
        if sensor is not None:
            root.remove(sensor)
            print("  Removed sensor section for multi-robot rendering")

        keyframe = root.find("keyframe")
        if keyframe is not None:
            root.remove(keyframe)
            print("  Removed keyframe section for multi-robot rendering")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            tree.write(f, encoding="unicode")
            temp_path = f.name

        try:
            self._render_model = mujoco.MjModel.from_xml_path(temp_path)
            self._render_data = mujoco.MjData(self._render_model)
            self._num_render_envs = num_envs
            print(f"Created multi-robot scene: {num_cols}x{num_rows} grid, "
                  f"spacing={spacing}m")
        except Exception as e:
            print(f"Warning: Failed to create multi-robot scene: {e}")
            print("Falling back to single robot view")
            self._render_model = self._mj_model
            self._render_data = self._mj_data
            self._num_render_envs = 1
        finally:
            os.unlink(temp_path)

        self._render_grid_cols = num_cols
        self._render_spacing = spacing

    def _rename_elements(self, element, suffix):
        """Recursively rename elements in XML tree."""
        if element.get("name"):
            element.set("name", element.get("name") + suffix)
        if element.get("joint"):
            element.set("joint", element.get("joint") + suffix)
        if element.get("body"):
            element.set("body", element.get("body") + suffix)
        if element.get("site"):
            element.set("site", element.get("site") + suffix)
        for child in element:
            self._rename_elements(child, suffix)

    def render(self, state=None, states=None, sync_frame_time: bool = True):
        """Real-time render. Pass `state` (single) or `states` (batched)."""
        if not self._enable_viewer:
            return

        if self._viewer is None:
            self.enable_viewer(True)

        if not self._viewer.is_running():
            print("Viewer closed by user.")
            self._enable_viewer = False
            self._viewer = None
            return

        num_render = getattr(self, "_num_render_envs", 1)

        if num_render > 1 and states is not None:
            self._update_multi_robot_scene(states)
        elif states is not None:
            qpos = np.array(states.data.qpos)
            qvel = np.array(states.data.qvel)
            if qpos.ndim > 1:
                qpos = qpos[0]
                qvel = qvel[0]
            self._mj_data.qpos[:] = qpos
            self._mj_data.qvel[:] = qvel
            mujoco.mj_forward(self._mj_model, self._mj_data)
            self._update_camera_follow(qpos[:3])
        elif state is not None:
            self._mj_data.qpos[:] = np.array(state.data.qpos)
            self._mj_data.qvel[:] = np.array(state.data.qvel)
            mujoco.mj_forward(self._mj_model, self._mj_data)
            robot_pos = np.array(state.data.qpos[:3])
            self._update_camera_follow(robot_pos)

        self._viewer.sync()

        if sync_frame_time:
            time.sleep(self._dt)

    def _update_camera_follow(self, robot_pos: np.ndarray):
        """Set viewer camera to look at the robot."""
        if self._viewer is None:
            return
        cam = self._viewer.cam
        cam.lookat[0] = robot_pos[0]
        cam.lookat[1] = robot_pos[1]
        cam.lookat[2] = robot_pos[2] + 0.3
        cam.distance = 3.0
        cam.azimuth = 135
        cam.elevation = -20

    def _update_multi_robot_scene(self, states):
        """Update multi-robot render data from batched states."""
        if not hasattr(self, "_render_data") or self._render_data is None:
            return

        num_render = min(self._num_render_envs, states.data.qpos.shape[0])
        qpos_per_robot = 19
        qvel_per_robot = 18

        qpos_batch = np.array(states.data.qpos)
        qvel_batch = np.array(states.data.qvel)

        for i in range(num_render):
            row = i // self._render_grid_cols
            col = i % self._render_grid_cols
            x_offset = col * self._render_spacing - (self._render_grid_cols - 1) * self._render_spacing / 2
            y_offset = row * self._render_spacing - (int(np.ceil(num_render / self._render_grid_cols)) - 1) * self._render_spacing / 2

            qpos_i = qpos_batch[i].copy()
            qvel_i = qvel_batch[i].copy()
            qpos_i[0] += x_offset
            qpos_i[1] += y_offset

            start_qpos = i * qpos_per_robot
            start_qvel = i * qvel_per_robot
            self._render_data.qpos[start_qpos:start_qpos + qpos_per_robot] = qpos_i
            self._render_data.qvel[start_qvel:start_qvel + qvel_per_robot] = qvel_i

        mujoco.mj_forward(self._render_model, self._render_data)

    def render_trajectory(
        self,
        trajectory: List[State],
        camera: Optional[str] = None,
        width: int = 640,
        height: int = 480,
    ) -> Sequence[np.ndarray]:
        """Render a trajectory to a list of images."""
        renderer = mujoco.Renderer(self._mj_model, height=height, width=width)
        camera_id = camera if camera is not None else -1

        images = []
        for state in trajectory:
            self._mj_data.qpos[:] = np.array(state.data.qpos)
            self._mj_data.qvel[:] = np.array(state.data.qvel)
            mujoco.mj_forward(self._mj_model, self._mj_data)
            renderer.update_scene(self._mj_data, camera=camera_id)
            images.append(renderer.render())

        renderer.close()
        return images

    def render_frame(
        self,
        state: State,
        width: int = 1280,
        height: int = 720,
        camera_mode: str = "tracking",
        track_body_id: int = 1,
        distance: float = 1.5,
        elevation: float = -20.0,
        azimuth: float = 135.0,
        env_index: int = 0,
    ) -> np.ndarray:
        """Offscreen-render a single State frame.

        Caches the offscreen renderer and camera on the env instance so this is
        cheap to call once per step. Handles both single-env states and vmap'd
        batched states (uses ``env_index`` to pick which env to render).

        Args:
            state: A ``State`` whose ``data.qpos/qvel`` may be 1D (single env)
                or 2D ``(num_envs, ...)`` (vmap'd batch).
            width / height: Frame resolution.
            camera_mode: ``"tracking"`` attaches the camera to ``track_body_id``
                and follows the robot (good for video); ``"free"`` uses the
                MuJoCo free camera (camera id -1).
            track_body_id / distance / elevation / azimuth: Tracking camera params.
            env_index: Which env to render when state is batched.

        Returns:
            RGB image as a ``np.ndarray`` with shape ``(height, width, 3)``.
        """
        # Lazy-init renderer (resolution baked in; recreate if it changes).
        existing = getattr(self, "_offscreen_renderer", None)
        existing_size = getattr(self, "_offscreen_size", None)
        if existing is None or existing_size != (width, height):
            if existing is not None:
                try:
                    existing.close()
                except Exception:
                    pass
            self._offscreen_renderer = mujoco.Renderer(
                self._mj_model, width=width, height=height,
            )
            self._offscreen_size = (width, height)

        # Lazy-init camera (mode + tracking target baked in; recreate if changed).
        existing_cam = getattr(self, "_offscreen_camera", None)
        cam_signature = (camera_mode, track_body_id, distance, elevation, azimuth)
        if existing_cam is None or getattr(self, "_offscreen_cam_sig", None) != cam_signature:
            cam = mujoco.MjvCamera()
            if camera_mode == "tracking":
                cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = track_body_id
                cam.distance = distance
                cam.elevation = elevation
                cam.azimuth = azimuth
            else:
                # Free camera: leave defaults; renderer.update_scene with camera=-1
                # would also work but keeping an MjvCamera makes the API uniform.
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.distance = distance
                cam.elevation = elevation
                cam.azimuth = azimuth
            self._offscreen_camera = cam
            self._offscreen_cam_sig = cam_signature

        # Pull qpos/qvel for the requested env. Works for both single (1D) and
        # batched (2D) State.data.
        qpos = np.asarray(state.data.qpos)
        qvel = np.asarray(state.data.qvel)
        if qpos.ndim > 1:
            qpos = qpos[env_index]
            qvel = qvel[env_index]

        self._mj_data.qpos[:] = qpos
        self._mj_data.qvel[:] = qvel
        mujoco.mj_forward(self._mj_model, self._mj_data)

        self._offscreen_renderer.update_scene(self._mj_data, camera=self._offscreen_camera)
        return self._offscreen_renderer.render()

    def close_viewer(self):
        """Close the viewer window."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
            self._enable_viewer = False
            print("Viewer closed.")

        offscreen = getattr(self, "_offscreen_renderer", None)
        if offscreen is not None:
            try:
                offscreen.close()
            except Exception:
                pass
            self._offscreen_renderer = None
            self._offscreen_camera = None
            self._offscreen_size = None
            self._offscreen_cam_sig = None
