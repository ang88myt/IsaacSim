# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dockable Isaac Sim window for chatting with an LM Studio server."""

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any

import omni.ext
import omni.ui as ui
import omni.usd
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdShade


DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant for Isaac Sim and robotics development."
EXTENSION_TITLE = "LM Studio Local LLM"
DEFAULT_DEPTH_SCALE = 0.25
MAX_MESH_SIZE = 160


def _request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Send a JSON request to LM Studio using only the Python standard library."""
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


class Extension(omni.ext.IExt):
    """Provide a small UI client for an OpenAI-compatible LM Studio server."""

    def on_startup(self, ext_id: str) -> None:
        """Create the LM Studio window."""
        self._window: ui.Window | None = None
        self._request_task: asyncio.Task[None] | None = None
        self._url_model: ui.AbstractValueModel | None = None
        self._model_model: ui.AbstractValueModel | None = None
        self._system_model: ui.AbstractValueModel | None = None
        self._prompt_model: ui.AbstractValueModel | None = None
        self._response_model: ui.AbstractValueModel | None = None
        self._status_label: ui.Label | None = None
        self._send_button: ui.Button | None = None
        self._image_path_model: ui.AbstractValueModel | None = None
        self._depth_scale_model: ui.AbstractValueModel | None = None
        self._filepicker = None
        self._build_window()

    def on_shutdown(self) -> None:
        """Cancel pending work and destroy the window."""
        if self._request_task and not self._request_task.done():
            self._request_task.cancel()
        self._request_task = None
        if self._filepicker:
            self._filepicker.hide()
            self._filepicker = None
        if self._window:
            self._window.destroy()
        self._window = None

    def _build_window(self) -> None:
        self._window = ui.Window(
            EXTENSION_TITLE,
            width=560,
            height=640,
            visible=True,
            dockPreference=ui.DockPreference.LEFT_BOTTOM,
        )
        with self._window.frame:
            with ui.ScrollingFrame(
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            ):
                with ui.VStack(spacing=8, height=0):
                    ui.Label(
                        "Connect Isaac Sim to a local model running in LM Studio.",
                        word_wrap=True,
                    )
                    self._url_model = self._add_field("Base URL", DEFAULT_BASE_URL)
                    self._model_model = self._add_field("Model", "local-model")
                    ui.Button("Discover Models", clicked_fn=self._discover_models)
                    self._system_model = self._add_field("System prompt", DEFAULT_SYSTEM_PROMPT, height=80)
                    self._prompt_model = self._add_field("User prompt", "", height=120)
                    with ui.HStack(height=30, spacing=8):
                        self._send_button = ui.Button("Send", clicked_fn=self._send_prompt)
                        ui.Button("Clear response", clicked_fn=self._clear_response)
                    ui.Label("Response")
                    self._response_model = ui.StringField(
                        height=220,
                        multiline=True,
                        read_only=True,
                    ).model
                    ui.Separator()
                    ui.Label("Create 2.5D model from an image")
                    self._image_path_model = self._add_field("Image path", "")
                    with ui.HStack(height=30, spacing=8):
                        ui.Button("Choose image", clicked_fn=self._choose_image)
                        ui.Button("Create model", clicked_fn=self._create_model)
                    self._depth_scale_model = self._add_field(
                        "Depth scale (meters)",
                        str(DEFAULT_DEPTH_SCALE),
                    )
                    self._status_label = ui.Label("Ready")

    @staticmethod
    def _add_field(label: str, value: str, height: int = 28) -> ui.AbstractValueModel:
        """Create a labeled text field and return its model."""
        with ui.VStack(spacing=2, height=0):
            ui.Label(label)
            field = (
                ui.StringField(height=height, multiline=True)
                if height > 28
                else ui.StringField(height=height)
            )
            field.model.set_value(value)
            return field.model

    def _choose_image(self) -> None:
        """Open a file picker for an input image."""
        from omni.kit.widget.filebrowser import FileBrowserItem
        from omni.kit.window.filepicker import FilePickerDialog

        def filter_images(item: FileBrowserItem) -> bool:
            return item.is_folder or os.path.splitext(item.path)[1].lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
            }

        self._filepicker = FilePickerDialog(
            "Choose image",
            allow_multi_selection=False,
            apply_button_label="Use image",
            click_apply_handler=self._set_image_path,
            item_filter_fn=filter_images,
        )
        self._filepicker.show()

    def _set_image_path(self, filename: str, folder_path: str) -> None:
        """Store the selected image path."""
        self._filepicker.hide()
        self._image_path_model.set_value(os.path.join(folder_path, filename))
        self._set_status(f"Selected image: {filename}")

    def _create_model(self) -> None:
        """Create a textured relief mesh from the selected image."""
        try:
            image_path = self._image_path_model.as_string.strip()
            if not image_path:
                raise ValueError("Choose an image first.")
            if not os.path.isfile(image_path):
                raise ValueError(f"Image does not exist: {image_path}")
            depth_scale = float(self._depth_scale_model.as_string)
            if depth_scale <= 0:
                raise ValueError("Depth scale must be greater than zero.")
            self._create_relief_mesh(image_path, depth_scale)
            self._set_status("Created textured 2.5D model in /World/ImageRelief.")
        except (OSError, ValueError, Image.UnidentifiedImageError) as error:
            self._set_status(f"Error: {error}")

    @staticmethod
    def _create_relief_mesh(image_path: str, depth_scale: float) -> None:
        """Create a USD relief mesh using image luminance as depth."""
        image = Image.open(image_path).convert("RGBA")
        image.thumbnail((MAX_MESH_SIZE, MAX_MESH_SIZE), Image.Resampling.LANCZOS)
        width, height = image.size
        pixels = list(image.getdata())
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("No USD stage is open.")

        root_path = omni.usd.get_stage_next_free_path(stage, "/World/ImageRelief", True)
        mesh_path = f"{root_path}/Mesh"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        points = []
        texcoords = []
        for row in range(height):
            for column in range(width):
                red, green, blue, alpha = pixels[row * width + column]
                luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0
                opacity = alpha / 255.0
                x = (column / max(width - 1, 1) - 0.5) * 2.0
                y = (0.5 - row / max(height - 1, 1)) * 2.0 * height / max(width, 1)
                points.append(Gf.Vec3f(x, y, luminance * opacity * depth_scale))
                texcoords.append(Gf.Vec2f(column / max(width - 1, 1), 1.0 - row / max(height - 1, 1)))

        indices = []
        counts = []
        for row in range(height - 1):
            for column in range(width - 1):
                top_left = row * width + column
                top_right = top_left + 1
                bottom_left = (row + 1) * width + column
                bottom_right = bottom_left + 1
                indices.extend([top_left, bottom_left, bottom_right, top_right])
                counts.append(4)

        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(texcoords)

        material = UsdShade.Material.Define(stage, f"{root_path}/Material")
        shader = UsdShade.Shader.Define(stage, f"{root_path}/Material/Shader")
        texture = UsdShade.Shader.Define(stage, f"{root_path}/Material/Texture")
        primvar = UsdShade.Shader.Define(stage, f"{root_path}/Material/Primvar")
        shader.CreateIdAttr("UsdPreviewSurface")
        texture.CreateIdAttr("UsdUVTexture")
        primvar.CreateIdAttr("UsdPrimvarReader_float2")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(image_path))
        primvar.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectTo(primvar.CreateOutput("result", Sdf.ValueTypeNames.Float2))
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectTo(
            texture.CreateOutput("rgb", Sdf.ValueTypeNames.Color3f)
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectTo(
            texture.CreateOutput("a", Sdf.ValueTypeNames.Float)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
        material.CreateSurfaceOutput().ConnectTo(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)

    def _send_prompt(self) -> None:
        """Start a non-blocking chat completion request."""
        if self._request_task and not self._request_task.done():
            return
        self._request_task = asyncio.ensure_future(self._send_prompt_async())

    async def _send_prompt_async(self) -> None:
        """Send the current prompt and update the window."""
        self._set_busy(True, "Sending request...")
        try:
            base_url = self._url_model.as_string.strip().rstrip("/")
            model = self._model_model.as_string.strip()
            system_prompt = self._system_model.as_string
            prompt = self._prompt_model.as_string.strip()
            if not prompt:
                raise ValueError("Enter a user prompt before sending.")
            if not model:
                raise ValueError("Enter an LM Studio model identifier.")
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
            result = await asyncio.to_thread(_request_json, f"{base_url}/chat/completions", payload)
            choices = result.get("choices")
            if not choices or not choices[0].get("message", {}).get("content"):
                raise ValueError("LM Studio returned no assistant message.")
            self._response_model.set_value(choices[0]["message"]["content"])
            self._set_status("Request complete.")
        except asyncio.CancelledError:
            raise
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as error:
            self._set_status(f"Error: {error}")
        finally:
            self._set_busy(False)

    def _discover_models(self) -> None:
        """Start a non-blocking request to populate the model field."""
        if self._request_task and not self._request_task.done():
            return
        self._request_task = asyncio.ensure_future(self._discover_models_async())

    async def _discover_models_async(self) -> None:
        """Read the models exposed by LM Studio."""
        self._set_busy(True, "Discovering models...")
        try:
            base_url = self._url_model.as_string.strip().rstrip("/")
            result = await asyncio.to_thread(_request_json, f"{base_url}/models")
            models = [item.get("id") for item in result.get("data", []) if item.get("id")]
            if not models:
                raise ValueError("LM Studio returned no models.")
            self._model_model.set_value(models[0])
            self._set_status(f"Selected model: {models[0]}")
        except asyncio.CancelledError:
            raise
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as error:
            self._set_status(f"Error: {error}")
        finally:
            self._set_busy(False)

    def _clear_response(self) -> None:
        """Clear the assistant response field."""
        self._response_model.set_value("")
        self._set_status("Response cleared.")

    def _set_busy(self, busy: bool, status: str | None = None) -> None:
        """Update controls while a request is in flight."""
        if self._send_button:
            self._send_button.enabled = not busy
        if status:
            self._set_status(status)

    def _set_status(self, status: str) -> None:
        """Display an operation status."""
        if self._status_label:
            self._status_label.text = status
