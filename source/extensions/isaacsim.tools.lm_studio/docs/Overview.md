# LM Studio Local LLM

The `isaacsim.tools.lm_studio` extension provides a dockable Isaac Sim window for
calling a local language model hosted by [LM Studio](https://lmstudio.ai/).

## Setup

1. Start LM Studio and load a model.
2. Start the LM Studio local server using its default OpenAI-compatible API.
3. Enable **LM Studio Local LLM** in Isaac Sim.
4. Open the window from the extension menu.
5. Enter a model identifier, or click **Discover Models**.
6. Enter a prompt and click **Send**.

The default endpoint is `http://127.0.0.1:1234/v1`. The extension uses Python's
standard library, so no extra package installation is required.

## API

The extension calls:

- `GET /v1/models` to discover available models.
- `POST /v1/chat/completions` to generate a response.

Requests run asynchronously so the Isaac Sim UI remains responsive while the
local model generates a response.

## Image to 3D

The **Choose image** and **Create model** controls generate a textured 2.5D
relief mesh in the current USD stage. The image luminance controls the relief
depth and the image alpha controls transparency. This is useful for quickly
bringing a concept image, product image, or robot reference into a scene, but
it is not a watertight full 3D reconstruction from a single view.

The supplied robot image can be selected directly from the file picker. Adjust
**Depth scale (meters)** to change the relief thickness.
