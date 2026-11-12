# Pictures (Stage 45)

`data/pictures.npz` holds 48 x 48 renderings (random size, position, rotation, mirror image, brightness) of 34
things (`concepts.json`: name, Unicode code point, keywords used to find the thing in the primer's picture
descriptions), each in four emoji styles:

| style | source | licence |
|---|---|---|
| openmoji | OpenMoji, github.com/hfg-gmuend/openmoji (color/618x618) | CC BY-SA 4.0 - the renderings derived from it are shared under the same licence |
| twemoji | Twemoji, github.com/twitter/twemoji (assets/72x72) | CC BY 4.0 |
| noto | Noto Emoji, github.com/googlefonts/noto-emoji (2D/png/128) | Apache 2.0 |
| fluent3d | Fluent Emoji, github.com/microsoft/fluentui-emoji (3D) | MIT |

Rendering: `scripts/make_pictures.py` (source files fetched by sparse checkout of the repositories above).
