### In this folder, most of the scripts/code for the animations used in the presentations are available. More information on how to run them please check [manim's documentation](https://docs.manim.community/en/stable/installation.html).

### Scenes will generate in /scenes folder by default. You can change any of these settings but I suggest to be careful with rendering options as above 720p30fps is quite performance heavy.  

Suggested run:
from week1/ folder,
```sh
python -m manim .\manim_scenes\$your_script_name.py$ $-your_tag$
# or
manim .\manim_scenes\$your_script_name.py$ $-your_tag$
```

For tags, you can use the following presets:
- ql: 480p15fps
- qm: 720p30fps
- qh: 1080p60fps
- qp: 1440p60fps

-qh --fps 30 is suggested, as it won't have alignment issues when converting to `gif` format either.


For light themed animations, to be used in powerpoint presentations, you can use this preset.
```py
config.background_color = "#F5F5DC"  # beige
Text.set_default(color=BLACK)
MathTex.set_default(color=BLACK)
```
