## Here is the most of the scripts/code for the animations used in the presentations. More information on how to run them please check [manim's documentation](https://docs.manim.community/en/stable/installation.html).

### Scenes will generate in /scenes folder by default. You can change any of these settings but I suggest to be careful with rendering options as above 720p30fps is quite performance heavy.  

Suggested run:
from week1/ folder,
`python -m manim $your_script_name.py$ $-your_tag$`
`manim .\manim_scenes\$your_script_name.py$ $-your_tag$`
for tag:
-ql: 480p15fps
-qm: 720p30fps
-qh: 1080p60fps
-qp: 1440p60fps

-qh --fps 30 is suggested, as it won't have alignment issues when converting to `gif` format either.
