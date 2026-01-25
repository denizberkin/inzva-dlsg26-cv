from manim import BLACK, MathTex, Scene, Text, Write, config

config.background_color = "#F5F5DC"  # beige
Text.set_default(color=BLACK)
MathTex.set_default(color=BLACK)


class ConvScene(Scene):
    def construct(self):
        Text.set_default(color=BLACK)
        # Note: double backslashes are required in Python strings for LaTeX commands
        formula = MathTex("(f * g)(t) = \\int_{-\\infty}^{\\infty} f(\\tau)g(t - \\tau) d\\tau")
        formula.scale(1.5)
        self.play(Write(formula), run_time=2)
        self.wait(1)
