from manim import (
    BLACK,
    BLUE,
    DOWN,
    LEFT,
    RED,
    RIGHT,
    UP,
    FadeOut,
    MathTex,
    Scene,
    Square,
    Text,
    Transform,
    VGroup,
    Write,
    config,
)

config.background_color = "#F5F5DC"  # beige
Text.set_default(color=BLACK)
MathTex.set_default(color=BLACK)


class DiceScene(Scene):
    def construct(self):
        # step 1
        square_size, spacing = 0.8, 0.1
        step_width = square_size + spacing
        text_font_size = 24

        def create_dice_row(values, color):
            row = VGroup(
                *[
                    VGroup(Square(side_length=square_size, color=color), Text(str(v), font_size=text_font_size))
                    for v in values
                ]
            ).arrange(RIGHT, buff=spacing)
            return row

        dice_a = create_dice_row(range(1, 7), BLUE).move_to(UP * 0.7)
        dice_b = create_dice_row(range(1, 7), RED).next_to(dice_a, DOWN, buff=0.5)
        dice_a_label = Text("Dice A", font_size=20).next_to(dice_a, LEFT)
        dice_b_label = Text("Dice B", font_size=20).next_to(dice_b, LEFT)
        self.add(dice_a, dice_b, dice_a_label, dice_b_label)
        self.wait(1)

        # step 2: reverse dice b
        original_center = dice_b.get_center()

        reversed_cells = list(reversed(dice_b.submobjects))

        target_group = VGroup(*[cell.copy() for cell in reversed_cells])
        target_group.arrange(RIGHT, buff=spacing)
        target_group.move_to(original_center)

        self.play(
            *[dice_b.submobjects[i].animate.move_to(target_group[5 - i].get_center()) for i in range(6)], run_time=1.5
        )
        dice_b.submobjects.reverse()
        self.wait(2)

        # step 3: sliding window
        self.play(FadeOut(dice_b_label, dice_a_label), run_time=0.5)
        self.play(dice_b.animate.shift(LEFT * step_width * 5))
        self.wait(0.5)

        prob_formula = MathTex("P(A + B = n) = \\frac{\\text{number of ways}}{36}", font_size=42)
        prob_formula.to_edge(UP, buff=1.0)
        self.play(Write(prob_formula), run_time=0.5)
        self.wait(1.0)

        for n in range(2, 13):
            ways = 6 - abs(n - 7)
            formula = MathTex(f"P(A + B = {n}) = \\frac{{{ways}}}{{36}}", font_size=42).move_to(
                prob_formula.get_center()
            )
            animations = [
                Transform(prob_formula, formula),
            ]

            if n > 2:
                animations.append(dice_b.animate.shift(RIGHT * step_width))

            self.play(*animations, run_time=0.5)
        self.wait(2)

        # reset pos
        self.play(dice_b.animate.move_to(dice_a.get_center() + DOWN * 1.2), FadeOut(prob_formula), run_time=1.0)

        # reverse cells
        original_cells = list(reversed(dice_b.submobjects))
        target_reset = VGroup(*[cell.copy() for cell in original_cells])
        target_reset.arrange(RIGHT, buff=spacing)
        target_reset.move_to(dice_b.get_center())

        self.play(
            *[dice_b.submobjects[i].animate.move_to(target_reset[5 - i].get_center()) for i in range(6)], run_time=1.0
        )

        self.wait(2)
