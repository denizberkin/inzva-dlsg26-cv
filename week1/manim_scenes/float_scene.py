import random

from manim import (
    BLACK,
    BLUE,
    DOWN,
    LEFT,
    RED,
    RIGHT,
    UP,
    Create,
    FadeIn,
    FadeOut,
    MathTex,
    Scene,
    Square,
    Text,
    Transform,
    VGroup,
    config,
)

# config.background_color = "#F5F5DC"  # beige
# Text.set_default(color=BLACK)
# MathTex.set_default(color=BLACK)


class FloatScene(Scene):
    def construct(self):
        square_size, spacing = 0.8, 0.1
        step_width = square_size + spacing
        text_font_size = 24
        label_font_size = 24

        def create_dice_row(values, color, name: str = "p"):
            row = []
            for i, v in enumerate(values):
                sq = Square(side_length=square_size, color=color)
                sq.set_fill(color=color, opacity=v)

                val_text = Text(f"{v:.2f}", font_size=text_font_size)

                # The p_i label at the top center
                # We use i+1 so it starts at p_1
                p_label = MathTex(f"{name}_{{{i + 1}}}", font_size=label_font_size, color=color)
                p_label.next_to(sq, UP, buff=0.05)  # Positioned just above the square

                # group square text and label
                row.append(VGroup(sq, val_text, p_label))

            return VGroup(*row).arrange(RIGHT, buff=spacing)

        vals_a = [random.random() for _ in range(6)]
        vals_b = [random.random() for _ in range(6)]
        dice_a = create_dice_row(vals_a, BLUE, name="a").move_to(UP * 0.7)
        dice_b = create_dice_row(vals_b, RED, name="b").next_to(dice_a, DOWN, buff=0.5)
        dice_a_label = Text("Data A", font_size=20).next_to(dice_a, LEFT)
        dice_b_label = Text("Data B", font_size=20).next_to(dice_b, LEFT)
        self.add(dice_a, dice_b, dice_a_label, dice_b_label)
        self.wait(1)

        # step 2: reverse
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

        self.play(FadeOut(dice_b_label, dice_a_label), run_time=0.5)
        self.play(dice_b.animate.shift(LEFT * step_width * 6))
        self.wait(1.0)

        formula_display = MathTex("", font_size=30).to_edge(UP, buff=1.0)
        highlights = VGroup()
        self.add(formula_display)
        self.add(highlights)

        self.play(FadeIn(formula_display), Create(highlights))

        for t in range(0, 11):
            terms = []
            indices = [i - (t - 5) for i in range(6)]
            for i, j in enumerate(indices):
                if 0 <= j < 6:
                    b_label_num = 6 - j
                    term_str = rf"a_{{{i + 1}}} \cdot b_{{{b_label_num}}}"
                    terms.append(term_str)

            formula_content = " + ".join(terms) if terms else ""
            formula_str = f"y({t+2}) = " + formula_content
            new_formula = MathTex(formula_str, font_size=26).to_edge(UP, buff=1.0)

            self.play(dice_b.animate.shift(RIGHT * step_width), Transform(formula_display, new_formula), run_time=1.0)

            self.wait(1.5)
