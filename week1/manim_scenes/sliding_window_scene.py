from manim import (
    Scene, SurroundingRectangle, Text, VGroup, MathTex, Rectangle, DecimalNumber,
    BLUE, GREEN, ORANGE, RED, PURE_RED,
    DOWN, LEFT, RIGHT, UP,
    FadeIn, FadeOut,
)


class SlidingWindowScene(Scene):
    def construct(self):
        # data
        input_data = [0.1] * 5 + [1.0] * 5 + [0.1] * 5 + [1.0] * 5 + [0.1] * 5
        filter_data = [0.1, 0.2, 0.4, 0.2, 0.1]
        conv_result = [sum(input_data[i : i + 5]) * 0.2 for i in range(len(input_data) - 4)]

        # ui setting
        bar_width = 0.35
        spacing = 0.1
        total_width = bar_width + spacing

        # adjust to change len and pos
        input_scale = 1.2
        filter_scale = 1.5
        result_scale = 2.0

        # center horizontally via offset
        # (Total bars - 1) * spacing / 2
        offset = (len(input_data) - 1) * total_width / 2

        def get_x_pos(index):
            return (index * total_width) - offset

        # row 1: input signal: top side
        input_bars = VGroup()
        for i, val in enumerate(input_data):
            b = Rectangle(
                width=bar_width,
                height=val * input_scale,
                fill_opacity=0.7,
                fill_color=BLUE,
                stroke_width=1,
            )
            lbl = DecimalNumber(val, num_decimal_places=1, font_size=14)
            b.move_to([get_x_pos(i), 2.5, 0], aligned_edge=DOWN)
            lbl.next_to(b, UP, buff=0.1)
            input_bars.add(VGroup(b, lbl))

        input_title = Text("Input Signal", font_size=20, color=BLUE).to_edge(LEFT + UP)

        # row 2 filter: middle
        filter_bars = VGroup()
        for i, val in enumerate(filter_data):
            b = Rectangle(
                width=bar_width, height=val * filter_scale, fill_opacity=0.7, fill_color=ORANGE, stroke_width=1
            )
            lbl = DecimalNumber(val, num_decimal_places=1, font_size=14)
            b.move_to([i * total_width, 0.5, 0], aligned_edge=DOWN)
            lbl.next_to(b, UP, buff=0.1)
            filter_bars.add(VGroup(b, lbl))

        filter_title = Text("Filter (w)", font_size=20, color=RED).to_edge(LEFT).shift(UP * 1.8)

        # row 3 results: bottom side
        result_title = Text("Result", font_size=20, color=GREEN).to_edge(LEFT).shift(DOWN * 1.5)
        result_group = VGroup()

        self.add(input_bars, input_title, filter_title, result_title)

        # left edge of filter group should align with the left edge of input
        filter_bars.move_to([input_bars[0:5].get_center()[0], 0.5, 0], aligned_edge=DOWN)
        filter_highlight = SurroundingRectangle(
            filter_bars, color=PURE_RED, buff=0.2, corner_radius=0.2, stroke_width=2
        )
        filter_notice_sum = (
            MathTex(r"\sum_i y_i = 1", font_size=24, color=PURE_RED).next_to(filter_highlight, RIGHT, buff=0.5)
            # .next_to(filter_title, RIGHT, buff=1.0)
        )

        self.play(FadeIn(filter_bars), FadeIn(filter_highlight), FadeIn(filter_notice_sum), offset=0.5)
        self.wait(2)
        self.play(FadeOut(filter_notice_sum), FadeOut(filter_highlight))
        # add yellow mathtex, sum of i y_i = 1

        for i in range(len(conv_result)):
            # alignment again
            current_window = input_bars[i : i + 5]
            target_x = current_window.get_center()[0]

            # Create Result Bar
            res_val = conv_result[i]
            r_bar = Rectangle(
                width=bar_width,
                height=max(res_val * result_scale, 0.05),
                fill_opacity=0.8,
                fill_color=GREEN,
                stroke_width=1,
            )
            r_label = DecimalNumber(res_val, num_decimal_places=2, font_size=14)

            # result position??
            res_unit = VGroup(r_bar, r_label)
            res_unit.move_to([target_x, -2.5, 0], aligned_edge=DOWN)
            r_label.next_to(r_bar, UP, buff=0.1)

            if i == 0:
                self.play(FadeIn(res_unit))
            else:
                self.play(filter_bars.animate.set_x(target_x), FadeIn(res_unit), run_time=0.25)
            result_group.add(res_unit)

        self.wait(2)
