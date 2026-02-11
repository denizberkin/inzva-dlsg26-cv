import numpy as np
from manim import (
    Scene, VGroup, Axes, Text, MathTex, DecimalNumber,
    Rectangle, LaggedStart, Create, GrowFromEdge, Write,
    BLUE, TEAL, YELLOW, GRAY, RED, RED_D, MAROON,
    UP, DOWN, RIGHT, ORIGIN,
)

class ActivationScene(Scene):
    def construct(self):
        relu_group = self.create_graph_quadrant(
            function=lambda x: np.maximum(0, x),
            title="ReLU",
            formula=r"f(x) = \max(0, x)",
            x_range=[-4, 4],
            y_range=[-1, 4],
            color=BLUE
        )

        leaky_relu_group = self.create_graph_quadrant(
            function=lambda x: np.maximum(0.15 * x, x),
            title="Leaky ReLU",
            formula=r"f(x) = \max(0.1x, x)",
            x_range=[-4, 4],
            y_range=[-2, 4],
            color=TEAL
        )

        sigmoid_group = self.create_graph_quadrant(
            function=lambda x: 1 / (1 + np.exp(-x)),
            title="Sigmoid",
            formula=r"\sigma(x) = \frac{1}{1+e^{-x}}",
            x_range=[-6, 6],
            y_range=[-0.5, 1.5],
            color=YELLOW
        )

        softmax_group = self.create_softmax_visualization()

        # Group top row and bottom row
        top_row = VGroup(relu_group, leaky_relu_group).arrange(RIGHT, buff=1.5)
        bottom_row = VGroup(sigmoid_group, softmax_group).arrange(RIGHT, buff=1.5)
        
        # Group everything
        whole_grid = VGroup(top_row, bottom_row).arrange(DOWN, buff=1)
        
        # Scale to fit screen
        whole_grid.scale_to_fit_height(7.0)
        whole_grid.move_to(ORIGIN)
        
        # Animation, draw axes, titles
        self.play(
            LaggedStart(
                *[Create(g[0]) for g in [relu_group, leaky_relu_group, sigmoid_group, softmax_group]],
                *[Write(g[2]) for g in [relu_group, leaky_relu_group, sigmoid_group, softmax_group]],
                lag_ratio=0.1
            ),
            run_time=2
        )
        
        self.wait(0.5)

        # draw curves, bars, labels
        self.play(
            Create(relu_group[1]),
            Create(leaky_relu_group[1]),
            Create(sigmoid_group[1]),
            Write(softmax_group[3]),
            LaggedStart(
                *[GrowFromEdge(bar, DOWN) for bar in softmax_group[1]],
                lag_ratio=0.2
            ),
            run_time=2.5
        )

        # write formulas
        self.play(
            Write(relu_group[3]),
            Write(leaky_relu_group[3]),
            Write(sigmoid_group[3]),
            Write(softmax_group[4]),
            run_time=2
        )

        self.wait(3)

    def create_graph_quadrant(self, function, title, formula, x_range, y_range, color):
        """Helper to create a standard function plot with axes."""
        
        axes = Axes(
            x_range=x_range,
            y_range=y_range,
            x_length=4,
            y_length=3,
            axis_config={"include_tip": False, "color": GRAY},
        )
        
        graph = axes.plot(function, color=color, stroke_width=4)
        
        title_text = Text(title, font_size=24).next_to(axes, UP)
        formula_text = MathTex(formula, font_size=20, color=color).next_to(axes, DOWN)
        return VGroup(axes, graph, title_text, formula_text)

    def create_softmax_visualization(self):
        """Helper to visualize Softmax as Logits -> Probabilities."""
        logits = np.array([2.0, 1.0, 0.1])
        exps = np.exp(logits)
        probs = exps / np.sum(exps)
        
        bar_width = 0.5
        axes = Axes(
            x_range=[0, 4], 
            y_range=[0, 3], 
            x_length=4, 
            y_length=3,
            axis_config={"include_ticks": False, "color": GRAY}
        )
        bars = VGroup()
        labels = VGroup()
        
        colors = [RED, RED_D, MAROON]
        for i, prob in enumerate(probs):
            bar = Rectangle(
                width=bar_width, 
                height=prob * 2.5,
                fill_color=colors[i], 
                fill_opacity=0.8,
                stroke_width=0
            )
            bar.move_to(axes.c2p(i + 1, prob * 2.5 / 2))
            
            val_label = DecimalNumber(prob, num_decimal_places=2, font_size=16)
            val_label.next_to(bar, UP, buff=0.1)
            
            bars.add(bar)
            labels.add(val_label)
            
        title_text = Text("Softmax", font_size=24).next_to(axes, UP)
        formula_text = MathTex(
            r"\frac{e^{z_i}}{\sum e^{z_j}}", 
            font_size=24, 
            color=RED
        ).next_to(axes, DOWN)
        
        return VGroup(axes, bars, title_text, labels, formula_text)