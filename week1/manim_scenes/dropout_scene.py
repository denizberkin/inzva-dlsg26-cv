import numpy as np
from manim import (
    Scene, VGroup, Square, Text, Axes, Line,
    BLUE_D, TEAL_D, GOLD_D, GRAY_E, WHITE, GRAY_B, YELLOW, BLUE_B, RED_B,
    ORIGIN, UP, DOWN, RIGHT, LEFT,
    FadeIn, FadeOut, Create, Transform
)

class DropoutScene(Scene):
    def construct(self):
        np.random.seed(42)
        
        # --- Parameters ---
        input_size, kernel_size = 5, 3
        output_size = input_size - kernel_size + 1
        dropout_rate = 0.4
        cell_size, gap = 0.35, 0.04

        # Colors
        c_input, c_kernel, c_output = BLUE_D, TEAL_D, GOLD_D
        c_drop, c_text = GRAY_E, WHITE

        # 1. Data Generation
        input_data = np.round(np.random.uniform(0, 1, (input_size, input_size)), 1)
        kernel_data = np.round(np.random.uniform(0, 1, (kernel_size, kernel_size)), 1)
        mask = np.random.choice([0, 1], size=(kernel_size, kernel_size), p=[dropout_rate, 1-dropout_rate])

        # --- Helper: Create Grid ---
        def create_visual_grid(data, color, label_text):
            rows, cols = data.shape
            squares, texts = VGroup(), VGroup()
            for i in range(rows):
                for j in range(cols):
                    sq = Square(side_length=cell_size, fill_color=color, fill_opacity=0.7)
                    sq.set_stroke(color=GRAY_B, width=1.0)
                    sq.move_to([j * (cell_size + gap), -i * (cell_size + gap), 0])
                    txt = Text(f"{data[i, j]:.1f}", font_size=10, color=c_text).move_to(sq.get_center())
                    squares.add(sq)
                    texts.add(txt)
            grid_group = VGroup(squares, texts).move_to(ORIGIN)
            label = Text(label_text, font_size=12).next_to(grid_group, UP, buff=0.15)
            return VGroup(grid_group, label), squares, texts

        # --- Helper: Create Row ---
        def create_row(is_dropout):
            title_str = "Dropout Conv2D" if is_dropout else "Standard Conv2D"
            k_eff = kernel_data * mask if is_dropout else kernel_data
            
            # Pre-calculate output
            out_data = np.zeros((output_size, output_size))
            for i in range(output_size):
                for j in range(output_size):
                    region = input_data[i:i+kernel_size, j:j+kernel_size]
                    out_data[i, j] = np.sum(region * k_eff)
            
            in_vg, in_sq, _ = create_visual_grid(input_data, c_input, "Input")
            ke_vg, ke_sq, ke_tx = create_visual_grid(kernel_data, c_kernel, "Kernel")
            ou_vg, ou_sq, ou_tx = create_visual_grid(np.round(out_data, 1), c_output, "Feature Map")
            
            row_content = VGroup(in_vg, ke_vg, ou_vg).arrange(RIGHT, buff=0.8)
            row_title = Text(title_str, font_size=16, color=YELLOW if is_dropout else BLUE_B).next_to(row_content, LEFT, buff=0.4)
            return VGroup(row_title, row_content), in_sq, ke_sq, ke_tx, ou_sq, ou_tx

        # --- Build Layout ---
        top_row, t_in_sq, t_ke_sq, t_ke_tx, t_ou_sq, t_ou_tx = create_row(False)
        bot_row, b_in_sq, b_ke_sq, b_ke_tx, b_ou_sq, b_ou_tx = create_row(True)
        
        top_row.to_edge(UP, buff=0.4).shift(RIGHT*0.5)
        bot_row.next_to(top_row, DOWN, buff=0.6)

        # --- Graph Setup ---
        x_steps = np.arange(0, output_size**2 + 1, 1)
        y_steps = np.arange(0, 5, 2)
        
        axes = Axes(
            x_range=[0, output_size**2 + 1, 1], y_range=[0, 4, 2],
            x_length=9, y_length=1.5,
            x_axis_config={
                "numbers_to_include": x_steps,
                "font_size": 12,
                "include_tip": False
            },
            y_axis_config={
                "numbers_to_include": y_steps,
                "font_size": 12,
                "include_tip": False
            },
        ).to_edge(DOWN, buff=0.5).shift(LEFT * 1)
        
        axes_labels = axes.get_axis_labels(x_label=Text("Steps", font_size=16), y_label=Text("Sum", font_size=16))
        
        # --- Legend ---
        legend_std_line = Line(start=LEFT, end=RIGHT, color=BLUE_B, stroke_width=3).scale(0.5)
        legend_std_text = Text("Standard", font_size=12, color=BLUE_B).next_to(legend_std_line, RIGHT, buff=0.1)
        
        legend_drop_line = Line(start=LEFT, end=RIGHT, color=YELLOW, stroke_width=4).scale(0.5)
        legend_drop_text = Text("Dropout", font_size=12, color=YELLOW).next_to(legend_drop_line, RIGHT, buff=0.1)
        
        legend = VGroup(
            VGroup(legend_std_line, legend_std_text),
            VGroup(legend_drop_line, legend_drop_text)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).next_to(axes, RIGHT, buff=0.3)
        
        graph_group = VGroup(axes, axes_labels, legend)

        # Graph State
        t_last_pos = axes.c2p(0, 0)
        b_last_pos = axes.c2p(0, 0)

        t_ou_tx.set_opacity(0)
        b_ou_tx.set_opacity(0)

        self.add(top_row, bot_row, graph_group)
        self.wait(1)

        # --- Animate Dropout Mask ---
        drop_indices = np.where(mask.flatten() == 0)[0]
        self.play(*(
            [b_ke_sq[idx].animate.set_fill(c_drop, 0.3) for idx in drop_indices] +
            [Transform(b_ke_tx[idx], Text("0.0", font_size=10, color=RED_B).move_to(b_ke_tx[idx])) for idx in drop_indices]
        ))
        self.wait(0.5)

        # --- Convolution Loop ---
        t_ghost = t_ke_sq.copy().set_opacity(0.4).set_stroke(YELLOW, 2)
        b_ghost = b_ke_sq.copy().set_opacity(0.4).set_stroke(YELLOW, 2)
        self.play(FadeIn(t_ghost), FadeIn(b_ghost))

        for i in range(output_size):
            for j in range(output_size):
                # 1. Calculate Indices Correctly
                out_idx = i * output_size + j     # Index for Output Grid (0 to 8)
                in_idx  = i * input_size + j      # Index for Input Grid (Top-Left corner of window)
                step = out_idx + 1

                # 2. Move Ghosts (Using in_idx for position)
                t_target = t_in_sq[in_idx].get_center() + (t_ghost.get_center() - t_ghost[0].get_center())
                b_target = b_in_sq[in_idx].get_center() + (b_ghost.get_center() - b_ghost[0].get_center())

                # 3. Calculate Graph Points (Using out_idx for data)
                t_val = float(t_ou_tx[out_idx].text)
                b_val = float(b_ou_tx[out_idx].text)
                
                t_new_pos = axes.c2p(step, t_val)
                b_new_pos = axes.c2p(step, b_val)
                t_line = Line(t_last_pos, t_new_pos, color=BLUE_B, stroke_width=3)
                b_line = Line(b_last_pos, b_new_pos, color=YELLOW, stroke_width=3)

                # Animation Sequence
                self.play(
                    t_ghost.animate.move_to(t_target),
                    b_ghost.animate.move_to(b_target),
                    run_time=0.3
                )
                
                self.play(
                    t_ou_tx[out_idx].animate.set_opacity(1),
                    t_ou_sq[out_idx].animate.set_fill(opacity=1),
                    b_ou_tx[out_idx].animate.set_opacity(1),
                    b_ou_sq[out_idx].animate.set_fill(opacity=1),
                    Create(t_line),
                    Create(b_line),
                    run_time=0.3
                )
                self.play(t_ou_sq[out_idx].animate.set_fill(opacity=0.7), b_ou_sq[out_idx].animate.set_fill(opacity=0.7), run_time=0.1)

                t_last_pos = t_new_pos
                b_last_pos = b_new_pos

        self.play(FadeOut(t_ghost), FadeOut(b_ghost))
        self.wait(2)